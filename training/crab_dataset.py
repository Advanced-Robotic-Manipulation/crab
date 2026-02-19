"""
Dataset loader for Crab robot LeRobot v3.0 format.
Handles: parquet data, video frames (AV1), tactile matrices, multi-episode.

v4: Disk-backed video cache using numpy memmap.
    Pre-decodes videos to .npy files on disk, then mmap's for O(1) random access.
    RAM usage: ~0 for video frames (OS pages in/out as needed).
    Disk usage: ~60GB for 100K frames at 256x256.

v3: Per-dataset task descriptions from tasks.parquet (no more single global prompt).
    Supports action_indices/state_indices for training on subset of joints.
    Fixed video frame indexing (uses global index, not per-episode index).
    uint8 video cache (4x RAM savings), shared_video_cache between train/val.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Default disk cache location
VIDEO_CACHE_DIR = Path("/tmp/crab_video_cache")


class CrabEpisodeDataset(Dataset):
    """
    Dataset for Crab bimanual robot in LeRobot v3.0 format.

    Each sample returns:
        - images: dict of {cam_name: [3, H, W] tensor} (3 cameras)
        - state: [state_dim] tensor (joint positions, optionally sliced)
        - tactile_left: [100] tensor (10x10 flattened)
        - tactile_right: [100] tensor (10x10 flattened)
        - action: [chunk_size, action_dim] tensor (action chunk, optionally sliced)
        - task: str (text instruction, per-dataset from tasks.parquet)
        - metadata: dict (episode_index, frame_index, timestamp)
    """

    def __init__(
        self,
        episodes_dir: str | Path,
        episode_names: list[str],
        chunk_size: int = 50,
        image_size: tuple[int, int] = (256, 256),
        fps: int = 15,
        task_description: str = "Pick and place object",
        split: str = "train",
        val_ratio: float = 0.1,
        image_keys: list[str] | None = None,
        action_indices: list[int] | None = None,
        state_indices: list[int] | None = None,
        transform=None,
        shared_video_cache: dict | None = None,
        video_cache_dir: str | Path | None = None,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.fps = fps
        self.task_description = task_description  # fallback only
        self.split = split
        self.transform = transform

        # Disk cache directory for decoded video frames
        self._cache_dir = Path(video_cache_dir) if video_cache_dir else VIDEO_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Optional joint subset selection
        self.action_indices = action_indices
        self.state_indices = state_indices
        if action_indices is not None:
            logger.info(f"Action indices: {action_indices} ({len(action_indices)} dims)")
        if state_indices is not None:
            logger.info(f"State indices: {state_indices} ({len(state_indices)} dims)")

        self.image_keys = image_keys or [
            "observation.images.main_camera",
            "observation.images.left_arm_camera",
            "observation.images.right_arm_camera",
        ]

        # Load all episodes
        self.samples = []
        self.episode_boundaries = []  # (start_idx, end_idx) per episode
        # Video cache: maps video_path -> (memmap_array, n_frames)
        self._video_cache = shared_video_cache if shared_video_cache is not None else {}

        # Track unique tasks for logging
        self._task_counts = {}

        for ep_name in episode_names:
            ep_path = self.episodes_dir / ep_name
            self._load_episode(ep_path, val_ratio)

        logger.info(
            f"CrabEpisodeDataset [{split}]: {len(self.samples)} samples "
            f"from {len(episode_names)} episode dirs"
        )
        for task, count in self._task_counts.items():
            logger.info(f"  task: \"{task}\" — {count} samples")

        # Pre-decode all video files to disk-backed memmap
        self._preload_videos()

    def _read_task_from_parquet(self, ep_path: Path) -> str | None:
        """Read the task description from tasks.parquet if it exists."""
        tasks_path = ep_path / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return None
        try:
            tasks_df = pd.read_parquet(tasks_path)
            # Task string is in __index_level_0__ column (LeRobot v3 format)
            if "__index_level_0__" in tasks_df.columns:
                task = str(tasks_df["__index_level_0__"].iloc[0])
                return task
            # Also check index name
            if tasks_df.index.name and len(tasks_df.index) > 0:
                return str(tasks_df.index[0])
        except Exception as e:
            logger.warning(f"Failed to read tasks.parquet from {ep_path}: {e}")
        return None

    def _load_episode(self, ep_path: Path, val_ratio: float):
        """Load data from a single episode directory."""
        # Load info
        info_path = ep_path / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        # Read per-dataset task description from tasks.parquet
        task = self._read_task_from_parquet(ep_path)
        if task is None:
            task = self.task_description
            logger.warning(f"No tasks.parquet in {ep_path}, using fallback: \"{task}\"")
        else:
            logger.info(f"Dataset {ep_path.name}: task = \"{task}\"")

        # Load main data parquet
        data_files = sorted((ep_path / "data").rglob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in data_files]
        df = pd.concat(dfs, ignore_index=True)

        # Load stats for normalization
        stats_path = ep_path / "meta" / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        # Split by episode_index
        episode_indices = sorted(df["episode_index"].unique())
        n_episodes = len(episode_indices)
        n_val = max(1, int(n_episodes * val_ratio))

        if self.split == "train":
            selected_episodes = episode_indices[:-n_val]
        else:
            selected_episodes = episode_indices[-n_val:]

        for ep_idx in selected_episodes:
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
            n_frames = len(ep_df)

            # Track task counts
            self._task_counts[task] = self._task_counts.get(task, 0) + n_frames

            # Each sample starts at a frame and produces a chunk of actions
            for start_frame in range(n_frames):
                self.samples.append({
                    "ep_path": ep_path,
                    "ep_df": ep_df,
                    "frame_idx": start_frame,
                    "n_frames": n_frames,
                    "ep_idx": ep_idx,
                    "info": info,
                    "stats": stats,
                    "task": task,
                })

    def __len__(self):
        return len(self.samples)

    def _video_cache_path(self, video_path: str) -> Path:
        """Get the disk cache path for a video file."""
        # Use a hash of the video path + image size as the cache key
        h, w = self.image_size
        key = f"{video_path}_{h}x{w}"
        cache_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{cache_hash}.npy"

    def _preload_videos(self):
        """Pre-decode all unique video files to disk-backed numpy memmap files.

        First run: decodes videos and saves as .npy files (~60GB for 100K frames).
        Subsequent runs: opens existing .npy files as read-only memmap (instant).
        RAM usage: near zero — OS pages frames in/out as needed.
        """
        import av
        from torchvision.transforms.functional import resize

        # Collect unique video paths
        video_paths = set()
        for sample in self.samples:
            ep_path = sample["ep_path"]
            for cam_key in self.image_keys:
                video_file = ep_path / "videos" / cam_key / "chunk-000" / "file-000.mp4"
                video_paths.add(str(video_file))

        logger.info(f"Loading {len(video_paths)} video files (disk-backed cache at {self._cache_dir})...")

        n_cached = 0
        n_decoded = 0

        for video_path in sorted(video_paths):
            if video_path in self._video_cache:
                continue

            cache_file = self._video_cache_path(video_path)
            meta_file = cache_file.with_suffix(".json")

            if cache_file.exists() and meta_file.exists():
                # Load existing memmap from disk (instant, no RAM)
                with open(meta_file) as f:
                    meta = json.load(f)
                n_frames = meta["n_frames"]
                h, w = meta["height"], meta["width"]
                mmap = np.memmap(cache_file, dtype=np.uint8, mode="r",
                                 shape=(n_frames, 3, h, w))
                self._video_cache[video_path] = mmap
                n_cached += 1
            else:
                # Decode video and save to disk
                try:
                    container = av.open(video_path)
                    frames = []
                    for frame in container.decode(video=0):
                        img = frame.to_ndarray(format="rgb24")
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # uint8 [3, H, W]
                        img_tensor = resize(img_tensor, list(self.image_size), antialias=True)
                        frames.append(img_tensor.to(torch.uint8).numpy())
                    container.close()

                    n_frames = len(frames)
                    h, w = self.image_size
                    stacked = np.stack(frames)  # [N, 3, H, W] uint8

                    # Write to disk as memmap
                    mmap = np.memmap(cache_file, dtype=np.uint8, mode="w+",
                                     shape=(n_frames, 3, h, w))
                    mmap[:] = stacked
                    mmap.flush()
                    del mmap, stacked, frames

                    # Save metadata
                    with open(meta_file, "w") as f:
                        json.dump({"n_frames": n_frames, "height": h, "width": w,
                                   "video_path": video_path}, f)

                    # Re-open as read-only memmap
                    mmap = np.memmap(cache_file, dtype=np.uint8, mode="r",
                                     shape=(n_frames, 3, h, w))
                    self._video_cache[video_path] = mmap
                    n_decoded += 1
                    logger.info(f"  Decoded {video_path}: {n_frames} frames -> {cache_file.name}")

                except Exception as e:
                    logger.warning(f"Failed to decode video {video_path}: {e}")

        logger.info(f"Video loading complete. {n_cached} from cache, {n_decoded} newly decoded. "
                     f"{len(self._video_cache)} total videos.")

    def _get_video_frame(self, ep_path: Path, camera_key: str, global_index: int) -> torch.Tensor:
        """Get a single frame from the disk-backed video cache using global index."""
        video_path = str(ep_path / "videos" / camera_key / "chunk-000" / "file-000.mp4")

        if video_path not in self._video_cache:
            logger.warning(f"Video not in cache: {video_path}")
            h, w = self.image_size
            return torch.zeros(3, h, w, dtype=torch.float32)

        mmap = self._video_cache[video_path]

        if global_index < mmap.shape[0]:
            # Read single frame from memmap (OS pages it in), convert uint8->float32
            frame = torch.from_numpy(mmap[global_index].copy()).float() / 255.0
            return frame
        else:
            logger.warning(f"global_index {global_index} >= video frames {mmap.shape[0]}")
            h, w = self.image_size
            return torch.zeros(3, h, w, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ep_df = sample["ep_df"]
        frame_idx = sample["frame_idx"]
        n_frames = sample["n_frames"]
        ep_path = sample["ep_path"]
        ep_idx = sample["ep_idx"]

        row = ep_df.iloc[frame_idx]

        # --- State ---
        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        if self.state_indices is not None:
            state = state[self.state_indices]

        # --- Tactile (100 each) ---
        tactile_left = torch.tensor(row["observation.tactile_left"], dtype=torch.float32)
        tactile_right = torch.tensor(row["observation.tactile_right"], dtype=torch.float32)

        # --- Action chunk ---
        action_frames = []
        for i in range(self.chunk_size):
            future_idx = min(frame_idx + i, n_frames - 1)
            action = torch.tensor(ep_df.iloc[future_idx]["action"], dtype=torch.float32)
            if self.action_indices is not None:
                action = action[self.action_indices]
            action_frames.append(action)
        actions = torch.stack(action_frames)  # [chunk_size, action_dim]

        # --- Images (3 cameras) — use global index for correct video frame ---
        global_index = int(row["index"])
        images = {}
        for cam_key in self.image_keys:
            images[cam_key] = self._get_video_frame(ep_path, cam_key, global_index)

        # --- Metadata ---
        metadata = {
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "index": global_index,
        }

        return {
            "images": images,
            "state": state,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
            "action": actions,
            "task": sample["task"],
            "metadata": metadata,
        }


def collate_crab_batch(batch: list[dict]) -> dict:
    """Custom collate function for CrabEpisodeDataset."""
    images = {}
    for cam_key in batch[0]["images"]:
        images[cam_key] = torch.stack([b["images"][cam_key] for b in batch])

    return {
        "images": images,
        "state": torch.stack([b["state"] for b in batch]),
        "tactile_left": torch.stack([b["tactile_left"] for b in batch]),
        "tactile_right": torch.stack([b["tactile_right"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "task": [b["task"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
    }


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    base_dir = Path(cfg.get("_config_dir", "."))
    episodes_dir = base_dir / ds_cfg["episodes_dir"]

    # Optional joint subset indices
    action_indices = ds_cfg.get("action_indices", None)
    state_indices = ds_cfg.get("state_indices", None)

    # Video cache directory (configurable, defaults to /tmp/crab_video_cache)
    video_cache_dir = ds_cfg.get("video_cache_dir", None)

    common_kwargs = dict(
        episodes_dir=episodes_dir,
        episode_names=ds_cfg["episode_names"],
        chunk_size=cfg["model"]["chunk_size"],
        image_size=tuple(ds_cfg["image_size"]),
        fps=ds_cfg["fps"],
        task_description=ds_cfg.get("task_description", "Pick and place object"),
        val_ratio=ds_cfg["val_ratio"],
        image_keys=ds_cfg["image_keys"],
        action_indices=action_indices,
        state_indices=state_indices,
        video_cache_dir=video_cache_dir,
    )

    train_ds = CrabEpisodeDataset(split="train", **common_kwargs)
    # Share video cache between train and val (same memmap files, no extra memory)
    val_ds = CrabEpisodeDataset(split="val", shared_video_cache=train_ds._video_cache, **common_kwargs)

    num_workers = ds_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_crab_batch,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_crab_batch,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
