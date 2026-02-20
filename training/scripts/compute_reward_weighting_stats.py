#!/usr/bin/env python3
"""
Compute robust reward-normalization statistics and RWFM weight diagnostics.

Outputs:
- reward_weighting_stats.json

Scope:
- pick_and_place*
- *to_tray*
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


TARGET_FAMILY_PREFIXES = ("pick_and_place",)
TARGET_FAMILY_CONTAINS = ("to_tray",)


def is_target_family(name: str) -> bool:
    if name.startswith(TARGET_FAMILY_PREFIXES):
        return True
    return any(x in name for x in TARGET_FAMILY_CONTAINS)


def robust_median_mad(x: np.ndarray, eps: float = 1e-6) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if scale < eps:
        std = float(np.std(x))
        scale = std if std > eps else 1.0
    return median, mad, float(scale)


def discounted_chunk_return(reward: np.ndarray, horizon: int, gamma: float) -> np.ndarray:
    """Compute finite-horizon discounted return for each timestep in O(T)."""
    reward = np.asarray(reward, dtype=np.float64)
    T = reward.shape[0]
    out = np.zeros(T + 1, dtype=np.float64)
    gamma_h = float(gamma**horizon)

    for t in range(T - 1, -1, -1):
        tail = reward[t + horizon] if (t + horizon) < T else 0.0
        out[t] = reward[t] + gamma * out[t + 1] - gamma_h * tail
    return out[:T]


def percentile_dict(x: np.ndarray, ps: list[float]) -> dict[str, float]:
    vals = np.percentile(x, ps)
    return {f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(ps, vals, strict=True)}


def read_task(dataset_dir: Path) -> str:
    tfile = dataset_dir / "meta" / "tasks.parquet"
    table = pq.read_table(tfile)
    if "__index_level_0__" in table.column_names and table.num_rows > 0:
        return str(table["__index_level_0__"][0].as_py())
    return dataset_dir.parent.name


def load_manifest_includes(manifest_csv: Path) -> set[str] | None:
    if not manifest_csv.exists():
        return None
    include = set()
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "")
            include_flag = str(row.get("include_for_training", "")).strip().lower() == "true"
            if dataset and include_flag:
                include.add(dataset)
    return include


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RWFM normalization/diagnostics stats.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument(
        "--manifest-csv",
        type=str,
        default="crab/training/docs/dataset_manifest.csv",
        help="If exists, only datasets with include_for_training=True are used.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="crab/training/docs/reward_weighting_stats.json",
    )
    parser.add_argument("--group-key", type=str, default="task", choices=["task", "family"])
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mix-beta", type=float, default=0.7, help="A = beta*z_ep + (1-beta)*z_chunk")
    parser.add_argument("--z-clip", type=float, default=6.0)
    parser.add_argument("--a-clip", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--w-min", type=float, default=0.25)
    parser.add_argument("--w-max", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sim-batches", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    include_set = load_manifest_includes(root / args.manifest_csv)

    records: list[dict[str, Any]] = []
    for family_dir in sorted([d for d in root.iterdir() if d.is_dir() and is_target_family(d.name)]):
        for dataset_dir in sorted(d for d in family_dir.iterdir() if d.is_dir()):
            rel = dataset_dir.relative_to(root).as_posix()
            if include_set is not None and rel not in include_set:
                continue
            data_file = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
            meta_file = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            if not (data_file.exists() and meta_file.exists()):
                continue

            task = read_task(dataset_dir)
            family = family_dir.name
            group = task if args.group_key == "task" else family

            dt = pq.read_table(data_file, columns=["episode_index", "reward"])
            episode_idx = np.array(dt["episode_index"].to_pylist(), dtype=np.int64)
            reward = np.array(dt["reward"].to_pylist(), dtype=np.float64)

            mt = pq.read_table(meta_file, columns=["episode_index", "episode_reward"])
            ep_meta_idx = np.array(mt["episode_index"].to_pylist(), dtype=np.int64)
            ep_meta_reward = np.array(mt["episode_reward"].to_pylist(), dtype=np.float64)
            episode_reward_map = {int(e): float(r) for e, r in zip(ep_meta_idx, ep_meta_reward, strict=True)}

            for e in np.unique(episode_idx):
                idx = np.where(episode_idx == e)[0]
                r = reward[idx]
                ch = discounted_chunk_return(r, horizon=args.horizon, gamma=args.gamma)
                ep = np.full(ch.shape[0], episode_reward_map[int(e)], dtype=np.float64)
                records.append(
                    {
                        "family": family,
                        "task": task,
                        "group": group,
                        "chunk_return": ch,
                        "episode_reward": ep,
                    }
                )

    if not records:
        raise RuntimeError("No records found in selected scope.")

    by_group: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunk": [], "episode": []})
    for rec in records:
        by_group[rec["group"]]["chunk"].append(rec["chunk_return"])
        by_group[rec["group"]]["episode"].append(rec["episode_reward"])

    group_stats: dict[str, dict[str, float]] = {}
    for group, data in by_group.items():
        chunk = np.concatenate(data["chunk"])
        episode = np.concatenate(data["episode"])
        med_c, mad_c, sc_c = robust_median_mad(chunk)
        med_e, mad_e, sc_e = robust_median_mad(episode)
        group_stats[group] = {
            "chunk_median": med_c,
            "chunk_mad": mad_c,
            "chunk_scale": sc_c,
            "episode_median": med_e,
            "episode_mad": mad_e,
            "episode_scale": sc_e,
            "n_samples": int(chunk.shape[0]),
        }

    A_all = []
    A_by_family: dict[str, list[np.ndarray]] = defaultdict(list)

    for rec in records:
        st = group_stats[rec["group"]]
        z_chunk = (rec["chunk_return"] - st["chunk_median"]) / st["chunk_scale"]
        z_episode = (rec["episode_reward"] - st["episode_median"]) / st["episode_scale"]
        z_chunk = np.clip(z_chunk, -args.z_clip, args.z_clip)
        z_episode = np.clip(z_episode, -args.z_clip, args.z_clip)
        A = args.mix_beta * z_episode + (1.0 - args.mix_beta) * z_chunk
        A = np.clip(A, -args.a_clip, args.a_clip)
        A_all.append(A)
        A_by_family[rec["family"]].append(A)

    A_all = np.concatenate(A_all)
    A_by_family = {k: np.concatenate(v) for k, v in A_by_family.items()}

    w_raw = np.exp(args.alpha * A_all)
    w_clip = np.clip(w_raw, args.w_min, args.w_max)
    w_norm = w_clip / np.mean(w_clip)

    global_diag = {
        "A_percentiles": percentile_dict(A_all, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]),
        "w_raw_percentiles": percentile_dict(w_raw, [1, 5, 25, 50, 75, 95, 99]),
        "w_norm_percentiles": percentile_dict(w_norm, [1, 5, 25, 50, 75, 95, 99]),
        "clip_low_frac": float(np.mean(w_raw < args.w_min)),
        "clip_high_frac": float(np.mean(w_raw > args.w_max)),
        "w_norm_mean": float(np.mean(w_norm)),
        "w_norm_std": float(np.std(w_norm)),
    }

    family_diag = {}
    for family, arr in sorted(A_by_family.items()):
        wr = np.exp(args.alpha * arr)
        family_diag[family] = {
            "n_samples": int(arr.shape[0]),
            "A_percentiles": percentile_dict(arr, [1, 50, 99]),
            "A_mean": float(np.mean(arr)),
            "clip_low_frac": float(np.mean(wr < args.w_min)),
            "clip_high_frac": float(np.mean(wr > args.w_max)),
        }

    # Batch-level stability simulation using Step-1 target family mix.
    mix = {
        "pick_and_place_can": 0.20,
        "pick_and_place_bottle": 0.20,
        "pick_and_place_waffle": 0.25,
        "egg_carton_to_tray": 0.25,
        "egg_carton_to_tray_cracked": 0.10,
    }
    fam_keys = [k for k in mix if k in A_by_family]
    probs = np.array([mix[k] for k in fam_keys], dtype=np.float64)
    probs = probs / probs.sum()

    rng = np.random.default_rng(args.seed)
    ess = []
    clip_low_batch = []
    clip_high_batch = []
    wmax_batch = []
    wmin_batch = []
    for _ in range(args.sim_batches):
        fam_batch = rng.choice(fam_keys, size=args.batch_size, p=probs)
        A_b = np.array([A_by_family[f][rng.integers(0, len(A_by_family[f]))] for f in fam_batch], dtype=np.float64)
        wr = np.exp(args.alpha * A_b)
        clip_low_batch.append(np.mean(wr < args.w_min))
        clip_high_batch.append(np.mean(wr > args.w_max))
        wc = np.clip(wr, args.w_min, args.w_max)
        wn = wc / np.mean(wc)
        wmax_batch.append(np.max(wn))
        wmin_batch.append(np.min(wn))
        ess.append((np.sum(wn) ** 2) / (args.batch_size * np.sum(wn**2)))

    batch_diag = {
        "batch_size": args.batch_size,
        "sim_batches": args.sim_batches,
        "clip_low_frac_mean": float(np.mean(clip_low_batch)),
        "clip_high_frac_mean": float(np.mean(clip_high_batch)),
        "ess_mean": float(np.mean(ess)),
        "ess_p10": float(np.percentile(ess, 10)),
        "wn_max_p99": float(np.percentile(wmax_batch, 99)),
        "wn_min_p1": float(np.percentile(wmin_batch, 1)),
    }

    out = {
        "config": {
            "group_key": args.group_key,
            "horizon": args.horizon,
            "gamma": args.gamma,
            "mix_beta": args.mix_beta,
            "z_clip": args.z_clip,
            "a_clip": args.a_clip,
            "alpha": args.alpha,
            "w_min": args.w_min,
            "w_max": args.w_max,
            "batch_size": args.batch_size,
            "sim_batches": args.sim_batches,
        },
        "group_stats": group_stats,
        "global_diagnostics": global_diag,
        "family_diagnostics": family_diag,
        "batch_diagnostics": batch_diag,
    }

    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
