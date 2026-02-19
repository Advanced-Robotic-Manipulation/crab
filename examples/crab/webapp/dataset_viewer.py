"""
Dataset Viewer tab for Crab robot web app.

Browse locally-cached LeRobot datasets, view metadata, and replay episodes
via embedded Rerun web viewer.
"""

import json
import logging
from pathlib import Path

from nicegui import ui

from process_manager import ProcessManager

logger = logging.getLogger(__name__)

DATASET_CACHE = Path("/mnt/drive/datasets")


def _scan_datasets() -> dict[str, dict]:
    """Scan the local LeRobot cache for datasets with info.json metadata."""
    datasets = {}
    if not DATASET_CACHE.exists():
        return datasets
    for d in sorted(DATASET_CACHE.iterdir()):
        info_file = d / "meta" / "info.json"
        if info_file.exists():
            try:
                with open(info_file) as f:
                    info = json.load(f)
                datasets[d.name] = info
            except Exception as e:
                logger.warning(f"Failed to read {info_file}: {e}")
    return datasets


def _format_features(info: dict) -> str:
    """Format feature info into a readable string."""
    features = info.get("features", {})
    parts = []
    for key, val in features.items():
        dtype = val.get("dtype", "?")
        shape = val.get("shape", "?")
        parts.append(f"{key}({dtype}, {shape})")
    return " | ".join(parts) if parts else "N/A"


def create_dataset_viewer():
    """Build the Dataset Viewer tab UI."""
    pm = ProcessManager()

    datasets: dict[str, dict] = {}
    current_dataset: dict = {}

    with ui.card().classes("w-full"):
        ui.label("DATASET BROWSER").classes("text-lg font-bold")

        with ui.row().classes("items-center gap-4 w-full"):
            dataset_select = ui.select(
                options=[],
                label="Dataset",
                on_change=lambda e: _on_dataset_change(e.value),
            ).classes("w-96")
            ui.button(icon="refresh", on_click=lambda: _refresh_datasets()).props("flat")

        # Metadata display
        with ui.card().classes("w-full bg-dark").bind_visibility_from(
            dataset_select, "value", backward=lambda v: v is not None
        ):
            meta_label = ui.label("").classes("text-sm font-mono")
            features_label = ui.label("").classes("text-xs text-grey-5 font-mono")

        ui.separator()

        # Episode controls
        with ui.row().classes("items-center gap-4 w-full"):
            episode_select = ui.select(
                options=[],
                value=None,
                label="Episode",
            ).classes("w-36")
            load_btn = ui.button(
                "Load in Rerun", icon="play_arrow",
                on_click=lambda: _load_rerun(),
            ).props("color=positive")
            stop_btn = ui.button(
                "Stop", icon="stop",
                on_click=lambda: _stop_rerun(),
            ).props("color=negative")
            rerun_status = ui.badge("Not running", color="grey").classes("text-sm")


    # ── Handlers ──────────────────────────────────────────────────────
    def _refresh_datasets():
        nonlocal datasets
        datasets = _scan_datasets()
        options = {name: name for name in datasets.keys()}
        dataset_select.options = options
        dataset_select.update()
        if datasets:
            ui.notify(f"Found {len(datasets)} dataset(s)", type="positive")
        else:
            ui.notify("No datasets found in cache", type="warning")

    def _on_dataset_change(name: str):
        nonlocal current_dataset
        if not name or name not in datasets:
            return
        current_dataset = datasets[name]
        info = current_dataset

        total_episodes = info.get("total_episodes", "?")
        fps = info.get("fps", "?")
        total_frames = info.get("total_frames", "?")
        meta_label.text = f"Episodes: {total_episodes} | FPS: {fps} | Frames: {total_frames}"
        features_label.text = _format_features(info)

        # Populate episode dropdown
        n_eps = info.get("total_episodes", 0)
        if isinstance(n_eps, int) and n_eps > 0:
            episode_select.options = {i: str(i) for i in range(n_eps)}
            episode_select.value = 0
        else:
            episode_select.options = {0: "0"}
            episode_select.value = 0
        episode_select.update()

    async def _load_rerun():
        name = dataset_select.value
        ep = episode_select.value
        if not name:
            ui.notify("Select a dataset first", type="warning")
            return

        rerun_status.text = "Loading..."
        rerun_status.props("color=warning")

        try:
            await pm.start_rerun_viewer(
                dataset_name=name,
                episode_index=int(ep) if ep is not None else 0,
            )
            rerun_status.text = "Opened in Rerun"
            rerun_status.props("color=positive")
            ui.notify("Dataset loaded in native Rerun viewer", type="positive")

        except Exception as e:
            rerun_status.text = "Error"
            rerun_status.props("color=negative")
            ui.notify(f"Failed to start Rerun: {e}", type="negative")

    async def _stop_rerun():
        await pm.stop_rerun_viewer()
        rerun_status.text = "Not running"
        rerun_status.props("color=grey")

    # Auto-scan on load
    _refresh_datasets()
