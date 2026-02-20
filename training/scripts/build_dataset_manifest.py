#!/usr/bin/env python3
"""
Build and clean dataset manifest for RWFM training scope.

Scope:
- pick_and_place*
- *to_tray*

Outputs:
- dataset_manifest.csv
- dataset_issues.md

Optional cleanup (`--fix`) currently performs:
1) Remove meta/episodes rows whose episode_index is absent in data parquet.
2) Fill NaN in meta `episode_reward` for valid episodes using:
   step_sum + R_succ*episode_success - R_drop*episode_drop - R_damage*episode_damage
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TARGET_FAMILY_PREFIXES = ("pick_and_place",)
TARGET_FAMILY_CONTAINS = ("to_tray",)


@dataclass
class DatasetScan:
    family: str
    dataset: str
    rel_path: str
    abs_path: Path
    task: str | None = None
    steps: int = 0
    episodes_data: int = 0
    episodes_meta: int | None = None
    has_reward: bool = True
    reward_nan: int = 0
    has_episode_reward: bool = False
    episode_reward_nan: int | None = None
    has_episode_success: bool = False
    episode_success_nan: int | None = None
    has_episode_damage: bool = False
    episode_damage_nan: int | None = None
    has_episode_drop: bool = False
    episode_drop_nan: int | None = None
    status: str = "OK"
    include_for_training: bool = True
    exclusion_reason: str = ""
    notes: str = ""
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


def is_target_family(name: str) -> bool:
    if name.startswith(TARGET_FAMILY_PREFIXES):
        return True
    return any(x in name for x in TARGET_FAMILY_CONTAINS)


def read_task_string(dataset_dir: Path) -> str | None:
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    table = pq.read_table(tasks_path)
    if "__index_level_0__" in table.column_names and table.num_rows > 0:
        return str(table["__index_level_0__"][0].as_py())
    # Fallback for unusual exports
    try:
        df = table.to_pandas()
        if len(df.index) > 0:
            return str(df.index[0])
    except Exception:
        pass
    return None


def load_reward_params(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "meta" / "reward_params.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_historical_cleanup_details(dataset_dir: Path) -> list[str]:
    details: list[str] = []
    backup = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet.bak_step2"
    current = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not (backup.exists() and current.exists()):
        return details

    try:
        b_names = pq.read_schema(backup).names
        c_names = pq.read_schema(current).names
        b_cols = ["episode_index"] + (["episode_reward"] if "episode_reward" in b_names else [])
        c_cols = ["episode_index"] + (["episode_reward"] if "episode_reward" in c_names else [])
        b = pq.read_table(backup, columns=b_cols)
        c = pq.read_table(current, columns=c_cols)

        b_ep = set(int(x) for x in b["episode_index"].to_pylist())
        c_ep = set(int(x) for x in c["episode_index"].to_pylist())
        removed = sorted(b_ep - c_ep)
        if removed:
            details.append(f"Historical fix: removed stale meta episodes {removed}")

        if "episode_reward" in b.column_names and "episode_reward" in c.column_names:
            b_nan = int(np.isnan(np.array(b["episode_reward"].to_pylist(), dtype=np.float32)).sum())
            c_nan = int(np.isnan(np.array(c["episode_reward"].to_pylist(), dtype=np.float32)).sum())
            if b_nan > c_nan:
                details.append(f"Historical fix: episode_reward NaN reduced {b_nan} -> {c_nan}")
    except Exception:
        details.append("Historical cleanup backup detected (details unavailable)")

    return details


def scan_data_files(data_files: list[Path]) -> tuple[int, set[int], bool, int, dict[int, float]]:
    steps = 0
    episode_ids: set[int] = set()
    has_reward = True
    reward_nan = 0
    step_sum_by_episode: dict[int, float] = defaultdict(float)

    for data_file in data_files:
        names = pq.read_schema(data_file).names
        cols = ["episode_index"]
        if "reward" in names:
            cols.append("reward")
        table = pq.read_table(data_file, columns=cols)

        steps += table.num_rows
        ep = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
        episode_ids.update(int(x) for x in ep)

        if "reward" in table.column_names:
            reward = np.array(table["reward"].to_pylist(), dtype=np.float32)
            reward_nan += int(np.isnan(reward).sum())
            # Per-episode step reward sum (needed for NaN fix in meta episode_reward).
            for e in np.unique(ep):
                idx = ep == e
                step_sum_by_episode[int(e)] += float(np.nansum(reward[idx]))
        else:
            has_reward = False

    return steps, episode_ids, has_reward, reward_nan, step_sum_by_episode


def cleanup_meta_episodes_if_needed(
    dataset_dir: Path,
    data_episode_ids: set[int],
    step_sum_by_episode: dict[int, float],
    scan: DatasetScan,
) -> None:
    meta_file = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta_file.exists():
        return

    table = pq.read_table(meta_file)
    names = table.column_names
    changed = False

    if "episode_index" not in names:
        return

    ep_idx = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
    keep_indices = [i for i, e in enumerate(ep_idx) if int(e) in data_episode_ids]
    if len(keep_indices) != len(ep_idx):
        removed = len(ep_idx) - len(keep_indices)
        table = table.take(pa.array(keep_indices, type=pa.int64()))
        scan.fixes.append(f"Removed {removed} stale meta episode rows not present in data")
        changed = True

    names = table.column_names
    if "episode_reward" in names:
        episode_reward = np.array(table["episode_reward"].to_pylist(), dtype=np.float32)
        nan_idx = np.where(np.isnan(episode_reward))[0]
        if len(nan_idx) > 0:
            reward_params = load_reward_params(dataset_dir)
            params = reward_params.get("params", {})
            r_succ = float(params.get("R_succ", 0.0))
            r_drop = float(params.get("R_drop", 0.0))
            r_damage = float(params.get("R_damage", 0.0))

            ep = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
            ep_success = (
                np.array(table["episode_success"].to_pylist(), dtype=np.float32)
                if "episode_success" in names
                else np.zeros(table.num_rows, dtype=np.float32)
            )
            ep_drop = (
                np.array(table["episode_drop"].to_pylist(), dtype=np.float32)
                if "episode_drop" in names
                else np.zeros(table.num_rows, dtype=np.float32)
            )
            ep_damage = (
                np.array(table["episode_damage"].to_pylist(), dtype=np.float32)
                if "episode_damage" in names
                else np.zeros(table.num_rows, dtype=np.float32)
            )

            filled = 0
            for i in nan_idx:
                e = int(ep[i])
                if e in step_sum_by_episode:
                    terminal = r_succ * ep_success[i] - r_drop * ep_drop[i] - r_damage * ep_damage[i]
                    episode_reward[i] = float(step_sum_by_episode[e] + terminal)
                    filled += 1

            if filled > 0:
                col_idx = names.index("episode_reward")
                table = table.set_column(
                    col_idx, "episode_reward", pa.array(episode_reward, type=pa.float32())
                )
                scan.fixes.append(f"Filled {filled} NaN values in meta episode_reward")
                changed = True

    if not changed:
        return

    backup = meta_file.with_suffix(meta_file.suffix + ".bak_step2")
    if not backup.exists():
        shutil.copy2(meta_file, backup)

    pq.write_table(table, meta_file)


def scan_dataset(root: Path, family_dir: Path, dataset_dir: Path, apply_fix: bool) -> DatasetScan:
    rel_path = dataset_dir.relative_to(root).as_posix()
    scan = DatasetScan(
        family=family_dir.name,
        dataset=dataset_dir.name,
        rel_path=rel_path,
        abs_path=dataset_dir,
    )

    historical_backup = (
        dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet.bak_step2"
    )
    if historical_backup.exists():
        details = infer_historical_cleanup_details(dataset_dir)
        if details:
            scan.fixes.extend(details)
        else:
            scan.fixes.append("Historical cleanup backup detected (file-000.parquet.bak_step2)")

    data_files = sorted(dataset_dir.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        scan.status = "EXCLUDED"
        scan.include_for_training = False
        scan.exclusion_reason = "No data parquet files"
        scan.issues.append("No data parquet files")
        return scan

    (
        scan.steps,
        data_episode_ids,
        scan.has_reward,
        scan.reward_nan,
        step_sum_by_episode,
    ) = scan_data_files(data_files)
    scan.episodes_data = len(data_episode_ids)

    scan.task = read_task_string(dataset_dir)
    if scan.task is None:
        scan.issues.append("Missing or unreadable meta/tasks.parquet")

    if apply_fix:
        cleanup_meta_episodes_if_needed(dataset_dir, data_episode_ids, step_sum_by_episode, scan)

    meta_file = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta_file.exists():
        scan.issues.append("Missing meta/episodes parquet")
    else:
        names = pq.read_schema(meta_file).names
        cols = ["episode_index"]
        for c in ["episode_reward", "episode_success", "episode_damage", "episode_drop"]:
            if c in names:
                cols.append(c)

        table = pq.read_table(meta_file, columns=cols)
        scan.episodes_meta = table.num_rows
        meta_episode_ids = set(int(x) for x in table["episode_index"].to_pylist())

        scan.has_episode_reward = "episode_reward" in table.column_names
        scan.has_episode_success = "episode_success" in table.column_names
        scan.has_episode_damage = "episode_damage" in table.column_names
        scan.has_episode_drop = "episode_drop" in table.column_names

        if scan.has_episode_reward:
            vals = np.array(table["episode_reward"].to_pylist(), dtype=np.float32)
            scan.episode_reward_nan = int(np.isnan(vals).sum())
        if scan.has_episode_success:
            vals = np.array(table["episode_success"].to_pylist(), dtype=np.float32)
            scan.episode_success_nan = int(np.isnan(vals).sum())
        if scan.has_episode_damage:
            vals = np.array(table["episode_damage"].to_pylist(), dtype=np.float32)
            scan.episode_damage_nan = int(np.isnan(vals).sum())
        if scan.has_episode_drop:
            vals = np.array(table["episode_drop"].to_pylist(), dtype=np.float32)
            scan.episode_drop_nan = int(np.isnan(vals).sum())

        if scan.episodes_meta != scan.episodes_data:
            scan.issues.append(
                f"meta episode rows ({scan.episodes_meta}) != data unique episodes ({scan.episodes_data})"
            )
        if meta_episode_ids != data_episode_ids:
            scan.issues.append("episode_index mismatch between data and meta")

    if not scan.has_reward:
        scan.issues.append("Missing reward column in at least one data parquet file")
    if scan.reward_nan > 0:
        scan.issues.append(f"Found {scan.reward_nan} NaN values in step reward")
    if scan.has_episode_reward is False:
        scan.issues.append("Missing episode_reward in meta/episodes")
    if scan.has_episode_success is False:
        scan.issues.append("Missing episode_success in meta/episodes")
    if scan.has_episode_damage is False:
        scan.issues.append("Missing episode_damage in meta/episodes")
    if scan.episode_reward_nan is not None and scan.episode_reward_nan > 0:
        scan.issues.append(f"Found {scan.episode_reward_nan} NaN values in episode_reward")

    if scan.issues:
        scan.status = "EXCLUDED"
        scan.include_for_training = False
        scan.exclusion_reason = "; ".join(scan.issues)
    else:
        scan.status = "OK"
        scan.include_for_training = True

    if scan.fixes:
        scan.notes = "; ".join(scan.fixes)

    return scan


def write_manifest_csv(path: Path, scans: list[DatasetScan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "family",
        "steps",
        "episodes_data",
        "episodes_meta",
        "task",
        "has_reward",
        "reward_nan",
        "has_episode_reward",
        "episode_reward_nan",
        "has_episode_success",
        "episode_success_nan",
        "has_episode_damage",
        "episode_damage_nan",
        "has_episode_drop",
        "episode_drop_nan",
        "status",
        "include_for_training",
        "exclusion_reason",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scans:
            writer.writerow(
                {
                    "dataset": s.rel_path,
                    "family": s.family,
                    "steps": s.steps,
                    "episodes_data": s.episodes_data,
                    "episodes_meta": s.episodes_meta,
                    "task": s.task or "",
                    "has_reward": s.has_reward,
                    "reward_nan": s.reward_nan,
                    "has_episode_reward": s.has_episode_reward,
                    "episode_reward_nan": s.episode_reward_nan,
                    "has_episode_success": s.has_episode_success,
                    "episode_success_nan": s.episode_success_nan,
                    "has_episode_damage": s.has_episode_damage,
                    "episode_damage_nan": s.episode_damage_nan,
                    "has_episode_drop": s.has_episode_drop,
                    "episode_drop_nan": s.episode_drop_nan,
                    "status": s.status,
                    "include_for_training": s.include_for_training,
                    "exclusion_reason": s.exclusion_reason,
                    "notes": s.notes,
                }
            )


def write_issues_md(path: Path, scans: list[DatasetScan], family_tasks: dict[str, set[str]], fix_mode: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(scans)
    ok = sum(1 for s in scans if s.status == "OK")
    excluded = total - ok
    total_steps = sum(s.steps for s in scans if s.status == "OK")
    total_eps = sum(s.episodes_data for s in scans if s.status == "OK")

    lines: list[str] = []
    lines.append("# Dataset Issues (Step 2)")
    lines.append("")
    lines.append(f"- Total datasets scanned: **{total}**")
    lines.append(f"- Status `OK`: **{ok}**")
    lines.append(f"- Status `EXCLUDED`: **{excluded}**")
    lines.append(f"- Included steps (OK only): **{total_steps}**")
    lines.append(f"- Included episodes (OK only): **{total_eps}**")
    lines.append(f"- Cleanup mode: **{'enabled' if fix_mode else 'disabled'}**")
    lines.append("")

    lines.append("## Task String Consistency")
    lines.append("")
    for family in sorted(family_tasks):
        tasks = sorted(t for t in family_tasks[family] if t is not None)
        lines.append(f"- `{family}`: {len(tasks)} unique task string(s)")
        for t in tasks:
            lines.append(f"  - `{t}`")
    lines.append("")

    fixed_scans = [s for s in scans if s.fixes]
    lines.append("## Applied Fixes")
    lines.append("")
    if fixed_scans:
        for s in fixed_scans:
            lines.append(f"- `{s.rel_path}`")
            for fix in s.fixes:
                lines.append(f"  - {fix}")
    else:
        lines.append("- No automatic fixes were applied.")
    lines.append("")

    excluded_scans = [s for s in scans if s.status == "EXCLUDED"]
    lines.append("## Excluded Datasets")
    lines.append("")
    if excluded_scans:
        for s in excluded_scans:
            lines.append(f"- `{s.rel_path}`: {s.exclusion_reason}")
    else:
        lines.append("- None. All datasets are marked `OK`.")
    lines.append("")

    lines.append("## Acceptance Check")
    lines.append("")
    if excluded_scans:
        lines.append("- NOT READY: there are excluded datasets.")
    else:
        lines.append("- READY: every dataset is either `OK` or explicitly excluded (none excluded).")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and clean dataset manifest for RWFM Step 2.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--manifest-csv", type=str, default="crab/training/docs/dataset_manifest.csv")
    parser.add_argument("--issues-md", type=str, default="crab/training/docs/dataset_issues.md")
    parser.add_argument("--fix", action="store_true", help="Apply safe automatic cleanup for meta anomalies.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    families = sorted([d for d in root.iterdir() if d.is_dir() and is_target_family(d.name)])

    scans: list[DatasetScan] = []
    family_tasks: dict[str, set[str]] = defaultdict(set)

    for family_dir in families:
        for dataset_dir in sorted(d for d in family_dir.iterdir() if d.is_dir()):
            if not (dataset_dir / "data").exists():
                continue
            scan = scan_dataset(root, family_dir, dataset_dir, apply_fix=args.fix)
            scans.append(scan)
            family_tasks[scan.family].add(scan.task)

    # Family-level task string consistency check.
    inconsistent_families = {fam for fam, tasks in family_tasks.items() if len([t for t in tasks if t]) > 1}
    if inconsistent_families:
        for scan in scans:
            if scan.family in inconsistent_families:
                scan.issues.append("Inconsistent task strings within family")
                scan.status = "EXCLUDED"
                scan.include_for_training = False
                scan.exclusion_reason = "; ".join(scan.issues)

    manifest_csv = root / args.manifest_csv
    issues_md = root / args.issues_md
    write_manifest_csv(manifest_csv, scans)
    write_issues_md(issues_md, scans, family_tasks, fix_mode=args.fix)

    ok = sum(1 for s in scans if s.status == "OK")
    excluded = len(scans) - ok
    print(f"Scanned {len(scans)} datasets: OK={ok}, EXCLUDED={excluded}")
    print(f"Manifest: {manifest_csv}")
    print(f"Issues:   {issues_md}")


if __name__ == "__main__":
    main()
