from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


ANALYSES = (
    "author_stacked_01",
    "stacked_centered_pm05",
    "paired_peb_of_pebs",
)
FIELDS = [
    "branch",
    "task",
    "analysis",
    "n",
    "selected_edge_count_all",
    "selected_edge_count_offdiag",
    "reanalysis_total_abs_offdiag_pp99",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def included_count(path: Path) -> int:
    data = loadmat(path, simplify_cells=True)
    value = data.get("included_subjects", [])
    return int(np.asarray(value, dtype=object).size)


def summarize(root: Path, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effects_path in sorted(root.glob("*/*/*/drug_effects.csv")):
        relative = effects_path.relative_to(root)
        branch, task, analysis = relative.parts[:3]
        if analysis not in ANALYSES:
            continue
        effects = read_csv(effects_path)
        selected = [row for row in effects if float(row["bmr_pp"]) > threshold]
        selected_offdiag = [row for row in selected if int(row["is_self"]) == 0]
        rows.append(
            {
                "branch": branch,
                "task": task,
                "analysis": analysis,
                "n": included_count(effects_path.with_name("PEB_BMA_effects.mat")),
                "selected_edge_count_all": len(selected),
                "selected_edge_count_offdiag": len(selected_offdiag),
                "reanalysis_total_abs_offdiag_pp99": sum(
                    abs(float(row["bma_ep"])) for row in selected_offdiag
                ),
            }
        )
    rows.sort(key=lambda row: (row["branch"], row["task"], row["analysis"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--posterior-probability-threshold", type=float, default=0.99)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = summarize(root, args.posterior_probability_threshold)
    write_csv(root / "dcm_peb_compact_summary.csv", rows)
    print(f"Summarized {len(rows)} task-wise PEB analyses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
