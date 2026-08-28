from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform

from tavrnn_reanalysis.config import CANONICAL_TASKS
from tavrnn_reanalysis.io import atomic_csv, atomic_json, atomic_npz


def classical_mds(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = distance.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ np.square(distance) @ centring
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    coordinates = vectors[:, :2] * np.sqrt(np.maximum(values[:2], 0.0))
    return coordinates, values


def align_to_reference(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref = reference - reference.mean(axis=0, keepdims=True)
    mov = moving - moving.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(mov, ref)
    return mov @ rotation


def aggregate_node_geometry(analysis_root: Path, output_dir: Path) -> dict:
    plan = pd.read_csv(analysis_root / "plan.csv")
    plan = plan[
        (plan["phase"] == "primary")
        & (plan["pipeline"] == "corrected_modulelist")
        & (plan["variant_id"] == "signed-d10")
    ].copy()
    distance_sum = {
        (session, task): np.zeros((332, 332), dtype=np.float64)
        for session in ("ses-01", "ses-02")
        for task in CANONICAL_TASKS
    }
    counts = Counter()
    scale_rows = []
    for record in plan.to_dict("records"):
        path = analysis_root / "runs" / str(record["run_id"]) / "embeddings.npz"
        with np.load(path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
            task_order = tuple(str(value) for value in archive["task_order"].tolist())
        for task_index, task in enumerate(task_order):
            points = embeddings[task_index]
            condensed = pdist(points, metric="euclidean")
            global_mean = float(condensed.mean())
            distance_sum[(str(record["session"]), task)] += squareform(condensed) / global_mean
            counts[(str(record["session"]), task)] += 1
            scale_rows.append(
                {
                    "run_id": str(record["run_id"]),
                    "subject": str(record["subject"]),
                    "session": str(record["session"]),
                    "seed": int(record["seed"]),
                    "condition": task,
                    "global_mean_distance": global_mean,
                }
            )
    if len(set(counts.values())) != 1:
        raise RuntimeError("visit-condition run counts differ")
    mean_distance = {key: value / counts[key] for key, value in distance_sum.items()}
    template_distance = np.mean(np.stack(list(mean_distance.values())), axis=0)
    reference, _ = classical_mds(template_distance)
    coordinates = {}
    projection_rows = []
    for key, distance in mean_distance.items():
        points, eigenvalues = classical_mds(distance)
        points = align_to_reference(reference, points)
        points /= float(pdist(points, metric="euclidean").mean())
        coordinates[key] = points
        positive_total = float(eigenvalues[eigenvalues > 0].sum())
        projection_rows.append(
            {
                "session": key[0],
                "condition": key[1],
                "n_runs": counts[key],
                "mds_variance_fraction_2d": float(
                    np.maximum(eigenvalues[:2], 0).sum() / positive_total
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        output_dir / "aggregate_normalized_distance_matrices.npz",
        **{
            f"{session.replace('-', '_')}_{task}": matrix.astype(np.float32)
            for (session, task), matrix in mean_distance.items()
        },
    )
    atomic_npz(
        output_dir / "aggregate_mds_coordinates.npz",
        **{
            f"{session.replace('-', '_')}_{task}": points.astype(np.float32)
            for (session, task), points in coordinates.items()
        },
    )
    atomic_csv(output_dir / "run_node_distance_scales.csv", pd.DataFrame(scale_rows))
    atomic_csv(
        output_dir / "aggregate_projection_metrics.csv",
        pd.DataFrame(projection_rows),
    )
    result = {
        "primary_runs": len(plan),
        "participants": int(plan["subject"].nunique()),
        "sessions": 2,
        "seeds": int(plan["seed"].nunique()),
        "runs_per_visit_condition": int(next(iter(counts.values()))),
    }
    atomic_json(output_dir / "node_geometry_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="node_geometry.py")
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate_node_geometry(args.analysis_root, args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
