from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def parse_entities(path: str) -> tuple[str, str, str]:
    subject = re.search(r"/(sub-PC\d+)/", "/" + path).group(1)
    session = re.search(r"/(ses-\d+)/", "/" + path).group(1)
    task = re.search(r"/task-([^/]+)/", "/" + path).group(1)
    return subject, session, task


def output_path(root: Path, source: str) -> Path:
    subject, session, task = parse_entities(source)
    return root / subject / session / f"task-{task}_surface-gfc32k.npz"


def fisher_gfc(data: np.ndarray, block_size: int) -> tuple[np.ndarray, dict]:
    data = np.asarray(data, dtype=np.float32)
    data -= data.mean(axis=0, keepdims=True)
    standard_deviation = data.std(axis=0, ddof=1)
    valid = np.isfinite(standard_deviation) & (standard_deviation > 0)
    standardized = np.ascontiguousarray(data[:, valid] / standard_deviation[valid])
    n_time, n_valid = standardized.shape
    result = np.zeros(data.shape[1], dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    valid_result = np.empty(n_valid, dtype=np.float32)
    for start in range(0, n_valid, block_size):
        stop = min(start + block_size, n_valid)
        correlations = standardized[:, start:stop].T @ standardized / (n_time - 1)
        rows = np.arange(stop - start)
        correlations[rows, start + rows] = 0.0
        np.clip(correlations, -1 + 1e-7, 1 - 1e-7, out=correlations)
        np.arctanh(correlations, out=correlations)
        valid_result[start:stop] = correlations.mean(axis=1, dtype=np.float64)
    result[valid_indices] = valid_result
    metadata = {
        "n_timepoints": int(n_time),
        "n_vertices": int(data.shape[1]),
        "n_valid_vertices": int(n_valid),
    }
    return result, metadata


def extract_one(arguments: tuple[str, str, str, int]) -> dict:
    import h5py

    dataset_text, output_text, source_text, block_size = arguments
    dataset = Path(dataset_text)
    output_root = Path(output_text)
    destination = output_path(output_root, source_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset / source_text, "r") as matlab:
        left = np.asarray(matlab["data_dtseries/lh"][5:, :], dtype=np.float32)
        right = np.asarray(matlab["data_dtseries/rh"][5:, :], dtype=np.float32)
    left_gfc, left_metadata = fisher_gfc(left, block_size)
    right_gfc, right_metadata = fisher_gfc(right, block_size)
    np.savez_compressed(
        destination,
        author_fisher_gfc=np.concatenate([left_gfc, right_gfc]),
    )
    subject, session, task = parse_entities(source_text)
    metadata = {
        "source": source_text,
        "subject": subject,
        "session": session,
        "task": task,
        "ignored_initial_volumes": 5,
        "left": left_metadata,
        "right": right_metadata,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def discover_surface_files(dataset: Path) -> list[str]:
    root = dataset / "derivatives" / "tedana-0.0.12-GLM"
    return sorted(
        path.relative_to(dataset).as_posix()
        for path in root.rglob("*desc-optcomGLM_fsLR_32k.mat")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = discover_surface_files(args.dataset)
    work = [(str(args.dataset), str(args.output_root), source, args.block_size) for source in files]
    if args.workers == 1:
        for item in work:
            extract_one(item)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(extract_one, work, chunksize=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
