from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


BRANCH_KEYS = {
    "author_gm_intersect": "author_literal",
    "author_sphere_only": "author_sphere_only",
}
EXPECTED_ROIS = 6


def parse_entities(path: Path) -> tuple[str, str, str]:
    text = "/" + path.as_posix()
    subject_match = re.search(r"/(sub-[A-Za-z0-9]+)/", text)
    session_match = re.search(r"/(ses-[A-Za-z0-9]+)/", text)
    task_match = re.search(r"/task-(.+?)_dcm-spheres6mm\.npz$", text)
    if subject_match is None or session_match is None or task_match is None:
        raise ValueError(f"Cannot parse subject, session and task from {path}")
    return subject_match.group(1), session_match.group(1), task_match.group(1)


def destination_path(output_root: Path, source: Path, branch: str) -> Path:
    subject, session, task = parse_entities(source)
    return (
        output_root
        / branch
        / subject
        / session
        / f"task-{task}_timeseries-dcm6mm_tedanaGLM.mat"
    )


def matlab_string(value: object) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    item = array.reshape(-1)[0]
    while isinstance(item, np.ndarray) and item.size == 1:
        item = item.reshape(-1)[0]
    return str(item)


def valid_mat_input(path: Path, branch: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        data = loadmat(
            path,
            variable_names=["time_series", "ROI_names", "branch", "source_key"],
            squeeze_me=False,
            struct_as_record=False,
        )
        series = np.asarray(data["time_series"], dtype=np.float64)
        names = np.asarray(data["ROI_names"], dtype=object)
        stored_branch = matlab_string(data["branch"])
        source_key = matlab_string(data["source_key"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(
        series.ndim == 2
        and series.shape[0] >= 8
        and series.shape[1] == EXPECTED_ROIS
        and names.size == EXPECTED_ROIS
        and np.isfinite(series).all()
        and stored_branch == branch
        and source_key == BRANCH_KEYS[branch]
    )


def atomic_savemat(destination: Path, variables: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial.mat", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        savemat(
            temporary,
            variables,
            appendmat=False,
            format="5",
            do_compression=False,
            oned_as="row",
        )
        if not valid_mat_input(temporary, str(variables["branch"])):
            raise ValueError(f"Invalid generated MAT input for {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_one(
    source: Path,
    output_root: Path,
    branch: str,
    tr_seconds: float,
    overwrite: bool,
) -> Path:
    destination = destination_path(output_root, source, branch)
    if destination.exists() and not overwrite:
        if valid_mat_input(destination, branch):
            return destination
        raise ValueError(f"Existing MAT input is invalid: {destination}")
    key = BRANCH_KEYS[branch]
    with np.load(source, allow_pickle=False) as archive:
        series = np.asarray(archive[key], dtype=np.float64)
        roi_names = [str(value) for value in np.asarray(archive["roi_names"]).reshape(-1)]
        if series.ndim != 2 or series.shape[1] != EXPECTED_ROIS or series.shape[0] < 8:
            raise ValueError(f"Unexpected {key} array shape {series.shape} in {source}")
        if len(roi_names) != EXPECTED_ROIS or not np.isfinite(series).all():
            raise ValueError(f"Invalid six-region time series in {source}")
        variables = {
            "time_series": series,
            "ROI_names": np.asarray(roi_names, dtype=object).reshape(1, -1),
            "branch": branch,
            "source_key": key,
            "tr_seconds": np.float64(tr_seconds),
            "n_timepoints": np.int32(series.shape[0]),
            "n_regions": np.int32(series.shape[1]),
        }
        for metadata_key in (
            "mni_centres_mm",
            "radius_mm",
            "voxels_sphere",
            "voxels_after_gm",
            "gm_fallback",
            "author_explained",
            "author_sphere_only_explained",
        ):
            if metadata_key in archive.files:
                variables[metadata_key] = np.asarray(archive[metadata_key])
    atomic_savemat(destination, variables)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=tuple(BRANCH_KEYS),
        default=tuple(BRANCH_KEYS),
    )
    parser.add_argument("--tr-seconds", type=float, default=0.91)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(input_root.glob("sub-*/ses-*/task-*_dcm-spheres6mm.npz"))
    outputs = [
        convert_one(source, output_root, branch, args.tr_seconds, args.overwrite)
        for source in sources
        for branch in dict.fromkeys(args.branches)
    ]
    print(f"Generated {len(outputs)} SPM input files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
