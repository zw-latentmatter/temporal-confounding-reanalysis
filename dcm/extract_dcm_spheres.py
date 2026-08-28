from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import svd


SPHERES = (
    ("aHip_L", (-26.0, -16.0, -20.0)),
    ("aHip_R", (28.0, -16.0, -20.0)),
    ("IPC_L", (-44.0, -60.0, 24.0)),
    ("IPC_R", (54.0, -62.0, 28.0)),
    ("mPFC", (2.0, 56.0, -4.0)),
    ("PCC", (2.0, -58.0, 30.0)),
)
RADIUS_MM = 6.0
SPHERE_CACHE: dict[tuple, list[np.ndarray]] = {}
GM_CACHE: dict[tuple, np.ndarray] = {}


def parse_entities(path: Path) -> tuple[str, str, str]:
    text = "/" + path.as_posix()
    subject_match = re.search(r"/(sub-[A-Za-z0-9]+)/", text)
    session_match = re.search(r"/(ses-[A-Za-z0-9]+)/", text)
    task_match = re.search(r"/task-([^/]+)/", text)
    if task_match is None:
        task_match = re.search(r"_task-([^_/.]+)", text)
    if subject_match is None or session_match is None or task_match is None:
        raise ValueError(f"Cannot parse subject, session and task from {path}")
    return subject_match.group(1), session_match.group(1), task_match.group(1)


def destination_path(root: Path, source: Path) -> Path:
    subject, session, task = parse_entities(source)
    return root / subject / session / f"task-{task}_dcm-spheres6mm.npz"


def matching_gm(dataset: Path, subject: str, pattern: str) -> Path:
    matches = sorted(dataset.glob(pattern.format(subject=subject)))
    if not matches:
        raise FileNotFoundError(f"Grey-matter probability image unavailable for {subject}")
    return matches[0]


def geometry_key(shape: tuple[int, ...], affine: np.ndarray) -> tuple:
    return shape, tuple(np.round(affine.ravel(), 7))


def sphere_masks(shape: tuple[int, ...], affine: np.ndarray) -> list[np.ndarray]:
    import nibabel as nib

    key = geometry_key(shape, affine)
    if key not in SPHERE_CACHE:
        indices = np.indices(shape, dtype=np.float64).reshape(3, -1).T
        world = nib.affines.apply_affine(affine, indices)
        SPHERE_CACHE[key] = [
            (np.square(world - np.asarray(centre)).sum(axis=1) <= RADIUS_MM**2 + 1e-8).reshape(shape)
            for _, centre in SPHERES
        ]
    return SPHERE_CACHE[key]


def gm_mask(
    dataset: Path,
    subject: str,
    shape: tuple[int, ...],
    affine: np.ndarray,
    pattern: str,
    threshold: float,
) -> np.ndarray:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    key = (subject, pattern, threshold, *geometry_key(shape, affine))
    if key not in GM_CACHE:
        image = nib.load(str(matching_gm(dataset, subject, pattern)))
        binary = nib.Nifti1Image(
            (np.asarray(image.dataobj) >= threshold).astype(np.uint8), image.affine
        )
        GM_CACHE[key] = np.asarray(
            resample_from_to(binary, (shape, affine), order=0).dataobj, dtype=np.uint8
        ) > 0
    return GM_CACHE[key]


def first_component(values: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.asarray(values, dtype=np.float64)
    u, singular_values, vh = svd(
        matrix,
        full_matrices=False,
        overwrite_a=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    sign = 1.0 if float(vh[0].sum()) >= 0 else -1.0
    component = u[:, 0] * singular_values[0] * sign / np.sqrt(min(matrix.shape))
    explained = float(singular_values[0] ** 2 / np.square(matrix).sum(dtype=np.float64))
    return component, explained


def extract_one(job: tuple[str, str, str, str, float, bool]) -> str:
    import nibabel as nib

    dataset_text, output_text, source_text, gm_pattern, gm_threshold, overwrite = job
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    dataset = Path(dataset_text)
    output_root = Path(output_text)
    source = Path(source_text)
    destination = destination_path(output_root, source)
    if destination.exists() and not overwrite:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subject, _, _ = parse_entities(source)
    image = nib.load(str(dataset / source))
    if len(image.shape) != 4:
        raise ValueError(f"Expected four-dimensional BOLD data in {source}")
    shape = image.shape[:3]
    bold = image.get_fdata(dtype=np.float32)
    grey_matter = gm_mask(
        dataset, subject, shape, image.affine, gm_pattern, gm_threshold
    )
    masks = sphere_masks(shape, image.affine)
    n_time = image.shape[3]
    gm_intersect = np.empty((n_time, len(SPHERES)), dtype=np.float32)
    sphere_only = np.empty_like(gm_intersect)
    sphere_voxels = np.empty(len(SPHERES), dtype=np.int32)
    gm_voxels = np.empty(len(SPHERES), dtype=np.int32)
    gm_fallback = np.zeros(len(SPHERES), dtype=bool)
    gm_explained = np.empty(len(SPHERES), dtype=np.float32)
    sphere_explained = np.empty(len(SPHERES), dtype=np.float32)
    for index, mask in enumerate(masks):
        sphere_voxels[index] = int(mask.sum())
        sphere_values = np.asarray(bold[mask, :].T, dtype=np.float64)
        sphere_only[:, index], sphere_explained[index] = first_component(sphere_values)
        selected = mask & grey_matter
        if not selected.any():
            selected = mask
            gm_fallback[index] = True
        gm_voxels[index] = int(selected.sum())
        selected_values = np.asarray(bold[selected, :].T, dtype=np.float64)
        gm_intersect[:, index], gm_explained[index] = first_component(selected_values)
    temporary = destination.with_name(f".{destination.stem}.partial.npz")
    np.savez_compressed(
        temporary,
        author_literal=gm_intersect[5:],
        author_sphere_only=sphere_only[5:],
        roi_names=np.asarray([name for name, _ in SPHERES]),
        mni_centres_mm=np.asarray([centre for _, centre in SPHERES], dtype=np.float32),
        radius_mm=np.float32(RADIUS_MM),
        voxels_sphere=sphere_voxels,
        voxels_after_gm=gm_voxels,
        gm_fallback=gm_fallback,
        author_explained=gm_explained,
        author_sphere_only_explained=sphere_explained,
        ignored_initial_volumes=np.int32(5),
        gm_threshold=np.float32(gm_threshold),
    )
    temporary.replace(destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--bold-pattern",
        default="*_space-MNI152NLin2009cAsym_desc-optcomGLM_bold.nii.gz",
    )
    parser.add_argument(
        "--gm-pattern",
        default=(
            "derivatives/fmriprep-22.0.2/{subject}/**/anat/"
            "{subject}*_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"
        ),
    )
    parser.add_argument("--gm-threshold", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(path.relative_to(dataset) for path in dataset.rglob(args.bold_pattern))
    jobs = [
        (
            str(dataset),
            str(output_root),
            str(source),
            args.gm_pattern,
            args.gm_threshold,
            args.overwrite,
        )
        for source in sources
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(extract_one, jobs))
    print(f"Extracted {len(outputs)} six-region time-series files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
