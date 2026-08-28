from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import svd

ATLAS_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def parse_entities(path: str) -> tuple[str, str, str]:
    subject = re.search(r"/(sub-PC\d+)/", "/" + path).group(1)
    session = re.search(r"/(ses-\d+)/", "/" + path).group(1)
    task = re.search(r"/task-([^/]+)/", "/" + path).group(1)
    return subject, session, task


def read_label_names(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = lines[0::2]
    if len(names) != 332:
        raise RuntimeError(f"expected 332 labels, found {len(names)}")
    return names


def atlas_arrays(atlas_dir: Path, shape: tuple[int, ...], affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    key = shape, tuple(np.round(affine.ravel(), 7))
    if key not in ATLAS_CACHE:
        tian_image = nib.load(str(atlas_dir / "Tian_Subcortex_S2_3T_2009cAsym_1mm.nii.gz"))
        schaefer_image = nib.load(
            str(atlas_dir / "Schaefer2018_300Parcels7Networks_MNI152NLin2009cAsym_res-01.nii.gz")
        )
        target = shape, affine
        tian = np.asarray(resample_from_to(tian_image, target, order=0).dataobj, dtype=np.int16)
        schaefer = np.asarray(resample_from_to(schaefer_image, target, order=0).dataobj, dtype=np.int16)
        ATLAS_CACHE[key] = tian, schaefer
    return ATLAS_CACHE[key]


def first_component(values: np.ndarray) -> tuple[np.ndarray, float]:
    left, singular_values, right = svd(
        values,
        full_matrices=False,
        overwrite_a=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    sign = 1.0 if float(right[0].sum()) >= 0 else -1.0
    component = left[:, 0] * singular_values[0] * sign
    explained = float(singular_values[0] ** 2 / np.square(values).sum(dtype=np.float64))
    return component, explained


def matching_gm(dataset: Path, subject: str) -> Path:
    pattern = (
        f"derivatives/fmriprep-22.0.2/{subject}/**/anat/"
        f"{subject}*_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz"
    )
    matches = sorted(dataset.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"GM probability map is unavailable for {subject}")
    return matches[0]


def output_path(output_root: Path, source: str) -> Path:
    subject, session, task = parse_entities(source)
    return output_root / subject / session / f"task-{task}_volume-roi332.npz"


def extract_one(arguments: tuple[str, str, str, str]) -> dict:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    dataset_text, atlas_text, output_text, source_text = arguments
    dataset = Path(dataset_text)
    atlas_dir = Path(atlas_text)
    output_root = Path(output_text)
    source = dataset / source_text
    subject, session, task = parse_entities(source_text)
    destination = output_path(output_root, source_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bold_image = nib.load(str(source))
    if len(bold_image.shape) != 4:
        raise RuntimeError(f"expected four-dimensional BOLD data for {source.name}")
    spatial_shape = bold_image.shape[:3]
    n_time = bold_image.shape[3]
    bold = bold_image.get_fdata(dtype=np.float32)
    gm_image = nib.load(str(matching_gm(dataset, subject)))
    gm_binary = np.asarray(gm_image.dataobj) >= 0.35
    gm_thresholded = nib.Nifti1Image(gm_binary.astype(np.uint8), gm_image.affine)
    gm = np.asarray(resample_from_to(gm_thresholded, (spatial_shape, bold_image.affine), order=0).dataobj) > 0
    tian, schaefer = atlas_arrays(atlas_dir, spatial_shape, bold_image.affine)
    author = np.empty((n_time, 332), dtype=np.float32)
    centered = np.empty((n_time, 332), dtype=np.float32)
    mean_signal = np.empty((n_time, 332), dtype=np.float32)
    voxels_atlas = np.empty(332, dtype=np.int32)
    voxels_used = np.empty(332, dtype=np.int32)
    gm_fallback = np.zeros(332, dtype=bool)
    author_explained = np.empty(332, dtype=np.float32)
    centered_explained = np.empty(332, dtype=np.float32)
    for index in range(332):
        roi = tian == index + 1 if index < 32 else schaefer == index - 31
        voxels_atlas[index] = int(roi.sum())
        intersection = roi & gm
        if intersection.any():
            roi = intersection
        else:
            gm_fallback[index] = True
        voxels_used[index] = int(roi.sum())
        if not roi.any():
            raise RuntimeError(f"empty ROI {index + 1}")
        values = np.asarray(bold[roi, :].T, dtype=np.float64)
        author_component, author_variance = first_component(values)
        author[:, index] = author_component / np.sqrt(min(values.shape))
        author_explained[index] = author_variance
        demeaned = values - values.mean(axis=0, keepdims=True)
        centered_component, centered_variance = first_component(demeaned)
        centered[:, index] = centered_component / np.sqrt(demeaned.shape[1])
        centered_explained[index] = centered_variance
        mean_signal[:, index] = demeaned.mean(axis=1)
    author = author[5:]
    centered = centered[5:]
    mean_signal = mean_signal[5:]
    author_fc = np.corrcoef(author, rowvar=False).astype(np.float32)
    centered_fc = np.corrcoef(centered, rowvar=False).astype(np.float32)
    mean_fc = np.corrcoef(mean_signal, rowvar=False).astype(np.float32)
    np.fill_diagonal(author_fc, 0.0)
    np.fill_diagonal(centered_fc, 0.0)
    np.fill_diagonal(mean_fc, 0.0)
    similarity = np.diag(np.corrcoef(author, centered, rowvar=False)[:332, 332:]).astype(np.float32)
    np.savez_compressed(
        destination,
        author_literal=author,
        corrected_centered_pc1=centered,
        corrected_mean=mean_signal,
        author_fc=author_fc,
        corrected_centered_fc=centered_fc,
        corrected_mean_fc=mean_fc,
        voxels_atlas=voxels_atlas,
        voxels_used=voxels_used,
        gm_fallback=gm_fallback,
        author_explained=author_explained,
        centered_explained=centered_explained,
        author_centered_similarity=similarity,
    )
    metadata = {
        "source": source_text,
        "subject": subject,
        "session": session,
        "task": task,
        "shape": list(bold_image.shape),
        "ignored_initial_volumes": 5,
        "gm_threshold": 0.35,
        "gm_fallback_count": int(gm_fallback.sum()),
        "roi_below_6_voxels": int((voxels_used < 6).sum()),
        "author_explained_median": float(np.median(author_explained)),
        "centered_explained_median": float(np.median(centered_explained)),
        "author_centered_similarity_median": float(np.nanmedian(similarity)),
        "author_centered_similarity_abs_median": float(np.nanmedian(np.abs(similarity))),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def discover_bold_files(dataset: Path) -> list[str]:
    root = dataset / "derivatives" / "tedana-0.0.12-GLM"
    return sorted(
        path.relative_to(dataset).as_posix()
        for path in root.rglob("*space-MNI152NLin2009cAsym_desc-optcomGLM_bold.nii.gz")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = read_label_names(
        args.atlas_dir / "Schaefer2018_300Parcels_7Networks_order_Tian_Subcortex_S2_label.txt"
    )
    (args.output_root / "roi_labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    files = discover_bold_files(args.dataset)
    work = [(str(args.dataset), str(args.atlas_dir), str(args.output_root), source) for source in files]
    if args.workers == 1:
        for item in work:
            extract_one(item)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(extract_one, work, chunksize=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
