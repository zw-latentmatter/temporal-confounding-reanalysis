from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

TASKS = ("rest", "meditation", "music", "movie")
EYES_CLOSED = ("rest", "meditation", "music")
SESSIONS = ("ses-01", "ses-02")
NETWORK_TOKENS = {
    "visual": "Vis",
    "somatomotor": "SomMot",
    "dorsal_attention": "DorsAttn",
    "limbic": "Limbic",
    "salience_ventral_attention": "SalVentAttn",
    "default": "Default",
    "control": "Cont",
}


def entities(path: Path) -> tuple[str, str, str]:
    text = path.as_posix()
    subject = re.search(r"/(sub-PC\d+)/", text).group(1)
    session = re.search(r"/(ses-\d+)/", text).group(1)
    task = re.search(r"/task-([^/_]+)_", text).group(1)
    return subject, session, task


def cortical_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(path)
    parcel_labels = np.asarray(image.dataobj).reshape(-1).astype(np.int32)
    axes = [image.header.get_axis(index) for index in range(len(image.shape))]
    label_axis = next(axis for axis in axes if axis.__class__.__name__ == "LabelAxis")
    table = label_axis.label[0]
    parcel_networks: dict[int, int] = {}
    for raw_key, value in table.items():
        key = int(raw_key)
        if key == 0:
            continue
        name = str(value[0])
        matches = [
            code
            for code, token in enumerate(NETWORK_TOKENS.values(), start=1)
            if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", name)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot assign parcel {key} to a network")
        parcel_networks[key] = matches[0]
    cortical_mask = parcel_labels > 0
    network_codes = np.zeros(parcel_labels.size, dtype=np.int8)
    for parcel, network in parcel_networks.items():
        network_codes[parcel_labels == parcel] = network
    return cortical_mask, network_codes


def complete_subjects(entity_paths: dict[tuple[str, str, str], Path]) -> list[str]:
    required = {(session, task) for session in SESSIONS for task in TASKS}
    subjects = sorted({subject for subject, _, _ in entity_paths})
    return [
        subject
        for subject in subjects
        if required.issubset(
            {(session, task) for local, session, task in entity_paths if local == subject}
        )
    ]


def load_arrays(entity_paths: dict[tuple[str, str, str], Path], key: str) -> dict:
    result = {}
    for entity, path in entity_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            result[entity] = np.asarray(archive[key], dtype=np.float32)
    return result


def contrast_specs() -> list[dict]:
    specs = [
        {
            "family": "task_session_effect",
            "contrast": f"{task}_psilocybin_minus_baseline",
            "task": task,
        }
        for task in TASKS
    ]
    specs.extend(
        {
            "family": "closed_movie_interaction",
            "contrast": f"{task}_minus_movie_session_interaction",
            "task": task,
        }
        for task in EYES_CLOSED
    )
    specs.append(
        {
            "family": "closed_movie_interaction",
            "contrast": "eyes_closed_mean_minus_movie_session_interaction",
            "task": None,
        }
    )
    return specs


def vertex_delta(arrays: dict, subjects: list[str], spec: dict) -> np.ndarray:
    def stack(session: str, task: str) -> np.ndarray:
        return np.stack([arrays[subject, session, task] for subject in subjects])

    if spec["family"] == "task_session_effect":
        return stack("ses-02", spec["task"]) - stack("ses-01", spec["task"])
    if spec["task"] is not None:
        return (
            stack("ses-02", spec["task"])
            - stack("ses-02", "movie")
            - stack("ses-01", spec["task"])
            + stack("ses-01", "movie")
        )
    psilocybin = np.mean([stack("ses-02", task) for task in EYES_CLOSED], axis=0) - stack(
        "ses-02", "movie"
    )
    baseline = np.mean([stack("ses-01", task) for task in EYES_CLOSED], axis=0) - stack(
        "ses-01", "movie"
    )
    return psilocybin - baseline


def analytic_statistics(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(delta, dtype=np.float64)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0, ddof=1)
    t_values = np.divide(
        mean,
        standard_deviation / math.sqrt(values.shape[0]),
        out=np.zeros_like(mean),
        where=standard_deviation > 0,
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), values.shape[0] - 1)
    q_values = multipletests(p_values, method="fdr_bh")[1]
    return mean, t_values, p_values, q_values


def max_t_sign_flip(arguments: tuple[str, np.ndarray, int, int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    name, delta, permutations, seed, chunk_size = arguments
    delta = np.ascontiguousarray(delta, dtype=np.float32)
    n_subjects = delta.shape[0]
    values = delta.astype(np.float64)
    sum_squares = np.square(values).sum(axis=0)
    observed_mean = values.mean(axis=0)
    observed_variance = (sum_squares - n_subjects * np.square(observed_mean)) / (n_subjects - 1)
    observed_t = np.divide(
        observed_mean,
        np.sqrt(np.maximum(observed_variance, 0) / n_subjects),
        out=np.zeros_like(observed_mean),
        where=observed_variance > 0,
    )
    random = np.random.default_rng(seed)
    maxima = np.empty(permutations, dtype=np.float32)
    completed = 0
    while completed < permutations:
        count = min(chunk_size, permutations - completed)
        signs = random.integers(0, 2, size=(count, n_subjects), dtype=np.int8)
        signs = (2 * signs - 1).astype(np.float32)
        means = signs @ delta / n_subjects
        variances = (sum_squares[None, :] - n_subjects * np.square(means)) / (n_subjects - 1)
        t_values = np.divide(
            means,
            np.sqrt(np.maximum(variances, 0) / n_subjects),
            out=np.zeros_like(means, dtype=np.float64),
            where=variances > 0,
        )
        maxima[completed : completed + count] = np.max(np.abs(t_values), axis=1)
        completed += count
    sorted_maxima = np.sort(maxima)
    exceedances = permutations - np.searchsorted(sorted_maxima, np.abs(observed_t), side="left")
    corrected = (exceedances + 1) / (permutations + 1)
    return name, corrected.astype(np.float32), maxima


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--dlabel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--permutation-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10910)
    parser.add_argument("--array-key", default="author_fisher_gfc")
    args = parser.parse_args()
    paths = sorted(args.surface_root.glob("sub-*/ses-*/task-*_surface-gfc32k.npz"))
    entity_paths = {entities(path): path for path in paths}
    subjects = complete_subjects(entity_paths)
    arrays = load_arrays(entity_paths, args.array_key)
    cortical_mask, network_codes = cortical_labels(args.dlabel)
    specs = contrast_specs()
    deltas = {
        spec["contrast"]: np.ascontiguousarray(
            vertex_delta(arrays, subjects, spec)[:, cortical_mask],
            dtype=np.float32,
        )
        for spec in specs
    }
    analytic = {name: analytic_statistics(delta) for name, delta in deltas.items()}
    jobs = [
        (
            spec["contrast"],
            deltas[spec["contrast"]],
            args.permutations,
            args.seed + index,
            args.permutation_chunk,
        )
        for index, spec in enumerate(specs)
    ]
    max_t_results = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(max_t_sign_flip, job) for job in jobs]
        for future in as_completed(futures):
            name, corrected, maxima = future.result()
            max_t_results[name] = corrected, maxima
    n_vertices = cortical_mask.size
    shape = len(specs), n_vertices
    mean_maps = np.full(shape, np.nan, dtype=np.float32)
    t_maps = np.full(shape, np.nan, dtype=np.float32)
    p_maps = np.full(shape, np.nan, dtype=np.float32)
    q_maps = np.full(shape, np.nan, dtype=np.float32)
    max_t_maps = np.full(shape, np.nan, dtype=np.float32)
    null_maxima = np.empty((len(specs), args.permutations), dtype=np.float32)
    summary_rows = []
    participant_rows = []
    for index, spec in enumerate(specs):
        name = spec["contrast"]
        mean, t_values, p_values, q_values = analytic[name]
        max_t_values, maxima = max_t_results[name]
        mean_maps[index, cortical_mask] = mean
        t_maps[index, cortical_mask] = t_values
        p_maps[index, cortical_mask] = p_values
        q_maps[index, cortical_mask] = q_values
        max_t_maps[index, cortical_mask] = max_t_values
        null_maxima[index] = maxima
        summary_rows.append(
            {
                "family": spec["family"],
                "contrast": name,
                "participant_n": len(subjects),
                "cortical_vertex_n": int(cortical_mask.sum()),
                "mean_delta_across_vertices": float(mean.mean()),
                "minimum_two_sided_p": float(p_values.min()),
                "minimum_bh_q": float(q_values.min()),
                "bh_significant_vertex_n": int((q_values < 0.05).sum()),
                "minimum_max_t_fwer_p": float(max_t_values.min()),
                "max_t_significant_vertex_n": int((max_t_values < 0.05).sum()),
                "sign_flip_permutations": args.permutations,
                "max_t_null_95th_percentile": float(np.quantile(maxima, 0.95)),
            }
        )
        participant_means = deltas[name].mean(axis=1)
        participant_rows.extend(
            {
                "subject": subject,
                "family": spec["family"],
                "contrast": name,
                "mean_delta_across_vertices": float(value),
            }
            for subject, value in zip(subjects, participant_means)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "surface_gfc_vertex_statistics.npz",
        contrasts=np.asarray([spec["contrast"] for spec in specs]),
        subjects=np.asarray(subjects),
        network_codes=network_codes,
        cortical_mask=cortical_mask,
        mean_delta=mean_maps,
        paired_t=t_maps,
        two_sided_p=p_maps,
        bh_fdr_q=q_maps,
        max_t_fwer_p=max_t_maps,
        max_t_null=null_maxima,
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "surface_gfc_vertex_statistics.csv",
        index=False,
    )
    pd.DataFrame(participant_rows).to_csv(
        args.output_dir / "surface_gfc_participant_contrasts.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
