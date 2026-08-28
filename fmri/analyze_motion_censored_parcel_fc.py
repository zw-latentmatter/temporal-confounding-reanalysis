from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

TASKS = ("rest", "meditation", "music", "movie")
NETWORKS = ("sub", "vis", "sommot", "dorsattn", "limbic", "salventattn", "default", "cont")
NETWORK_TOKENS = (None, "_Vis_", "_SomMot_", "_DorsAttn_", "_Limbic_", "_SalVentAttn_", "_Default_", "_Cont_")


def entities(path: Path) -> tuple[str, str, str]:
    text = path.as_posix()
    subject = re.search(r"/(sub-PC\d+)/", text).group(1)
    session = re.search(r"/(ses-\d+)/", text).group(1)
    task = re.search(r"/task-([^/_]+)_", text).group(1)
    return subject, session, task


def atlas_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = np.asarray(lines[0::2], dtype="U96")
    if labels.size != 332:
        raise RuntimeError(f"expected 332 atlas labels, found {labels.size}")
    modules = np.zeros(332, dtype=np.int8)
    modules[:32] = 1
    for index, label in enumerate(labels[32:], start=32):
        modules[index] = next(
            module
            for module, token in enumerate(NETWORK_TOKENS[1:], start=2)
            if token in label
        )
    return labels, modules


def confounds_path(dataset: Path, subject: str, session: str, task: str) -> Path:
    root = dataset / "derivatives" / "fmriprep-22.0.2" / subject / session / "func"
    matches = sorted(root.glob(f"{subject}_{session}_task-{task}_run-*_desc-confounds_timeseries.tsv"))
    if not matches:
        raise FileNotFoundError(f"confounds are unavailable for {subject} {session} {task}")
    return matches[0]


def negative_asymmetric_modularity(weights: np.ndarray, modules: np.ndarray) -> float:
    positive = np.maximum(weights, 0.0).astype(np.float64, copy=False)
    negative = np.maximum(-weights, 0.0).astype(np.float64, copy=False)
    positive_strength = float(positive.sum())
    negative_strength = float(negative.sum())
    same = modules[:, None] == modules[None, :]
    if positive_strength:
        degree = positive.sum(axis=1)
        positive_score = float(
            ((positive - np.outer(degree, degree) / positive_strength) * same).sum()
        )
    else:
        positive_score = 0.0
    if negative_strength:
        degree = negative.sum(axis=1)
        negative_score = float(
            ((negative - np.outer(degree, degree) / negative_strength) * same).sum()
        )
    else:
        negative_score = 0.0
    positive_term = positive_score / positive_strength if positive_strength else 0.0
    total_strength = positive_strength + negative_strength
    negative_term = negative_score / total_strength if negative_strength and total_strength else 0.0
    return positive_term - negative_term


def analyze_run(arguments: tuple[str, str, np.ndarray, float, int, str]) -> dict | None:
    path_text, dataset_text, modules, fd_threshold, minimum_frames, series_key = arguments
    path = Path(path_text)
    dataset = Path(dataset_text)
    subject, session, task = entities(path)
    with np.load(path, allow_pickle=False) as archive:
        time_series = np.asarray(archive[series_key], dtype=np.float64)
    confounds = pd.read_csv(
        confounds_path(dataset, subject, session, task),
        sep="\t",
        usecols=["framewise_displacement"],
    )["framewise_displacement"].to_numpy(dtype=np.float64)
    offset = confounds.size - time_series.shape[0]
    if offset < 0:
        raise RuntimeError(f"confounds are shorter than ROI data for {path.name}")
    fd = confounds[offset:]
    keep = np.isfinite(fd) & (fd <= fd_threshold)
    retained = int(keep.sum())
    if retained < minimum_frames:
        return None
    selected = time_series[keep]
    if not np.isfinite(selected).all():
        raise RuntimeError(f"non-finite ROI data for {path.name}")
    fc = np.corrcoef(selected, rowvar=False).astype(np.float64, copy=False)
    np.fill_diagonal(fc, 0.0)
    within = np.empty(8, dtype=np.float64)
    for module in range(1, 9):
        indices = np.flatnonzero(modules == module)
        local = fc[np.ix_(indices, indices)]
        within[module - 1] = local[np.triu_indices(indices.size, k=1)].mean()
    return {
        "subject": subject,
        "session": session,
        "task": task,
        "retained_frames": retained,
        "fc": fc.astype(np.float32),
        "within_network_mean_r": within,
        "modularity_negative_asym": negative_asymmetric_modularity(fc, modules),
    }


def paired_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    test = stats.ttest_1samp(values, 0.0)
    return {
        "n": n,
        "mean_delta": mean,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "paired_t": float(test.statistic),
        "paired_t_p_two_sided": float(test.pvalue),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fd-threshold", type=float, default=0.2)
    parser.add_argument("--minimum-frames", type=int, default=264)
    parser.add_argument("--series-key", default="author_literal")
    args = parser.parse_args()
    labels, modules = atlas_labels(args.labels)
    order = np.argsort(modules, kind="stable")
    ordered_modules = modules[order]
    paths = sorted(args.volume_root.glob("sub-*/ses-*/task-*_volume-roi332.npz"))
    work = [
        (
            str(path),
            str(args.dataset),
            modules,
            args.fd_threshold,
            args.minimum_frames,
            args.series_key,
        )
        for path in paths
    ]
    if args.workers == 1:
        results = [analyze_run(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(analyze_run, work, chunksize=1))
    runs = [result for result in results if result is not None]
    lookup = {(row["subject"], row["session"], row["task"]): row for row in runs}
    paired_subjects = {
        task: sorted(
            subject
            for subject in {key[0] for key in lookup if key[2] == task}
            if (subject, "ses-01", task) in lookup and (subject, "ses-02", task) in lookup
        )
        for task in TASKS
    }
    max_participants = max(map(len, paired_subjects.values()))
    mean_baseline = np.full((4, 332, 332), np.nan, dtype=np.float32)
    mean_psilocybin = np.full_like(mean_baseline, np.nan)
    mean_delta = np.full_like(mean_baseline, np.nan)
    subject_ids = np.full((4, max_participants), "", dtype="U12")
    within_delta = np.full((4, max_participants, 8), np.nan, dtype=np.float32)
    modularity_delta = np.full((4, max_participants), np.nan, dtype=np.float32)
    participant_rows: list[dict] = []
    statistic_rows: list[dict] = []
    for task_index, task in enumerate(TASKS):
        subjects = paired_subjects[task]
        baseline_matrices = []
        psilocybin_matrices = []
        for subject_index, subject in enumerate(subjects):
            baseline = lookup[subject, "ses-01", task]
            psilocybin = lookup[subject, "ses-02", task]
            baseline_fc = baseline["fc"][np.ix_(order, order)]
            psilocybin_fc = psilocybin["fc"][np.ix_(order, order)]
            baseline_matrices.append(baseline_fc)
            psilocybin_matrices.append(psilocybin_fc)
            subject_ids[task_index, subject_index] = subject
            local_within = psilocybin["within_network_mean_r"] - baseline["within_network_mean_r"]
            local_modularity = psilocybin["modularity_negative_asym"] - baseline["modularity_negative_asym"]
            within_delta[task_index, subject_index] = local_within
            modularity_delta[task_index, subject_index] = local_modularity
            for network_index, network in enumerate(NETWORKS):
                participant_rows.append(
                    {
                        "subject": subject,
                        "task": task,
                        "measure": "within_network_mean_r",
                        "network": network,
                        "baseline": float(baseline["within_network_mean_r"][network_index]),
                        "psilocybin": float(psilocybin["within_network_mean_r"][network_index]),
                        "delta": float(local_within[network_index]),
                    }
                )
            participant_rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "measure": "modularity_negative_asym",
                    "network": "all",
                    "baseline": float(baseline["modularity_negative_asym"]),
                    "psilocybin": float(psilocybin["modularity_negative_asym"]),
                    "delta": float(local_modularity),
                }
            )
        baseline_stack = np.asarray(baseline_matrices, dtype=np.float32)
        psilocybin_stack = np.asarray(psilocybin_matrices, dtype=np.float32)
        mean_baseline[task_index] = baseline_stack.mean(axis=0)
        mean_psilocybin[task_index] = psilocybin_stack.mean(axis=0)
        mean_delta[task_index] = (psilocybin_stack - baseline_stack).mean(axis=0)
        for network_index, network in enumerate(NETWORKS):
            values = within_delta[task_index, : len(subjects), network_index]
            statistic_rows.append(
                {
                    "task": task,
                    "measure": "within_network_mean_r",
                    "network": network,
                    **paired_summary(values),
                }
            )
        statistic_rows.append(
            {
                "task": task,
                "measure": "modularity_negative_asym",
                "network": "all",
                **paired_summary(modularity_delta[task_index, : len(subjects)]),
            }
        )
    statistics = pd.DataFrame(statistic_rows)
    statistics["paired_t_q_bh"] = np.nan
    for measure, indices in statistics.groupby("measure").groups.items():
        local = np.asarray(list(indices), dtype=int)
        statistics.loc[local, "paired_t_q_bh"] = multipletests(
            statistics.loc[local, "paired_t_p_two_sided"],
            method="fdr_bh",
        )[1]
    run_rows = []
    for row in runs:
        run_rows.append(
            {
                "subject": row["subject"],
                "session": row["session"],
                "task": row["task"],
                "retained_frames": row["retained_frames"],
                "modularity_negative_asym": row["modularity_negative_asym"],
                **{
                    f"within_{network}_mean_r": float(row["within_network_mean_r"][index])
                    for index, network in enumerate(NETWORKS)
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(args.output_dir / "strict_motion_run_metrics.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(
        args.output_dir / "strict_motion_participant_contrasts.csv",
        index=False,
    )
    statistics.to_csv(args.output_dir / "strict_motion_paired_statistics.csv", index=False)
    np.savez_compressed(
        args.output_dir / "strict_motion_fc_matrices.npz",
        tasks=np.asarray(TASKS),
        networks=np.asarray(NETWORKS),
        labels=labels[order],
        modules=ordered_modules,
        mean_baseline_r=mean_baseline,
        mean_psilocybin_r=mean_psilocybin,
        mean_delta_r=mean_delta,
        subject_ids=subject_ids,
        within_network_delta_r=within_delta,
        modularity_negative_asym_delta=modularity_delta,
        fd_threshold_mm=np.asarray(args.fd_threshold),
        minimum_retained_frames=np.asarray(args.minimum_frames),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
