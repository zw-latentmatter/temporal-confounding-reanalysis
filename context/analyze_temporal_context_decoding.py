from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

TASKS = ("rest", "meditation", "music", "movie")
SESSIONS = ("ses-01", "ses-02")
FEATURES = ("absolute_concatenated_time_only", "within_run_phase_only")
TASK_TO_LABEL = {task: index for index, task in enumerate(TASKS)}


def parse_entities(path: Path) -> tuple[str, str, str]:
    text = path.as_posix()
    subject = re.search(r"/(sub-PC\d+)/", text).group(1)
    session = re.search(r"/(ses-\d+)/", text).group(1)
    task = re.search(r"/task-([^/_]+)_", text).group(1)
    return subject, session, task


def numeric_subject(subject: str) -> int:
    return int(re.search(r"\d+", subject).group())


def deterministic_seed(base: int, subject: str, session: str, offset: int = 0) -> int:
    session_number = int(session.split("-")[1]) if session.startswith("ses-") else 0
    return base + numeric_subject(subject) * 10000 + session_number * 1000 + offset


def discover_sessions(root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    sessions: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root.glob("sub-*/ses-*/task-*_volume-roi332.npz")):
        subject, session, task = parse_entities(path)
        if task in TASK_TO_LABEL:
            sessions.setdefault((subject, session), {})[task] = path
    return {
        key: paths
        for key, paths in sessions.items()
        if all(task in paths for task in TASKS)
    }


def load_temporal_session(task_paths: dict[str, str | Path], length_key: str) -> dict:
    labels = []
    absolute_parts = []
    phase_parts = []
    run_slices = []
    offset = 0
    for task in TASKS:
        path = Path(task_paths[task])
        with np.load(path, allow_pickle=False) as archive:
            n_frames = int(np.asarray(archive[length_key]).shape[0])
        labels.append(np.full(n_frames, TASK_TO_LABEL[task], dtype=np.int8))
        absolute_parts.append(np.arange(offset, offset + n_frames, dtype=np.float64)[:, None])
        phase_parts.append(np.linspace(0.0, 1.0, n_frames, dtype=np.float64)[:, None])
        run_slices.append((offset, offset + n_frames))
        offset += n_frames
    absolute_time = np.concatenate(absolute_parts)
    absolute_time /= max(offset - 1, 1)
    return {
        "y": np.concatenate(labels),
        "absolute_concatenated_time_only": absolute_time,
        "within_run_phase_only": np.concatenate(phase_parts),
        "run_slices": run_slices,
    }


def classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight="balanced",
                    cache_size=512,
                ),
            ),
        ]
    )


def score_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    model = classifier()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    matrix = confusion_matrix(test_y, prediction, labels=np.arange(4), normalize="true")
    return {
        "accuracy": float(accuracy_score(test_y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "recall_rest": float(matrix[0, 0]),
        "recall_meditation": float(matrix[1, 1]),
        "recall_music": float(matrix[2, 2]),
        "recall_movie": float(matrix[3, 3]),
        "train_n": int(train_y.size),
        "test_n": int(test_y.size),
    }


def random_frame_splits(
    y: np.ndarray,
    repeats: int,
    test_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=test_fraction,
        random_state=seed,
    )
    return [
        (train, test, fold)
        for fold, (train, test) in enumerate(splitter.split(np.arange(y.size), y))
    ]


def purged_block_splits(
    y: np.ndarray,
    run_slices: list[tuple[int, int]],
    folds: int,
    gap: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    result = []
    all_indices = np.arange(y.size)
    for fold in range(folds):
        test_mask = np.zeros(y.size, dtype=bool)
        purge_mask = np.zeros(y.size, dtype=bool)
        for start, stop in run_slices:
            length = stop - start
            test_start = start + fold * length // folds
            test_stop = start + (fold + 1) * length // folds
            test_mask[test_start:test_stop] = True
            purge_mask[max(start, test_start - gap) : min(stop, test_stop + gap)] = True
        train = all_indices[~purge_mask]
        test = all_indices[test_mask]
        result.append((train, test, fold))
    return result


def evaluate_splits(
    data: dict,
    subject: str,
    session: str,
    feature: str,
    validation: str,
    splits: list[tuple[np.ndarray, np.ndarray, int]],
) -> list[dict]:
    rows = []
    for train, test, fold in splits:
        score = score_model(
            data[feature][train],
            data["y"][train],
            data[feature][test],
            data["y"][test],
        )
        rows.append(
            {
                "subject": subject,
                "session": session,
                "train_session": session,
                "test_session": session,
                "feature_set": feature,
                "validation": validation,
                "model": "linear_svm",
                "fold": fold,
                **score,
            }
        )
    return rows


def analyze_within_session(arguments: tuple) -> list[dict]:
    subject, session, task_paths, length_key, random_repeats, test_fraction, blocked_folds, purge_frames, seed = arguments
    data = load_temporal_session(task_paths, length_key)
    local_seed = deterministic_seed(seed, subject, session)
    random_splits = random_frame_splits(data["y"], random_repeats, test_fraction, local_seed)
    blocked_splits = purged_block_splits(data["y"], data["run_slices"], blocked_folds, purge_frames)
    rows = []
    for feature in FEATURES:
        rows.extend(
            evaluate_splits(
                data,
                subject,
                session,
                feature,
                "random_stratified_frames",
                random_splits,
            )
        )
        rows.extend(
            evaluate_splits(
                data,
                subject,
                session,
                feature,
                f"purged_contiguous_blocks_gap{purge_frames}",
                blocked_splits,
            )
        )
    return rows


def analyze_cross_visit(arguments: tuple) -> list[dict]:
    subject, session_paths, length_key = arguments
    data = {
        session: load_temporal_session(session_paths[session], length_key)
        for session in SESSIONS
    }
    rows = []
    directions = (("ses-01", "ses-02"), ("ses-02", "ses-01"))
    for fold, (train_session, test_session) in enumerate(directions):
        train_data = data[train_session]
        test_data = data[test_session]
        for feature in FEATURES:
            score = score_model(
                train_data[feature],
                train_data["y"],
                test_data[feature],
                test_data["y"],
            )
            rows.append(
                {
                    "subject": subject,
                    "session": "cross_session",
                    "train_session": train_session,
                    "test_session": test_session,
                    "feature_set": feature,
                    "validation": "cross_session_independent_acquisition_fixed_order",
                    "model": "linear_svm",
                    "fold": fold,
                    **score,
                }
            )
    return rows


def participant_folds(subjects: list[str], folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    shuffled = np.asarray(sorted(subjects, key=numeric_subject), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    result = []
    for test_chunk in np.array_split(shuffled, folds):
        test = sorted(map(str, test_chunk), key=numeric_subject)
        test_set = set(test)
        train = [subject for subject in sorted(subjects, key=numeric_subject) if subject not in test_set]
        result.append((train, test))
    return result


def uniformly_spaced_class_indices(y: np.ndarray, frames_per_class: int) -> np.ndarray:
    selected = []
    for label in range(len(TASKS)):
        available = np.flatnonzero(y == label)
        positions = np.linspace(0, available.size - 1, frames_per_class, dtype=int)
        selected.append(available[positions])
    return np.sort(np.concatenate(selected))


def participant_held_out_rows(
    sessions: dict[tuple[str, str], dict[str, Path]],
    length_key: str,
    folds: int,
    train_frames_per_class: int,
    seed: int,
) -> list[dict]:
    rows = []
    for session_index, session in enumerate(SESSIONS):
        subjects = sorted(
            [subject for subject, local_session in sessions if local_session == session],
            key=numeric_subject,
        )
        data = {
            subject: load_temporal_session(sessions[subject, session], length_key)
            for subject in subjects
        }
        validation = f"grouped_{folds}fold_participant_held_out_same_session_fixed_order"
        for fold, (train_subjects, test_subjects) in enumerate(
            participant_folds(subjects, folds, seed + 40000 + session_index * 1000)
        ):
            frames_per_class = min(
                train_frames_per_class,
                min(
                    int(np.bincount(data[subject]["y"], minlength=len(TASKS)).min())
                    for subject in train_subjects
                ),
            )
            train_indices = {
                subject: uniformly_spaced_class_indices(data[subject]["y"], frames_per_class)
                for subject in train_subjects
            }
            for feature in FEATURES:
                train_x = np.concatenate(
                    [data[subject][feature][train_indices[subject]] for subject in train_subjects]
                )
                train_y = np.concatenate(
                    [data[subject]["y"][train_indices[subject]] for subject in train_subjects]
                )
                model = classifier()
                model.fit(train_x, train_y)
                for subject in test_subjects:
                    test_data = data[subject]
                    prediction = model.predict(test_data[feature])
                    matrix = confusion_matrix(
                        test_data["y"],
                        prediction,
                        labels=np.arange(4),
                        normalize="true",
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "session": session,
                            "train_session": session,
                            "test_session": session,
                            "feature_set": feature,
                            "validation": validation,
                            "model": "linear_svm",
                            "fold": fold,
                            "accuracy": float(accuracy_score(test_data["y"], prediction)),
                            "balanced_accuracy": float(
                                balanced_accuracy_score(test_data["y"], prediction)
                            ),
                            "recall_rest": float(matrix[0, 0]),
                            "recall_meditation": float(matrix[1, 1]),
                            "recall_music": float(matrix[2, 2]),
                            "recall_movie": float(matrix[3, 3]),
                            "train_n": int(train_y.size),
                            "test_n": int(test_data["y"].size),
                        }
                    )
    return rows


def aggregate_fold_rows(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "subject",
        "session",
        "train_session",
        "test_session",
        "feature_set",
        "validation",
        "model",
    ]
    measures = [
        "accuracy",
        "balanced_accuracy",
        "recall_rest",
        "recall_meditation",
        "recall_music",
        "recall_movie",
        "train_n",
        "test_n",
    ]
    grouped = frame.groupby(identifiers, dropna=False)[measures]
    means = grouped.mean().add_suffix("_mean")
    standard_deviations = grouped.std(ddof=1).add_suffix("_sd")
    counts = grouped.size().rename("fold_n")
    return pd.concat([means, standard_deviations, counts], axis=1).reset_index()


def group_statistics(participant: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        participant.groupby(["subject", "feature_set", "validation"], as_index=False)[
            "balanced_accuracy_mean"
        ]
        .mean()
    )
    rows = []
    for (feature, validation), frame in averaged.groupby(["feature_set", "validation"]):
        values = frame["balanced_accuracy_mean"].to_numpy(dtype=np.float64)
        n = values.size
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / math.sqrt(n))
        critical = float(stats.t.ppf(0.975, n - 1))
        test = stats.ttest_1samp(values, 0.25)
        rows.append(
            {
                "feature_set": feature,
                "validation": validation,
                "participant_n": n,
                "balanced_accuracy_mean": mean,
                "balanced_accuracy_median": float(np.median(values)),
                "balanced_accuracy_ci95_low": mean - critical * standard_error,
                "balanced_accuracy_ci95_high": mean + critical * standard_error,
                "one_sample_t_vs_chance": float(test.statistic),
                "one_sample_p_two_sided": float(test.pvalue),
            }
        )
    result = pd.DataFrame(rows)
    result["one_sample_q_bh"] = multipletests(result["one_sample_p_two_sided"], method="fdr_bh")[1]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--length-key", default="author_literal")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--blocked-folds", type=int, default=4)
    parser.add_argument("--purge-frames", type=int, default=20)
    parser.add_argument("--participant-folds", type=int, default=5)
    parser.add_argument("--train-frames-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    sessions = discover_sessions(args.volume_root)
    within_jobs = [
        (
            subject,
            session,
            {task: str(path) for task, path in paths.items()},
            args.length_key,
            args.random_repeats,
            args.test_fraction,
            args.blocked_folds,
            args.purge_frames,
            args.seed,
        )
        for (subject, session), paths in sessions.items()
    ]
    cross_subjects = sorted(
        {
            subject
            for subject, _ in sessions
            if all((subject, session) in sessions for session in SESSIONS)
        },
        key=numeric_subject,
    )
    cross_jobs = [
        (
            subject,
            {
                session: {
                    task: str(path)
                    for task, path in sessions[subject, session].items()
                }
                for session in SESSIONS
            },
            args.length_key,
        )
        for subject in cross_subjects
    ]
    if args.workers == 1:
        within_results = [analyze_within_session(job) for job in within_jobs]
        cross_results = [analyze_cross_visit(job) for job in cross_jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            within_results = list(pool.map(analyze_within_session, within_jobs, chunksize=1))
            cross_results = list(pool.map(analyze_cross_visit, cross_jobs, chunksize=1))
    fold_rows = [row for result in within_results for row in result]
    fold_rows.extend(row for result in cross_results for row in result)
    fold_rows.extend(
        participant_held_out_rows(
            sessions,
            args.length_key,
            args.participant_folds,
            args.train_frames_per_class,
            args.seed,
        )
    )
    fold_frame = pd.DataFrame(fold_rows)
    participant = aggregate_fold_rows(fold_frame)
    group = group_statistics(participant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(args.output_dir / "context_time_control_fold_metrics.csv", index=False)
    participant.to_csv(args.output_dir / "context_time_control_participant_metrics.csv", index=False)
    group.to_csv(args.output_dir / "context_time_control_group_statistics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
