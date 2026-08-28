from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SESSIONS = ("ses-01", "ses-02")
METRICS = (
    "silhouette",
    "context_neighbour_fraction",
    "context_neighbour_excess",
    "nearest_neighbour_time_delta",
    "random_pair_time_delta",
    "nearest_neighbour_time_ratio",
)


def balanced_indices(labels: np.ndarray, maximum_per_class: int) -> np.ndarray:
    selected = []
    for label in range(4):
        candidates = np.flatnonzero(labels == label)
        count = min(maximum_per_class, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count)
        selected.append(candidates[np.rint(positions).astype(int)])
    return np.concatenate(selected)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min())
    span = float(values.max() - minimum)
    if span == 0:
        return np.zeros_like(values)
    return (values - minimum) / span


def latent_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    normalized_time: np.ndarray,
    maximum_per_class: int,
    neighbours: int,
) -> dict[str, float | int]:
    chosen = balanced_indices(labels, maximum_per_class)
    x = np.asarray(embedding[chosen], dtype=np.float64)
    y = np.asarray(labels[chosen], dtype=np.int8)
    t = np.asarray(normalized_time[chosen], dtype=np.float64)
    distances = np.sqrt(
        np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2), 0.0)
    )
    identity = np.eye(len(y), dtype=bool)
    same = y[:, None] == y[None, :]
    within = np.asarray(
        [distances[index, same[index] & ~identity[index]].mean() for index in range(len(y))]
    )
    between = np.asarray(
        [
            min(
                distances[index, y == other].mean()
                for other in range(4)
                if other != y[index]
            )
            for index in range(len(y))
        ]
    )
    denominator = np.maximum(within, between)
    silhouette = np.divide(
        between - within,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    ).mean()
    nearest = np.argsort(distances, axis=1)[:, 1 : neighbours + 1]
    context_fraction = float(np.mean(y[nearest] == y[:, None]))
    class_frequency = np.bincount(y, minlength=4) / len(y)
    context_chance = float(np.sum(class_frequency**2))
    context_excess = (context_fraction - context_chance) / (1.0 - context_chance)
    time_distances = np.abs(t[:, None] - t[None, :])
    nearest_time_delta = float(
        np.take_along_axis(time_distances, nearest, axis=1).mean()
    )
    random_pair_time_delta = float(time_distances[~identity].mean())
    return {
        "silhouette": float(silhouette),
        "context_neighbour_fraction": context_fraction,
        "context_neighbour_excess": float(context_excess),
        "nearest_neighbour_time_delta": nearest_time_delta,
        "random_pair_time_delta": random_pair_time_delta,
        "nearest_neighbour_time_ratio": nearest_time_delta / random_pair_time_delta,
        "source_frame_n": int(len(labels)),
        "sampled_frame_n": int(len(chosen)),
    }


def parse_archive_name(path: Path) -> dict[str, str | int]:
    name = path.parent.name
    session = re.search(r"ses-0[12]", name)
    seed = re.search(r"seed-(\d+)", name)
    fold = re.search(r"fold-(\d+)", name)
    subject = re.search(r"sub-PC\d+", name)
    return {
        "session": session.group() if session else "",
        "seed": int(seed.group(1)) if seed else -1,
        "fold": int(fold.group(1)) if fold else -1,
        "archive_subject": subject.group() if subject else "",
    }


def archive_units(
    archive: Path,
    regime: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    metadata = parse_archive_name(archive)
    arrays = np.load(archive, allow_pickle=False)
    if regime == "random_frame":
        embedding = arrays["embedding_float32"]
        labels = arrays["embedding_labels"]
        units = [
            (
                str(metadata["archive_subject"]),
                np.arange(len(labels)),
                np.linspace(0.0, 1.0, len(labels)),
            )
        ]
    elif regime == "blocked":
        embedding = arrays["test_embedding_float32"]
        labels = arrays["test_embedding_labels"]
        units = [
            (
                str(metadata["archive_subject"]),
                np.arange(len(labels)),
                normalize(arrays["test_embedding_centers"]),
            )
        ]
    else:
        embedding = arrays["test_embedding_float32"]
        labels = arrays["test_embedding_labels"]
        subjects = arrays["test_embedding_subject"].astype(str)
        units = []
        for subject in sorted(np.unique(subjects)):
            indices = np.flatnonzero(subjects == subject)
            units.append(
                (subject, indices, np.linspace(0.0, 1.0, len(indices)))
            )
    return embedding, labels, units


def extract_rows(
    roots: dict[str, Path],
    maximum_per_class: int,
    neighbours: int,
) -> pd.DataFrame:
    rows = []
    for regime, root in roots.items():
        for archive in sorted(root.rglob("*.npz")):
            metadata = parse_archive_name(archive)
            embedding, labels, units = archive_units(archive, regime)
            for subject, indices, local_time in units:
                row = {
                    "regime": regime,
                    "subject": subject,
                    "session": metadata["session"],
                    "fold": metadata["fold"],
                    "seed": metadata["seed"],
                    "archive_id": archive.relative_to(root).as_posix(),
                }
                row.update(
                    latent_metrics(
                        np.asarray(embedding[indices]),
                        np.asarray(labels[indices]),
                        local_time,
                        maximum_per_class,
                        neighbours,
                    )
                )
                rows.append(row)
    return pd.DataFrame(rows)


def participant_metrics(archive_metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {metric: (metric, "median") for metric in METRICS}
    aggregations.update(
        {
            "source_frame_n": ("source_frame_n", "median"),
            "sampled_frame_n": ("sampled_frame_n", "median"),
        }
    )
    result = (
        archive_metrics.groupby(["regime", "subject", "session"], as_index=False)
        .agg(fit_n=("archive_id", "size"), **aggregations)
        .sort_values(["regime", "subject", "session"])
    )
    return result


def group_statistics(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, session), group in participants.groupby(
        ["regime", "session"], sort=True
    ):
        for metric in METRICS:
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            low, high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=mean,
                scale=stats.sem(values),
            )
            rows.append(
                {
                    "regime": regime,
                    "session": session,
                    "metric": metric,
                    "n_participants": int(len(values)),
                    "mean": mean,
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
            )
    return pd.DataFrame(rows)


def paired_visit_differences(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in participants.groupby("regime", sort=True):
        for metric in METRICS:
            paired = group.pivot(index="subject", columns="session", values=metric).dropna()
            differences = paired[SESSIONS[1]] - paired[SESSIONS[0]]
            for subject, value in differences.items():
                rows.append(
                    {
                        "regime": regime,
                        "subject": subject,
                        "metric": metric,
                        "session_02_minus_session_01": float(value),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-frame-dir", required=True, type=Path)
    parser.add_argument("--blocked-dir", required=True, type=Path)
    parser.add_argument("--participant-heldout-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-per-class", type=int, default=40)
    parser.add_argument("--nearest-neighbours", type=int, default=15)
    args = parser.parse_args()
    roots = {
        "random_frame": args.random_frame_dir,
        "blocked": args.blocked_dir,
        "participant_heldout": args.participant_heldout_dir,
    }
    archive = extract_rows(
        roots,
        args.maximum_per_class,
        args.nearest_neighbours,
    )
    participants = participant_metrics(archive)
    groups = group_statistics(participants)
    paired = paired_visit_differences(participants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive.to_csv(
        args.output_dir / "latent_geometry_archive_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    participants.to_csv(
        args.output_dir / "latent_geometry_participant_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    groups.to_csv(
        args.output_dir / "latent_geometry_group_statistics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    paired.to_csv(
        args.output_dir / "latent_geometry_paired_visit_differences.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
