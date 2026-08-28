from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("rest", "meditation", "music", "movie")
SESSIONS = ("ses-01", "ses-02")


def load_spectra(feature_root: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    rows = []
    spectra = []
    frequencies = None
    for path in sorted(feature_root.glob("sub-*/ses-*/task-*_eeg-features.npz")):
        subject = path.parent.parent.name
        session = path.parent.name
        task = path.stem.removeprefix("task-").removesuffix("_eeg-features")
        with np.load(path, allow_pickle=False) as archive:
            current_frequencies = np.asarray(archive["frequencies"], dtype=np.float64)
            current_spectrum = np.asarray(archive["power_spectral_density"], dtype=np.float64).mean(axis=0)
        if frequencies is None:
            frequencies = current_frequencies
        rows.append({"subject": subject, "session": session, "task": task})
        spectra.append(current_spectrum)
    return np.asarray(frequencies), pd.DataFrame(rows), np.asarray(spectra)


def complete_participants(runs: pd.DataFrame) -> list[str]:
    groups = [
        set(runs.loc[(runs.session == session) & (runs.task == task), "subject"])
        for session in SESSIONS
        for task in TASKS
    ]
    return sorted(set.intersection(*groups))


def participant_gaps(runs: pd.DataFrame, spectra: np.ndarray, participants: list[str]) -> dict[str, np.ndarray]:
    lookup = {
        (row.subject, row.session, row.task): index
        for index, row in enumerate(runs.itertuples(index=False))
    }
    result = {}
    for session in SESSIONS:
        values = []
        for subject in participants:
            closed = np.mean([spectra[lookup[(subject, session, task)]] for task in TASKS[:3]], axis=0)
            values.append(closed - spectra[lookup[(subject, session, "movie")]])
        result[session] = np.asarray(values)
    return result


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    draws = np.empty((repetitions, values.shape[1]), dtype=np.float32)
    for start in range(0, repetitions, 250):
        stop = min(start + 250, repetitions)
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5), axis=0)
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    frequencies, runs, spectra = load_spectra(arguments.feature_root)
    participants = complete_participants(runs)
    gaps = participant_gaps(runs, spectra, participants)
    keep = (frequencies >= 1.0) & (frequencies <= 25.0) & np.isclose(frequencies, np.round(frequencies), atol=1e-8)
    rows = []
    for session in SESSIONS:
        values = gaps[session]
        low, high = bootstrap_interval(values, arguments.bootstrap_repetitions, arguments.seed)
        for index in np.flatnonzero(keep):
            rows.append(
                {
                    "session": session,
                    "frequency_hz": float(frequencies[index]),
                    "participant_n": len(participants),
                    "mean_closed_minus_movie_power": float(values[:, index].mean()),
                    "bootstrap_ci95_low": float(low[index]),
                    "bootstrap_ci95_high": float(high[index]),
                    "bootstrap_repetitions": arguments.bootstrap_repetitions,
                    "seed": arguments.seed,
                }
            )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(arguments.output_dir / "eeg_spectral_gap_bootstrap.csv", index=False)
    np.savez_compressed(
        arguments.output_dir / "eeg_spectral_gap_participant_values.npz",
        frequencies=frequencies[keep].astype(np.float32),
        subjects=np.asarray(participants),
        baseline_closed_minus_movie=gaps["ses-01"][:, keep].astype(np.float32),
        psilocybin_closed_minus_movie=gaps["ses-02"][:, keep].astype(np.float32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
