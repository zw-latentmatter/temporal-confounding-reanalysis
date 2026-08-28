from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit
from scipy.io import loadmat
from scipy.signal import welch


@njit(cache=True)
def lz76_matlab_port(sequence: np.ndarray) -> int:
    n = len(sequence)
    if n == 0:
        return 0
    i = 0
    k = 1
    location = 1
    complexity = 1
    maximum = 1
    while True:
        if sequence[i + k - 1] == sequence[location + k - 1]:
            k += 1
            if location + k > n:
                complexity += 1
                break
        else:
            if k > maximum:
                maximum = k
            i += 1
            if i == location:
                complexity += 1
                location += maximum
                if location + 1 > n:
                    break
                i = 0
                k = 1
                maximum = 1
            else:
                k = 1
    return complexity


def parse_entities(path: Path) -> tuple[str, str, str]:
    match = re.match(r"(sub-PC\d+)_(ses-\d+)_task-([^_]+)_", path.name)
    if match is None:
        raise ValueError(path.name)
    return match.group(1), match.group(2), match.group(3)


def decode_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([str(value) for value in np.ravel(labels)], dtype="U32")


def alpha_mask(frequencies: np.ndarray) -> np.ndarray:
    mask = (frequencies >= 8.0 - 1e-8) & (frequencies <= 12.0 + 1e-8)
    return mask & np.isclose(frequencies, np.round(frequencies), atol=1e-7)


def extract_one(arguments: tuple[str, str]) -> str:
    source_text, output_text = arguments
    source = Path(source_text)
    output_root = Path(output_text)
    subject, session, task = parse_entities(source)
    destination = output_root / subject / session / f"task-{task}_eeg-features.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldtrip = loadmat(source, simplify_cells=True, variable_names=["ftData"])["ftData"]
    signal = np.asarray(fieldtrip["trial"], dtype=np.float64)
    sampling_rate = float(fieldtrip["fsample"])
    labels = decode_labels(fieldtrip["label"])
    frequencies, power = welch(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=int(round(2 * sampling_rate)),
        noverlap=int(round(sampling_rate)),
        detrend="constant",
        scaling="density",
        axis=1,
    )
    support = (frequencies >= 1.0) & (frequencies <= 80.0)
    frequencies = frequencies[support]
    power = power[:, support]
    linear_alpha = power[:, alpha_mask(frequencies)].mean(axis=1)
    alpha_db = 10.0 * np.log10(np.maximum(linear_alpha, np.finfo(float).tiny))
    binary = signal > signal.mean(axis=1, keepdims=True)
    complexity_counts = np.asarray([lz76_matlab_port(binary[index]) for index in range(signal.shape[0])], dtype=np.int32)
    sample_count = signal.shape[1]
    corrected_lz = complexity_counts.astype(np.float64) * np.log2(sample_count) / sample_count
    np.savez_compressed(
        destination,
        frequencies=frequencies.astype(np.float32),
        power_spectral_density=power.astype(np.float32),
        alpha_linear_integer_8_12=linear_alpha.astype(np.float32),
        alpha_db_integer_8_12=alpha_db.astype(np.float32),
        corrected_lz_normalized_by_sequence_length=corrected_lz.astype(np.float32),
        channel_labels=labels,
    )
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    sources = sorted(arguments.input_root.rglob("*_Clean-ft.mat"))
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(str(source), str(arguments.output_root)) for source in sources]
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        list(executor.map(extract_one, jobs, chunksize=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
