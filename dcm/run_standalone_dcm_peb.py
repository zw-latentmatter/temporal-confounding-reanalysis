from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat, savemat


EXPECTED_ROIS = ["aHip_L", "aHip_R", "IPC_L", "IPC_R", "mPFC", "PCC"]
BMC_RNG_SEED = 20260825
Z90 = 1.6448536269514722
ANALYSES = {
    "author_stacked_01": {
        "coding": [0.0, 1.0],
        "effect_column": 2,
        "effect_prefix": "",
        "group_Q": "all",
    },
    "stacked_centered_pm05": {
        "coding": [-0.5, 0.5],
        "effect_column": 2,
        "effect_prefix": "",
        "group_Q": "all",
    },
    "paired_peb_of_pebs": {
        "coding": [-0.5, 0.5],
        "effect_column": 1,
        "effect_prefix": "drug_psilocybin_minus_baseline",
        "group_Q": "single",
    },
}
A_PATTERN = re.compile(r"A\s*[({\[]\s*(\d+)\s*,\s*(\d+)\s*[)}\]]")
EFFECT_FIELDS = [
    "effect",
    "parameter_index",
    "spm_parameter",
    "to_index",
    "from_index",
    "from_roi",
    "to_roi",
    "is_self",
    "units",
    "full_ep",
    "full_var",
    "full_sd",
    "full_ci90_low",
    "full_ci90_high",
    "full_p_direction",
    "bma_ep",
    "bma_var",
    "bma_sd",
    "bma_ci90_low",
    "bma_ci90_high",
    "bma_p_direction",
    "bmr_pp",
    "included_pp99",
]
SUMMARY_FIELDS = [
    "branch",
    "task",
    "analysis",
    "n",
    "selected_edge_count_all",
    "selected_edge_count_offdiag",
    "reanalysis_total_abs_offdiag_pp99",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matlab_cellstr(values: Iterable[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object).reshape(-1, 1)


def matlab_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, np.str_)):
        return [str(value).strip()]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace").strip()]
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return []
    result: list[str] = []
    for item in array.reshape(-1):
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.reshape(-1)[0]
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="replace")
        elif isinstance(item, np.ndarray):
            text = "".join(str(part) for part in item.reshape(-1))
        else:
            text = str(item)
        result.append(text.strip())
    return result


def numeric_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1, order="F")


def effect_vector(value: Any, n_parameters: int, effect_column: int) -> np.ndarray:
    vector = numeric_vector(value)
    if n_parameters == 0 or vector.size % n_parameters:
        raise ValueError("Effect-vector dimensions are incompatible")
    n_effects = vector.size // n_parameters
    if effect_column < 1 or effect_column > n_effects:
        raise ValueError("Effect column is outside the PEB design")
    start = (effect_column - 1) * n_parameters
    return vector[start : start + n_parameters]


def effect_variance(
    value: Any, n_parameters: int, n_effects: int, effect_column: int
) -> np.ndarray:
    expected = n_parameters * n_effects
    covariance = np.asarray(value, dtype=float)
    squeezed = np.squeeze(covariance)
    if squeezed.ndim == 1 and squeezed.size == expected:
        diagonal = squeezed.reshape(-1, order="F")
    elif covariance.shape == (expected, expected):
        diagonal = np.diag(covariance)
    elif covariance.shape == (n_parameters, n_parameters) and n_effects == 1:
        diagonal = np.diag(covariance)
    else:
        raise ValueError(f"Unexpected posterior covariance shape {covariance.shape}")
    start = (effect_column - 1) * n_parameters
    return diagonal[start : start + n_parameters]


def inclusion_probability(
    value: Any, n_parameters: int, n_effects: int, effect_column: int
) -> np.ndarray:
    expected = n_parameters * n_effects
    probabilities = np.asarray(value, dtype=float)
    squeezed = np.squeeze(probabilities)
    if squeezed.ndim == 1 and squeezed.size == expected:
        vector = squeezed.reshape(-1, order="F")
    elif probabilities.shape == (expected, expected):
        vector = np.diag(probabilities)
    else:
        raise ValueError(f"Unexpected inclusion-probability shape {probabilities.shape}")
    start = (effect_column - 1) * n_parameters
    return vector[start : start + n_parameters]


def directional_probability(effect: float, standard_deviation: float) -> float:
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        return 1.0 if effect != 0 else 0.5
    return 0.5 * math.erfc(-abs(effect / standard_deviation) / math.sqrt(2.0))


def parse_a_indices(parameter_name: str) -> tuple[int, int]:
    match = A_PATTERN.search(parameter_name)
    if match is None:
        raise ValueError(f"Cannot parse A-matrix indices from {parameter_name}")
    return int(match.group(1)), int(match.group(2))


def export_effects(
    plain_mat: Path, output_csv: Path, threshold: float
) -> list[dict[str, Any]]:
    data = loadmat(plain_mat, simplify_cells=True)
    pnames = matlab_strings(data["Pnames"])
    effect_column = int(np.asarray(data["effect_column"]).reshape(-1)[0])
    effect_prefix_values = matlab_strings(data.get("effect_prefix", ""))
    effect_prefix = effect_prefix_values[0] if effect_prefix_values else ""
    full_all = numeric_vector(data["PEB_Ep"])
    if full_all.size % len(pnames):
        raise ValueError("PEB effect dimensions are incompatible with parameter names")
    n_effects = full_all.size // len(pnames)
    full_ep = effect_vector(data["PEB_Ep"], len(pnames), effect_column)
    bma_ep = effect_vector(data["BMA_Ep"], len(pnames), effect_column)
    full_var = effect_variance(data["PEB_Cp"], len(pnames), n_effects, effect_column)
    bma_var = effect_variance(data["BMA_Cp"], len(pnames), n_effects, effect_column)
    bmr_pp = inclusion_probability(data["BMA_Pp"], len(pnames), n_effects, effect_column)
    if effect_prefix:
        keep = [
            index
            for index, name in enumerate(pnames)
            if name.startswith(effect_prefix + ":") or name.startswith(effect_prefix)
        ]
    else:
        keep = list(range(len(pnames)))
    if len(keep) != 36:
        raise ValueError(f"Expected 36 selected A parameters, found {len(keep)}")
    rows: list[dict[str, Any]] = []
    for index in keep:
        to_index, from_index = parse_a_indices(pnames[index])
        full_variance = max(float(full_var[index]), 0.0)
        bma_variance = max(float(bma_var[index]), 0.0)
        full_sd = math.sqrt(full_variance)
        bma_sd = math.sqrt(bma_variance)
        full_mean = float(full_ep[index])
        bma_mean = float(bma_ep[index])
        probability = float(bmr_pp[index])
        rows.append(
            {
                "effect": effect_prefix,
                "parameter_index": index + 1,
                "spm_parameter": pnames[index],
                "to_index": to_index,
                "from_index": from_index,
                "from_roi": EXPECTED_ROIS[from_index - 1],
                "to_roi": EXPECTED_ROIS[to_index - 1],
                "is_self": int(to_index == from_index),
                "units": "log scaling" if to_index == from_index else "Hz",
                "full_ep": full_mean,
                "full_var": full_variance,
                "full_sd": full_sd,
                "full_ci90_low": full_mean - Z90 * full_sd,
                "full_ci90_high": full_mean + Z90 * full_sd,
                "full_p_direction": directional_probability(full_mean, full_sd),
                "bma_ep": bma_mean,
                "bma_var": bma_variance,
                "bma_sd": bma_sd,
                "bma_ci90_low": bma_mean - Z90 * bma_sd,
                "bma_ci90_high": bma_mean + Z90 * bma_sd,
                "bma_p_direction": directional_probability(bma_mean, bma_sd),
                "bmr_pp": probability,
                "included_pp99": int(probability > threshold),
            }
        )
    write_csv(output_csv, rows, EFFECT_FIELDS)
    return rows


def write_design(
    output_dir: Path,
    analysis: str,
    subjects: list[str],
    baseline_session: str,
    drug_session: str,
) -> None:
    if analysis in {"author_stacked_01", "stacked_centered_pm05"}:
        coding = [0.0, 1.0] if analysis == "author_stacked_01" else [-0.5, 0.5]
        intercept = "baseline_intercept" if analysis == "author_stacked_01" else "grand_mean"
        rows = [
            {
                "subject": subject,
                "session": session,
                intercept: 1,
                "drug_psilocybin_minus_baseline": drug_code,
            }
            for subject in subjects
            for session, drug_code in zip(
                (baseline_session, drug_session), coding, strict=True
            )
        ]
        write_csv(
            output_dir / "design_matrix.csv",
            rows,
            ["subject", "session", intercept, "drug_psilocybin_minus_baseline"],
        )
    else:
        write_csv(
            output_dir / "participant_design_matrix.csv",
            [
                {
                    "session": baseline_session,
                    "subject_mean": 1,
                    "drug_psilocybin_minus_baseline": -0.5,
                },
                {
                    "session": drug_session,
                    "subject_mean": 1,
                    "drug_psilocybin_minus_baseline": 0.5,
                },
            ],
            ["session", "subject_mean", "drug_psilocybin_minus_baseline"],
        )
        write_csv(
            output_dir / "group_design_matrix.csv",
            [{"subject": subject, "group_mean": 1} for subject in subjects],
            ["subject", "group_mean"],
        )


def run_analysis(
    rows: list[dict[str, str]],
    branch: str,
    task: str,
    analysis: str,
    dcm_root: Path,
    output_root: Path,
    spm_runner: Path,
    mcr_home: Path,
    script: Path,
    threads_per_job: int,
    threshold: float,
) -> dict[str, Any]:
    spec = ANALYSES[analysis]
    output_dir = output_root / safe_name(branch) / safe_name(task) / analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_peb_dir = output_dir / "subject_pebs"
    subject_peb_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "PEB_BMA_results.mat"
    plain_file = output_dir / "PEB_BMA_effects.mat"
    input_paths = np.asarray(
        [
            [str((dcm_root / row["baseline_path"]).resolve()), str((dcm_root / row["drug_path"]).resolve())]
            for row in rows
        ],
        dtype=object,
    )
    baseline_session = rows[0]["baseline_session"]
    drug_session = rows[0]["drug_session"]
    with tempfile.TemporaryDirectory(prefix="peb-", dir=output_dir) as temporary_text:
        temporary = Path(temporary_text)
        job_file = temporary / "job.mat"
        job = {
            "input_paths": input_paths,
            "condition_sessions": matlab_cellstr([baseline_session, drug_session]),
            "condition_tasks": matlab_cellstr([task, task]),
            "subjects": matlab_cellstr(row["subject"] for row in rows),
            "branch": branch,
            "task": task,
            "analysis_name": analysis,
            "baseline_session": baseline_session,
            "drug_session": drug_session,
            "output_file": str(result_file.resolve()),
            "plain_output_file": str(plain_file.resolve()),
            "subject_peb_dir": str(subject_peb_dir.resolve()),
            "coding": np.asarray(spec["coding"], dtype=float),
            "participant_X": np.asarray([[1.0, -0.5], [1.0, 0.5]], dtype=float),
            "participant_Xnames": matlab_cellstr(
                ["subject_mean", "drug_psilocybin_minus_baseline"]
            ),
            "effect_column": np.asarray([[spec["effect_column"]]], dtype=np.int32),
            "effect_prefix": spec["effect_prefix"],
            "group_Q": spec["group_Q"],
            "bmc_rng_seed": np.asarray([[BMC_RNG_SEED]], dtype=np.int64),
            "expected_roi_names": matlab_cellstr(EXPECTED_ROIS),
        }
        savemat(job_file, job, do_compression=True, long_field_names=True)
        environment = os.environ.copy()
        environment["DCM_PEB_JOB"] = str(job_file.resolve())
        environment["OMP_NUM_THREADS"] = str(threads_per_job)
        environment["MKL_NUM_THREADS"] = str(threads_per_job)
        environment["OPENBLAS_NUM_THREADS"] = str(threads_per_job)
        environment["TMPDIR"] = str(temporary.resolve())
        environment["MATLAB_PREFDIR"] = str(temporary.resolve())
        command = [str(spm_runner), str(mcr_home), "script", str(script)]
        process = subprocess.run(
            command,
            cwd=temporary,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if process.returncode:
            raise RuntimeError(f"SPM job failed with exit code {process.returncode}")
    effects = export_effects(plain_file, output_dir / "drug_effects.csv", threshold)
    plain = loadmat(plain_file, simplify_cells=True)
    subjects = matlab_strings(plain.get("included_subjects"))
    failed_subjects = matlab_strings(plain.get("failed_subjects"))
    failed_sessions = matlab_strings(plain.get("failed_sessions"))
    failed_error_types = matlab_strings(plain.get("failed_error_types"))
    failure_rows = [
        {
            "subject": subject,
            "session": session,
            "error_type": error_type,
        }
        for subject, session, error_type in zip(
            failed_subjects, failed_sessions, failed_error_types, strict=True
        )
    ]
    write_csv(
        output_dir / "failed_samples.csv",
        failure_rows,
        ["subject", "session", "error_type"],
    )
    write_design(output_dir, analysis, subjects, baseline_session, drug_session)
    selected = [row for row in effects if float(row["bmr_pp"]) > threshold]
    selected_offdiag = [row for row in selected if not int(row["is_self"])]
    return {
        "branch": branch,
        "task": task,
        "analysis": analysis,
        "n": len(subjects),
        "selected_edge_count_all": len(selected),
        "selected_edge_count_offdiag": len(selected_offdiag),
        "reanalysis_total_abs_offdiag_pp99": sum(
            abs(float(row["bma_ep"])) for row in selected_offdiag
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dcm-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spm-runner", required=True, type=Path)
    parser.add_argument("--mcr-home", required=True, type=Path)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("run_group_dcm_peb_job_standalone.m"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--posterior-probability-threshold", type=float, default=0.99)
    parser.add_argument("--branches", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--analyses", nargs="+", choices=tuple(ANALYSES), default=tuple(ANALYSES))
    args = parser.parse_args()
    manifest = read_csv(args.manifest.resolve())
    branches = args.branches or sorted({row["branch"] for row in manifest})
    tasks = args.tasks or sorted({row["task"] for row in manifest})
    jobs = [
        (
            [row for row in manifest if row["branch"] == branch and row["task"] == task],
            branch,
            task,
            analysis,
        )
        for branch in branches
        for task in tasks
        for analysis in dict.fromkeys(args.analyses)
    ]
    jobs = [job for job in jobs if len(job[0]) >= 2]
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(
            pool.map(
                lambda job: run_analysis(
                    job[0],
                    job[1],
                    job[2],
                    job[3],
                    args.dcm_root.resolve(),
                    output_root,
                    args.spm_runner.resolve(),
                    args.mcr_home.resolve(),
                    args.script.resolve(),
                    args.threads_per_job,
                    args.posterior_probability_threshold,
                ),
                jobs,
            )
        )
    summaries.sort(key=lambda row: (row["branch"], row["task"], row["analysis"]))
    write_csv(output_root / "dcm_peb_compact_summary.csv", summaries, SUMMARY_FIELDS)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
