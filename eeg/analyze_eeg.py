from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


TASKS = ("rest", "meditation", "music", "movie")
SESSIONS = ("ses-01", "ses-02")
SESSION_LABELS = {"ses-01": "baseline", "ses-02": "psilocybin"}
LINEAR_ALPHA = "linear_integer_8_12"
DECIBEL_ALPHA = "decibel_integer_8_12"
CORRECTED_LZ = "corrected_sequence_length"


def parse_entities(path: Path) -> tuple[str, str, str]:
    subject = path.parent.parent.name
    session = path.parent.name
    task = path.stem.removeprefix("task-").removesuffix("_eeg-features")
    return subject, session, task


def load_feature(path_text: str) -> dict:
    path = Path(path_text)
    subject, session, task = parse_entities(path)
    with np.load(path, allow_pickle=False) as archive:
        return {
            "subject": subject,
            "session": session,
            "task": task,
            "frequencies": np.asarray(archive["frequencies"], dtype=np.float64),
            "power": np.asarray(archive["power_spectral_density"], dtype=np.float64),
            "lz": np.asarray(archive["corrected_lz_normalized_by_sequence_length"], dtype=np.float64),
            "labels": np.asarray(archive["channel_labels"]).astype(str),
        }


def harmonize_channel_order(records: list[dict]) -> list[str]:
    reference = [label.upper() for label in records[0]["labels"]]
    for record in records:
        labels = [label.upper() for label in record["labels"]]
        if labels != reference:
            positions = {label: index for index, label in enumerate(labels)}
            order = np.asarray([positions[label] for label in reference])
            record["power"] = record["power"][order]
            record["lz"] = record["lz"][order]
            record["labels"] = np.asarray(reference)
    return reference


def alpha_mask(frequencies: np.ndarray) -> np.ndarray:
    mask = (frequencies >= 8.0 - 1e-8) & (frequencies <= 12.0 + 1e-8)
    return mask & np.isclose(frequencies, np.round(frequencies), atol=1e-7)


def build_tables(records: list[dict], channel_labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha_channel_parts = []
    alpha_run_rows = []
    spectrum_run_parts = []
    lz_channel_parts = []
    lz_run_rows = []
    channel_index = np.arange(len(channel_labels), dtype=np.int16)
    for record in records:
        common = {"subject": record["subject"], "session": record["session"], "task": record["task"]}
        frequencies = record["frequencies"]
        spectrum_run_parts.append(
            pd.DataFrame(
                {
                    **common,
                    "frequency_hz": frequencies,
                    "power_linear_mean_channels": record["power"].mean(axis=0),
                }
            )
        )
        linear_alpha = record["power"][:, alpha_mask(frequencies)].mean(axis=1)
        for scale, values, units in (
            (LINEAR_ALPHA, linear_alpha, "linear_power"),
            (DECIBEL_ALPHA, 10.0 * np.log10(np.maximum(linear_alpha, np.finfo(float).tiny)), "dB"),
        ):
            alpha_channel_parts.append(
                pd.DataFrame(
                    {
                        **common,
                        "scale": scale,
                        "units": units,
                        "channel_index": channel_index,
                        "channel": channel_labels,
                        "alpha": values,
                    }
                )
            )
            alpha_run_rows.append({**common, "scale": scale, "units": units, "alpha_mean_channels": float(values.mean())})
        lz_channel_parts.append(
            pd.DataFrame(
                {
                    **common,
                    "normalization": CORRECTED_LZ,
                    "channel_index": channel_index,
                    "channel": channel_labels,
                    "lz76": record["lz"],
                }
            )
        )
        lz_run_rows.append({**common, "normalization": CORRECTED_LZ, "lz76_mean_channels": float(record["lz"].mean())})
    return (
        pd.concat(alpha_channel_parts, ignore_index=True),
        pd.DataFrame(alpha_run_rows),
        pd.concat(spectrum_run_parts, ignore_index=True),
        pd.concat(lz_channel_parts, ignore_index=True),
        pd.DataFrame(lz_run_rows),
    )


def paired_values(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return frame.pivot(index="subject", columns="session", values=value).dropna(subset=list(SESSIONS)).rename(columns=SESSION_LABELS)


def rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def paired_statistics(pairs: pd.DataFrame, bootstrap_repetitions: int, seed: int) -> dict:
    baseline = pairs["baseline"].to_numpy(dtype=float)
    psilocybin = pairs["psilocybin"].to_numpy(dtype=float)
    delta = psilocybin - baseline
    participant_n = len(delta)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, participant_n, size=(bootstrap_repetitions, participant_n))
    bootstrap_means = delta[indices].mean(axis=1)
    delta_sd = float(delta.std(ddof=1))
    paired_t = stats.ttest_rel(psilocybin, baseline)
    wilcoxon = stats.wilcoxon(psilocybin, baseline, alternative="two-sided", zero_method="wilcox")
    return {
        "participant_n": participant_n,
        "baseline_mean": float(baseline.mean()),
        "psilocybin_mean": float(psilocybin.mean()),
        "percent_contraction_of_baseline_mean": float(100.0 * (baseline.mean() - psilocybin.mean()) / abs(baseline.mean())),
        "delta_psilocybin_minus_baseline": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "paired_cohen_dz": float(delta.mean() / delta_sd),
        "paired_hedges_gz": float(delta.mean() / delta_sd * (1.0 - 3.0 / (4.0 * participant_n - 5.0))),
        "paired_rank_biserial": rank_biserial(delta),
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_two_sided_p": float(paired_t.pvalue),
        "paired_wilcoxon_two_sided_p": float(wilcoxon.pvalue),
    }


def add_fdr(frame: pd.DataFrame, p_column: str, group_columns: list[str], output_column: str) -> pd.DataFrame:
    frame[output_column] = np.nan
    groups = frame.groupby(group_columns, dropna=False).groups.values() if group_columns else [frame.index]
    for indices in groups:
        indices = list(indices)
        frame.loc[indices, output_column] = multipletests(frame.loc[indices, p_column].astype(float), method="fdr_bh")[1]
    return frame


def participant_task_statistics(frame: pd.DataFrame, value: str, dimensions: list[str], bootstrap_repetitions: int, seeds: dict[tuple, int], fdr_groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(dimensions, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(dimensions, keys)), **paired_statistics(paired_values(subset, value), bootstrap_repetitions, seeds[keys])})
    return add_fdr(pd.DataFrame(rows), "paired_wilcoxon_two_sided_p", fdr_groups, "paired_wilcoxon_fdr_bh")


def channel_statistics(frame: pd.DataFrame, value: str, dimensions: list[str], family_dimensions: list[str]) -> pd.DataFrame:
    parts = []
    for keys, subset in frame.groupby(dimensions, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(dimensions, keys))
        labels = subset[["channel_index", "channel"]].drop_duplicates().sort_values("channel_index")
        wide = subset.pivot(index="subject", columns=["session", "channel_index"], values=value)
        available = sorted(set(wide["ses-01"].columns) & set(wide["ses-02"].columns))
        wide = wide.dropna(subset=[(session, channel) for session in SESSIONS for channel in available])
        baseline = wide["ses-01"][available].to_numpy(float)
        psilocybin = wide["ses-02"][available].to_numpy(float)
        delta = psilocybin - baseline
        participant_n = delta.shape[0]
        delta_mean = delta.mean(axis=0)
        delta_sd = delta.std(axis=0, ddof=1)
        paired_t = stats.ttest_rel(psilocybin, baseline, axis=0)
        wilcoxon = stats.wilcoxon(psilocybin, baseline, axis=0, alternative="two-sided", zero_method="wilcox")
        part = pd.DataFrame(
            {
                **group_values,
                "channel_index": available,
                "participant_n": participant_n,
                "baseline_mean": baseline.mean(axis=0),
                "psilocybin_mean": psilocybin.mean(axis=0),
                "delta_psilocybin_minus_baseline": delta_mean,
                "paired_cohen_dz": np.divide(delta_mean, delta_sd, out=np.full_like(delta_mean, np.nan), where=delta_sd != 0),
                "paired_t_two_sided_p": paired_t.pvalue,
                "paired_wilcoxon_two_sided_p": wilcoxon.pvalue,
            }
        ).merge(labels, on="channel_index", how="left")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result = add_fdr(result, "paired_wilcoxon_two_sided_p", dimensions, "paired_wilcoxon_fdr_bh_within_map")
    return add_fdr(result, "paired_wilcoxon_two_sided_p", family_dimensions, "paired_wilcoxon_fdr_bh_global")


def alpha_gap_statistics(alpha_run: pd.DataFrame, bootstrap_repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_parts = []
    statistic_rows = []
    offset = 0
    required = [(session, task) for session in SESSIONS for task in TASKS]
    for scale in (LINEAR_ALPHA, DECIBEL_ALPHA):
        subset = alpha_run[alpha_run.scale == scale]
        units = subset.units.iloc[0]
        wide = subset.pivot(index="subject", columns=["session", "task"], values="alpha_mean_channels")
        for task in sorted(TASKS[:3]):
            contrast_wide = wide.dropna(subset=[(session, current) for session in SESSIONS for current in (task, "movie")])
            gap = pd.DataFrame(
                {
                    "baseline": contrast_wide[("ses-01", task)] - contrast_wide[("ses-01", "movie")],
                    "psilocybin": contrast_wide[("ses-02", task)] - contrast_wide[("ses-02", "movie")],
                }
            )
            contrast = f"{task}_minus_movie"
            for session, column in SESSION_LABELS.items():
                value_parts.append(pd.DataFrame({"subject": gap.index, "session": session, "scale": scale, "units": units, "contrast": contrast, "gap_value": gap[column].to_numpy()}))
            statistic_rows.append({"scale": scale, "units": units, "contrast": contrast, **paired_statistics(gap, bootstrap_repetitions, seed + offset)})
            offset += 1
        complete = wide.dropna(subset=required)
        overall = pd.DataFrame(
            {
                "baseline": complete["ses-01"][list(TASKS[:3])].mean(axis=1) - complete[("ses-01", "movie")],
                "psilocybin": complete["ses-02"][list(TASKS[:3])].mean(axis=1) - complete[("ses-02", "movie")],
            }
        )
        contrast = "mean_three_closed_minus_movie"
        for session, column in SESSION_LABELS.items():
            value_parts.append(pd.DataFrame({"subject": overall.index, "session": session, "scale": scale, "units": units, "contrast": contrast, "gap_value": overall[column].to_numpy()}))
        statistic_rows.append({"scale": scale, "units": units, "contrast": contrast, **paired_statistics(overall, bootstrap_repetitions, seed + offset)})
        offset += 1
    statistics = add_fdr(pd.DataFrame(statistic_rows), "paired_wilcoxon_two_sided_p", ["scale"], "paired_wilcoxon_fdr_bh")
    return pd.concat(value_parts, ignore_index=True), statistics


def interaction_statistics(alpha_run: pd.DataFrame, lz_run: pd.DataFrame, bootstrap_repetitions: int, seeds: dict[str, int]) -> pd.DataFrame:
    rows = []
    specifications = [(f"alpha_{scale}", alpha_run[alpha_run.scale == scale], "alpha_mean_channels") for scale in (LINEAR_ALPHA, DECIBEL_ALPHA)]
    specifications.append(("lz76_corrected_sequence_length", lz_run, "lz76_mean_channels"))
    required = [(session, task) for session in SESSIONS for task in TASKS]
    for metric, frame, value in specifications:
        wide = frame.pivot(index="subject", columns=["session", "task"], values=value).dropna(subset=required)
        movie_delta = wide[("ses-02", "movie")] - wide[("ses-01", "movie")]
        closed_delta = wide["ses-02"][list(TASKS[:3])].mean(axis=1) - wide["ses-01"][list(TASKS[:3])].mean(axis=1)
        paired = pd.DataFrame({"baseline": movie_delta, "psilocybin": closed_delta})
        result = paired_statistics(paired, bootstrap_repetitions, seeds[metric])
        rows.append(
            {
                "metric": metric,
                "participant_n": result["participant_n"],
                "movie_session_delta_mean": result["baseline_mean"],
                "closed_mean_session_delta_mean": result["psilocybin_mean"],
                "interaction_closed_minus_movie": result["delta_psilocybin_minus_baseline"],
                "interaction_bootstrap_ci95_low": result["bootstrap_ci95_low"],
                "interaction_bootstrap_ci95_high": result["bootstrap_ci95_high"],
                "interaction_paired_cohen_dz": result["paired_cohen_dz"],
                "interaction_paired_t_two_sided_p": result["paired_t_two_sided_p"],
                "interaction_paired_wilcoxon_two_sided_p": result["paired_wilcoxon_two_sided_p"],
            }
        )
    return add_fdr(pd.DataFrame(rows), "interaction_paired_wilcoxon_two_sided_p", [], "interaction_paired_wilcoxon_fdr_bh")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    arguments = parser.parse_args()
    paths = sorted(arguments.feature_root.glob("sub-*/ses-*/task-*_eeg-features.npz"))
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        records = list(executor.map(load_feature, map(str, paths), chunksize=8))
    channel_labels = harmonize_channel_order(records)
    alpha_channel, alpha_run, spectrum_run, lz_channel, lz_run = build_tables(records, channel_labels)
    ordered_tasks = sorted(TASKS)
    alpha_seeds = {
        (scale, task): arguments.seed + offset
        for scale, width, start in ((LINEAR_ALPHA, 5, 0), (DECIBEL_ALPHA, 5, 0))
        for offset, task in ((start + width * index, task) for index, task in enumerate(ordered_tasks))
    }
    lz_seeds = {(task,): arguments.seed + 10004 + index for index, task in enumerate(ordered_tasks)}
    alpha_stats = participant_task_statistics(alpha_run, "alpha_mean_channels", ["scale", "task"], arguments.bootstrap_repetitions, alpha_seeds, ["scale"])
    lz_stats = participant_task_statistics(lz_run, "lz76_mean_channels", ["task"], arguments.bootstrap_repetitions, lz_seeds, [])
    alpha_channel_stats = channel_statistics(alpha_channel, "alpha", ["scale", "task"], ["scale"])
    lz_channel_stats = channel_statistics(lz_channel, "lz76", ["task"], [])
    alpha_gap_values, alpha_gap_stats = alpha_gap_statistics(alpha_run, arguments.bootstrap_repetitions, arguments.seed + 20000)
    interaction_seeds = {
        f"alpha_{LINEAR_ALPHA}": arguments.seed + 30000,
        f"alpha_{DECIBEL_ALPHA}": arguments.seed + 30000,
        "lz76_corrected_sequence_length": arguments.seed + 40001,
    }
    interaction_stats = interaction_statistics(alpha_run, lz_run, arguments.bootstrap_repetitions, interaction_seeds)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    spectrum_run.to_parquet(arguments.output_dir / "eeg_spectrum_run_metrics.parquet", index=False)
    alpha_channel.to_parquet(arguments.output_dir / "eeg_alpha_channel_metrics.parquet", index=False)
    alpha_run.to_parquet(arguments.output_dir / "eeg_alpha_run_metrics.parquet", index=False)
    alpha_stats.to_csv(arguments.output_dir / "eeg_alpha_paired_stats.csv", index=False)
    alpha_channel_stats.to_csv(arguments.output_dir / "eeg_alpha_channel_paired_stats.csv", index=False)
    alpha_gap_values.to_parquet(arguments.output_dir / "eeg_alpha_participant_gap_values.parquet", index=False)
    alpha_gap_stats.to_csv(arguments.output_dir / "eeg_alpha_participant_gap_stats.csv", index=False)
    lz_channel.to_parquet(arguments.output_dir / "eeg_lz_channel_metrics.parquet", index=False)
    lz_run.to_parquet(arguments.output_dir / "eeg_lz_run_metrics.parquet", index=False)
    lz_stats.to_csv(arguments.output_dir / "eeg_lz_paired_stats.csv", index=False)
    lz_channel_stats.to_csv(arguments.output_dir / "eeg_lz_channel_paired_stats.csv", index=False)
    interaction_stats.to_csv(arguments.output_dir / "eeg_task_interaction_stats.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
