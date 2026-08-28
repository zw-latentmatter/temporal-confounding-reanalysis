from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .io import atomic_csv, atomic_json, read_json
from .metrics import network_embedding_metric_rows, roi_network_indices


PAIR_KEYS = [
    "phase",
    "subject",
    "pipeline",
    "variant_id",
    "feature_mode",
    "topology_score",
    "density",
    "loss_mode",
    "seed",
    "task_order",
    "scope",
    "condition_a",
    "condition_b",
    "metric",
]


def collect_metrics(
    output_root: Path, plan: pd.DataFrame, roi_labels_path: Path
) -> pd.DataFrame:
    labels = json.loads(roi_labels_path.read_text(encoding="utf-8"))
    network_indices = roi_network_indices([str(label) for label in labels])
    identity_columns = [
        "phase",
        "subject",
        "session",
        "pipeline",
        "variant_id",
        "feature_mode",
        "topology_score",
        "density",
        "loss_mode",
        "seed",
        "task_order",
    ]
    rows = []
    for record in plan.to_dict("records"):
        run_dir = output_root / "runs" / str(record["run_id"])
        state = read_json(run_dir / "state.json")
        if state.get("status") != "completed":
            raise RuntimeError(f"run is incomplete: {record['run_id']}")
        with np.load(run_dir / "embeddings.npz", allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
            task_order = tuple(str(value) for value in archive["task_order"].tolist())
        identity = {column: record[column] for column in identity_columns}
        rows.extend(
            {**identity, **metric}
            for metric in network_embedding_metric_rows(
                embeddings, task_order, network_indices
            )
        )
    metrics = pd.DataFrame(rows)
    atomic_csv(output_root / "all_run_metrics.csv", metrics)
    return metrics


def append_four_condition_composites(seed_level: pd.DataFrame) -> pd.DataFrame:
    conditions = {"rest", "meditation", "music", "movie"}
    source = seed_level[seed_level["condition_a"].isin(conditions)].copy()
    group_columns = [column for column in PAIR_KEYS if column != "condition_a"]
    composites = (
        source.groupby(group_columns, dropna=False)
        .agg(
            condition_n=("condition_a", "nunique"),
            baseline_value=("baseline_value", "mean"),
            drug_value=("drug_value", "mean"),
            delta=("delta", "mean"),
        )
        .reset_index()
    )
    composites = composites[composites["condition_n"] == 4].drop(
        columns="condition_n"
    )
    composites["condition_a"] = "all_four_condition_mean"
    composites = composites[seed_level.columns]
    return pd.concat([seed_level, composites], ignore_index=True)


def participant_pairs(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = metrics.pivot_table(
        index=PAIR_KEYS, columns="session", values="value", aggfunc="first"
    ).reset_index()
    seed_level = pivot.dropna(subset=["ses-01", "ses-02"]).rename(
        columns={"ses-01": "baseline_value", "ses-02": "drug_value"}
    )
    seed_level["delta"] = seed_level["drug_value"] - seed_level["baseline_value"]
    seed_level = append_four_condition_composites(seed_level)
    aggregate_keys = [column for column in PAIR_KEYS if column != "seed"]
    participant = (
        seed_level.groupby(aggregate_keys, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            baseline_value=("baseline_value", "mean"),
            drug_value=("drug_value", "mean"),
            delta=("delta", "mean"),
        )
        .reset_index()
    )
    return seed_level, participant


def _sign_flip_p(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(values.mean())
    if len(values) <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
        null = np.abs((signs * values).mean(axis=1))
        return float(np.mean(null >= observed - 1e-15))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (draws + 1))


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def group_summaries(
    pairs: pd.DataFrame,
    signflip_draws: int = 100000,
    bootstrap_draws: int = 10000,
    seed: int = 260826,
) -> pd.DataFrame:
    group_columns = [
        "phase",
        "pipeline",
        "variant_id",
        "feature_mode",
        "topology_score",
        "density",
        "loss_mode",
        "task_order",
        "scope",
        "condition_a",
        "condition_b",
        "metric",
    ]
    rows = []
    for group_index, (keys, local) in enumerate(
        pairs.groupby(group_columns, dropna=False, sort=True)
    ):
        rng = np.random.default_rng(seed + group_index)
        delta = local["delta"].to_numpy(dtype=float)
        low, high = _bootstrap_ci(delta, rng, bootstrap_draws)
        standard_deviation = float(delta.std(ddof=1)) if len(delta) > 1 else np.nan
        values = dict(zip(group_columns, keys))
        rows.append(
            {
                **values,
                "n": len(local),
                "mean_baseline": float(local["baseline_value"].mean()),
                "mean_drug": float(local["drug_value"].mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "sd_delta": standard_deviation,
                "ci95_low": low,
                "ci95_high": high,
                "effect_dz": (
                    float(delta.mean() / standard_deviation)
                    if standard_deviation and np.isfinite(standard_deviation)
                    else np.nan
                ),
                "p_signflip": _sign_flip_p(delta, rng, signflip_draws),
                "inferential_family": (
                    "main"
                    if values["phase"] == "primary"
                    and values["variant_id"] == "signed-d10"
                    else "model_sensitivity"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["q_bh_within_declared_family"] = np.nan
    for _, indices in summary.groupby("inferential_family").groups.items():
        summary.loc[indices, "q_bh_within_declared_family"] = _bh(
            summary.loc[indices, "p_signflip"]
        )
    return summary


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _first_order_partial_correlation(
    x: np.ndarray, y: np.ndarray, control: np.ndarray
) -> tuple[float, float]:
    correlation = float(
        stats.pearsonr(_residualize(x, control), _residualize(y, control)).statistic
    )
    degrees_of_freedom = len(x) - 3
    statistic = correlation * np.sqrt(
        degrees_of_freedom / max(1.0 - correlation**2, np.finfo(float).eps)
    )
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom))
    return correlation, p_value


def phenotype_associations(
    pairs: pd.DataFrame,
    phenotype_csv: Path,
    phenotype_subject_column: str,
    phenotype_score_columns: tuple[str, ...],
) -> pd.DataFrame:
    selected = pairs[
        (pairs["phase"] == "primary")
        & (pairs["variant_id"] == "signed-d10")
        & (pairs["condition_a"] == "all_four_condition_mean")
    ].copy()
    phenotype = pd.read_csv(phenotype_csv).rename(
        columns={phenotype_subject_column: "subject"}
    )
    phenotype["subject"] = phenotype["subject"].astype(str)
    score_columns = [
        column for column in phenotype_score_columns if column in phenotype.columns
    ]
    meq_column = "MEQ30_MEAN" if "MEQ30_MEAN" in score_columns else None
    rows = []
    endpoint_keys = ["scope", "condition_b", "metric"]
    for endpoint, local in selected.groupby(endpoint_keys, dropna=False, sort=True):
        joined = local.merge(
            phenotype[["subject", *score_columns]], on="subject", how="inner"
        )
        for value_kind in ("drug_value", "delta"):
            for outcome in score_columns:
                columns = [value_kind, outcome]
                if meq_column is not None and outcome != meq_column:
                    columns.append(meq_column)
                complete = joined[columns].dropna()
                x = complete[value_kind].to_numpy(dtype=float)
                y = complete[outcome].to_numpy(dtype=float)
                for method in ("pearson", "spearman"):
                    result = (
                        stats.pearsonr(x, y)
                        if method == "pearson"
                        else stats.spearmanr(x, y)
                    )
                    rows.append(
                        {
                            "scope": endpoint[0],
                            "condition_a": "all_four_condition_mean",
                            "condition_b": endpoint[1],
                            "metric": endpoint[2],
                            "value_kind": value_kind,
                            "outcome": outcome,
                            "method": method,
                            "partial_control": "none",
                            "n": len(complete),
                            "correlation": float(result.statistic),
                            "p_value": float(result.pvalue),
                        }
                    )
                if meq_column is not None and outcome != meq_column:
                    control = complete[meq_column].to_numpy(dtype=float)
                    for method in ("pearson", "spearman"):
                        if method == "spearman":
                            local_x = stats.rankdata(x)
                            local_y = stats.rankdata(y)
                            local_control = stats.rankdata(control)
                        else:
                            local_x, local_y, local_control = x, y, control
                        correlation, p_value = _first_order_partial_correlation(
                            local_x, local_y, local_control
                        )
                        rows.append(
                            {
                                "scope": endpoint[0],
                                "condition_a": "all_four_condition_mean",
                                "condition_b": endpoint[1],
                                "metric": endpoint[2],
                                "value_kind": value_kind,
                                "outcome": outcome,
                                "method": f"partial_{method}",
                                "partial_control": meq_column,
                                "n": len(complete),
                                "correlation": correlation,
                                "p_value": p_value,
                            }
                        )
    associations = pd.DataFrame(rows)
    associations["q_bh_all_tavrnn_behavior_tests"] = _bh(
        associations["p_value"]
    )
    return associations


def finalize_analysis(
    output_root: Path,
    roi_labels_path: Path,
    phenotype_csv: Path | None = None,
    phenotype_subject_column: str = "subject",
    phenotype_score_columns: tuple[str, ...] = (
        "MEQ30_MEAN",
        "MINDSET_AVAILABLE_MEAN",
    ),
    signflip_draws: int = 100000,
    bootstrap_draws: int = 10000,
) -> dict:
    plan = pd.read_csv(output_root / "plan.csv")
    metrics = collect_metrics(output_root, plan, roi_labels_path)
    seed_level, pairs = participant_pairs(metrics)
    atomic_csv(output_root / "participant_pairs_seed_level.csv", seed_level)
    atomic_csv(output_root / "participant_pairs.csv", pairs)
    groups = group_summaries(pairs, signflip_draws, bootstrap_draws)
    atomic_csv(output_root / "group_summary.csv", groups)
    associations = pd.DataFrame()
    if phenotype_csv is not None:
        associations = phenotype_associations(
            pairs,
            phenotype_csv,
            phenotype_subject_column,
            phenotype_score_columns,
        )
        atomic_csv(output_root / "phenotype_associations.csv", associations)
    result = {
        "schema_version": "1.0",
        "completed_runs": len(plan),
        "participant_pair_rows": len(pairs),
        "group_summary_rows": len(groups),
        "phenotype_association_rows": len(associations),
    }
    atomic_json(output_root / "analysis_manifest.json", result)
    return result
