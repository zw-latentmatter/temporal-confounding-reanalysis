from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


NETWORKS = (
    "subcortical",
    "visual",
    "somatomotor",
    "dorsal_attention",
    "limbic",
    "salience_ventral_attention",
    "default_mode",
    "control",
    "default_mode_plus_visual",
)
SINGLE_NETWORKS = NETWORKS[:-1]
FIXED = "fixed_post_model"
REFITTED = "retrained_hybrid_model"
DROP_METRIC = "balanced_accuracy_drop_signed"
OOD_METRIC = "embedding_nearest_post_distance_over_post_self_nn"


def participant_contrasts(
    participant: pd.DataFrame,
    fit: pd.DataFrame,
    fixed_null: pd.DataFrame,
) -> pd.DataFrame:
    primary = participant[
        participant["row_kind"].eq("primary")
        & participant["target"].isin(NETWORKS)
        & participant["model_mode"].isin((FIXED, REFITTED))
    ]
    paired = primary.pivot(
        index=["subject", "target"],
        columns="model_mode",
        values=DROP_METRIC,
    ).reset_index()
    adaptation = paired[["subject", "target"]].copy()
    adaptation["statistic"] = "fixed_minus_refitted_drop"
    adaptation["model_mode"] = "fixed_minus_refitted"
    adaptation["value"] = paired[FIXED] - paired[REFITTED]
    ood = fit[
        fit["row_kind"].eq("primary")
        & fit["model_mode"].eq(FIXED)
        & fit["target"].isin(NETWORKS)
    ].groupby(["subject", "target"], as_index=False)[OOD_METRIC].median()
    ood["statistic"] = "hybrid_to_post_nearest_neighbour_ratio"
    ood["model_mode"] = FIXED
    ood["value"] = ood[OOD_METRIC]
    ood = ood[["subject", "target", "statistic", "model_mode", "value"]]
    equal_size = fixed_null[
        fixed_null["preprocessing"].eq("none")
        & fixed_null["metric"].eq(DROP_METRIC)
        & fixed_null["target"].isin(NETWORKS)
    ][["subject", "target", "observed_minus_random_mean"]].copy()
    equal_size["statistic"] = "named_minus_size_matched_random_drop"
    equal_size["model_mode"] = FIXED
    equal_size["value"] = equal_size["observed_minus_random_mean"]
    equal_size = equal_size[
        ["subject", "target", "statistic", "model_mode", "value"]
    ]
    interaction = participant[
        participant["row_kind"].eq("interaction")
        & participant["target"].eq("default_mode_plus_visual")
        & participant["model_mode"].isin((FIXED, REFITTED))
    ][
        [
            "subject",
            "target",
            "model_mode",
            "balanced_accuracy_interaction_signed",
        ]
    ].copy()
    interaction["statistic"] = "default_mode_visual_nonadditivity"
    interaction["value"] = interaction["balanced_accuracy_interaction_signed"]
    interaction = interaction[
        ["subject", "target", "statistic", "model_mode", "value"]
    ]
    result = pd.concat(
        [adaptation, ood, equal_size, interaction],
        ignore_index=True,
    )
    result["target"] = pd.Categorical(result["target"], NETWORKS, ordered=True)
    return result.sort_values(["statistic", "target", "model_mode", "subject"])


def group_statistics(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in contrasts.groupby(
        ["statistic", "target", "model_mode"],
        sort=False,
        observed=True,
    ):
        values = group["value"].astype(float)
        mean = float(values.mean())
        low, high = stats.t.interval(
            0.95,
            len(values) - 1,
            loc=mean,
            scale=stats.sem(values),
        )
        rows.append(
            {
                "statistic": keys[0],
                "target": keys[1],
                "model_mode": keys[2],
                "n_participants": int(len(values)),
                "mean": mean,
                "sd": float(values.std(ddof=1)),
                "median": float(values.median()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def mask_size_statistics(
    participant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = participant[
        participant["row_kind"].eq("primary")
        & participant["target"].isin(SINGLE_NETWORKS)
        & participant["model_mode"].isin((FIXED, REFITTED))
    ]
    summaries = []
    associations = []
    for mode in (FIXED, REFITTED):
        values = (
            primary[primary["model_mode"].eq(mode)]
            .groupby("target", as_index=False, observed=True)
            .agg(mask_size=("mask_size", "first"), mean_drop=(DROP_METRIC, "mean"))
        )
        values["model_mode"] = mode
        summaries.append(values[["target", "model_mode", "mask_size", "mean_drop"]])
        result = stats.spearmanr(values["mask_size"], values["mean_drop"])
        associations.append(
            {
                "model_mode": mode,
                "n_masks": int(len(values)),
                "spearman_rho": float(result.statistic),
                "p_two_sided": float(result.pvalue),
            }
        )
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(associations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant-metrics", required=True, type=Path)
    parser.add_argument("--fit-metrics", required=True, type=Path)
    parser.add_argument("--fixed-null", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    participant = pd.read_csv(args.participant_metrics)
    fit = pd.read_csv(args.fit_metrics)
    fixed_null = pd.read_csv(args.fixed_null)
    contrasts = participant_contrasts(participant, fit, fixed_null)
    groups = group_statistics(contrasts)
    mask_sizes, mask_associations = mask_size_statistics(participant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(
        args.output_dir / "network_replacement_participant_contrasts.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    groups.to_csv(
        args.output_dir / "network_replacement_group_statistics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    mask_sizes.to_csv(
        args.output_dir / "network_replacement_mask_size_statistics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    mask_associations.to_csv(
        args.output_dir / "network_replacement_mask_size_spearman.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
