from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tavrnn_reanalysis.data import complete_paired_groups
from tavrnn_reanalysis.plan import DEFAULT_SEEDS, build_analysis_plan
from tavrnn_reanalysis.summary import finalize_analysis
from tavrnn_reanalysis.training import execute_run


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def command_plan(args: argparse.Namespace) -> int:
    _, metadata = build_analysis_plan(
        input_manifest=args.input_manifest,
        output_root=args.output_root,
        cohort_membership=args.cohort_membership,
        cohort_column=args.cohort_column,
        fc_key=args.fc_key,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    plan = pd.read_csv(args.plan_csv)
    sources = {
        (record["subject"], record["session"]): record["input_paths_json"]
        for record in complete_paired_groups(args.input_manifest)
    }
    if args.phases:
        plan = plan[plan["phase"].isin(args.phases)]
    selected = plan.iloc[
        [index for index in range(len(plan)) if index % args.shard_count == args.shard_index]
    ]
    counts = {}
    for record in selected.to_dict("records"):
        record["input_paths_json"] = sources[(record["subject"], record["session"])]
        status = execute_run(
            record,
            args.output_root,
            resume=not args.no_resume,
            checkpoint_interval=args.checkpoint_interval,
            device_override=args.device,
        )
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"selected_runs": len(selected), "status_counts": counts}, indent=2))
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    result = finalize_analysis(
        output_root=args.output_root,
        roi_labels_path=args.roi_labels,
        phenotype_csv=args.phenotype_csv,
        phenotype_subject_column=args.phenotype_subject_column,
        phenotype_score_columns=tuple(args.phenotype_score_columns),
        signflip_draws=args.signflip_draws,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(result, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="run_tavrnn_reanalysis.py")
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--input-manifest", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--cohort-membership", type=Path)
    plan.add_argument("--cohort-column")
    plan.add_argument("--fc-key", default="author_fc")
    plan.add_argument("--seeds", type=_seeds, default=DEFAULT_SEEDS)
    plan.add_argument("--epochs", type=int, default=150)
    plan.set_defaults(function=command_plan)
    run = commands.add_parser("run")
    run.add_argument("--plan-csv", required=True, type=Path)
    run.add_argument("--input-manifest", required=True, type=Path)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument(
        "--phases",
        nargs="*",
        choices=("primary", "feature_density_sensitivity"),
    )
    run.add_argument("--checkpoint-interval", type=int, default=25)
    run.add_argument("--device")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(function=command_run)
    final = commands.add_parser("finalize")
    final.add_argument("--output-root", required=True, type=Path)
    final.add_argument("--roi-labels", required=True, type=Path)
    final.add_argument("--phenotype-csv", type=Path)
    final.add_argument("--phenotype-subject-column", default="subject")
    final.add_argument(
        "--phenotype-score-columns",
        nargs="*",
        default=("MEQ30_MEAN", "MINDSET_AVAILABLE_MEAN"),
    )
    final.add_argument("--signflip-draws", type=int, default=100000)
    final.add_argument("--bootstrap-draws", type=int, default=10000)
    final.set_defaults(function=command_finalize)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
