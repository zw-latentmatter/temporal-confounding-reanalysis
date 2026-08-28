from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CANONICAL_TASKS, CORE_VARIANTS
from .data import complete_paired_groups
from .io import atomic_csv, atomic_json


DEFAULT_SEEDS = (1107, 2303, 7919)


def _run_id(
    phase: str, variant: str, subject: str, session: str, seed: int
) -> str:
    return (
        f"{phase}__corrected_modulelist__{variant}__{subject}__{session}__"
        f"seed-{seed:05d}"
    )


def build_analysis_plan(
    input_manifest: Path,
    output_root: Path,
    cohort_membership: Path | None = None,
    cohort_column: str | None = None,
    fc_key: str = "author_fc",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    epochs: int = 150,
) -> tuple[pd.DataFrame, dict]:
    groups = complete_paired_groups(
        input_manifest, cohort_membership, cohort_column
    )
    rows = []
    for source in groups:
        for seed in seeds:
            variant = CORE_VARIANTS[0]
            rows.append(
                {
                    "run_id": _run_id(
                        "primary", variant.variant_id, source["subject"], source["session"], seed
                    ),
                    "phase": "primary",
                    "subject": source["subject"],
                    "session": source["session"],
                    "pipeline": "corrected_modulelist",
                    **variant.as_dict(),
                    "fc_key": fc_key,
                    "expected_nodes": 332,
                    "loss_mode": "notebook_full_matrix",
                    "seed": seed,
                    "task_order": "|".join(CANONICAL_TASKS),
                    "epochs": epochs,
                    "requested_device": "cuda",
                }
            )
        for variant in CORE_VARIANTS[1:]:
            seed = seeds[0]
            rows.append(
                {
                    "run_id": _run_id(
                        "feature_density_sensitivity",
                        variant.variant_id,
                        source["subject"],
                        source["session"],
                        seed,
                    ),
                    "phase": "feature_density_sensitivity",
                    "subject": source["subject"],
                    "session": source["session"],
                    "pipeline": "corrected_modulelist",
                    **variant.as_dict(),
                    "fc_key": fc_key,
                    "expected_nodes": 332,
                    "loss_mode": "notebook_full_matrix",
                    "seed": seed,
                    "task_order": "|".join(CANONICAL_TASKS),
                    "epochs": epochs,
                    "requested_device": "cuda",
                }
            )
    plan = pd.DataFrame(rows)
    plan.insert(0, "plan_index", range(len(plan)))
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_root / "plan.csv", plan)
    metadata = {
        "schema_version": "1.0",
        "fc_key": fc_key,
        "canonical_tasks": list(CANONICAL_TASKS),
        "seeds": list(seeds),
        "subject_count": int(plan["subject"].nunique()),
        "run_count": len(plan),
        "epochs": epochs,
        "loss_mode": "notebook_full_matrix",
        "variants": [variant.as_dict() for variant in CORE_VARIANTS],
        "inference_unit": "participant",
    }
    atomic_json(output_root / "plan.json", metadata)
    return plan, metadata
