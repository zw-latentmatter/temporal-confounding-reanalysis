from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANONICAL_TASKS


def load_fc_sequence(
    input_paths_json: str, fc_key: str, expected_nodes: int = 332
) -> np.ndarray:
    paths = json.loads(input_paths_json)
    matrices = []
    for path in paths:
        with np.load(Path(path)) as archive:
            if fc_key not in archive:
                raise KeyError(f"{path} does not contain {fc_key}")
            matrices.append(np.asarray(archive[fc_key], dtype=np.float32))
    sequence = np.stack(matrices)
    if sequence.shape != (4, expected_nodes, expected_nodes):
        raise ValueError(
            f"expected FC shape (4,{expected_nodes},{expected_nodes}), got {sequence.shape}"
        )
    return sequence


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def complete_paired_groups(
    manifest_path: Path,
    cohort_membership_path: Path | None = None,
    cohort_column: str | None = None,
) -> list[dict]:
    frame = pd.read_csv(manifest_path)
    required = {"subject", "session", "task", "roi_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input manifest is missing columns: {sorted(missing)}")
    frame = frame[
        frame["session"].isin(("ses-01", "ses-02"))
        & frame["task"].isin(CANONICAL_TASKS)
    ].copy()
    if cohort_membership_path is not None:
        membership = pd.read_csv(cohort_membership_path)
        if cohort_column is None or cohort_column not in membership.columns:
            raise ValueError("cohort column must exist in membership CSV")
        retained = set(membership.loc[_truthy(membership[cohort_column]), "subject"])
        frame = frame[frame["subject"].isin(retained)]
    groups = []
    for (subject, session), local in frame.groupby(["subject", "session"], sort=True):
        if set(local["task"]) != set(CANONICAL_TASKS) or len(local) != 4:
            continue
        indexed = local.set_index("task")
        groups.append(
            {
                "subject": str(subject),
                "session": str(session),
                "input_paths_json": json.dumps(
                    [str(indexed.loc[task, "roi_path"]) for task in CANONICAL_TASKS]
                ),
            }
        )
    group_frame = pd.DataFrame(groups)
    if group_frame.empty:
        raise RuntimeError("no complete four-condition groups found")
    paired_subjects = {
        subject
        for subject, local in group_frame.groupby("subject")
        if set(local["session"]) == {"ses-01", "ses-02"}
    }
    return [row for row in groups if row["subject"] in paired_subjects]
