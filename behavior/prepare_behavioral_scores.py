from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MEQ_SCALES = {
    "MEQ30_MYSTICAL": [4, 5, 6, 9, 14, 15, 16, 18, 20, 21, 23, 24, 25, 26, 28],
    "MEQ30_POSITIVE": [2, 8, 12, 17, 27, 30],
    "MEQ30_TRANSCEND": [1, 7, 11, 13, 19, 22],
    "MEQ30_INEFFABILITY": [3, 10, 29],
    "MEQ30_MEAN": list(range(1, 31)),
}
MINDSET_DIMENSIONS = (
    "PATIENCE",
    "CREATIVE",
    "MEANING",
    "HARMONY",
    "SELF",
    "OTHERS",
    "NATURE",
    "OPENNESS",
    "ACCEPT",
    "PEACE",
    "IMAGINE",
)


def normalize_subject(value: object) -> str:
    text = str(value)
    return text if text.startswith("sub-") else f"sub-{text}"


def score_meq30(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["participant_id"]].copy()
    for scale, item_numbers in MEQ_SCALES.items():
        columns = [f"MEQ30_{number}" for number in item_numbers]
        items = frame[columns].apply(pd.to_numeric, errors="coerce")
        admissible = items.notna().mean(axis=1) >= 0.5
        filled = items.T.fillna(items.median(axis=1)).T
        result[scale] = filled.mean(axis=1).mul(20).where(admissible)
    return result


def score_mindset(frame: pd.DataFrame) -> pd.DataFrame:
    raw_columns = [f"EXP_MINDSET_{name}" for name in MINDSET_DIMENSIONS]
    values = frame[raw_columns].apply(pd.to_numeric, errors="coerce")
    values.columns = [f"MINDSET_{name}" for name in MINDSET_DIMENSIONS]
    result = frame[["participant_id"]].join(values)
    result["MINDSET_AVAILABLE_MEAN"] = values.mean(axis=1, skipna=True)
    result["MINDSET_COMPLETE11_MEAN"] = values.mean(axis=1, skipna=False)
    result["MINDSET_AVAILABLE_DIMENSIONS"] = values.notna().sum(axis=1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phenotype = args.dataset / "derivatives" / "phenotype" / "rawdata"
    meq = pd.read_csv(phenotype / "ses-02_raw.tsv", sep="\t")
    mindset = pd.read_csv(phenotype / "followup-1day_raw.tsv", sep="\t")
    meq["participant_id"] = meq["participant_id"].map(normalize_subject)
    mindset["participant_id"] = mindset["participant_id"].map(normalize_subject)
    scores = score_meq30(meq).merge(score_mindset(mindset), on="participant_id", how="outer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        scores.to_parquet(args.output, index=False)
    else:
        scores.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
