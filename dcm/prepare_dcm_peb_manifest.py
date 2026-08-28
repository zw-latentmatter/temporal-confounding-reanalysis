from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scipy.io import whosmat


ENTITY_PATTERNS = {
    "subject": re.compile(r"(?<![A-Za-z0-9])(sub-[A-Za-z0-9]+)(?![A-Za-z0-9])"),
    "session": re.compile(r"(?<![A-Za-z0-9])(ses-[A-Za-z0-9]+)(?![A-Za-z0-9])"),
    "task": re.compile(r"(?<![A-Za-z0-9])task-([A-Za-z0-9]+)(?![A-Za-z0-9])"),
    "branch": re.compile(r"(?<![A-Za-z0-9])branch-([A-Za-z0-9_.-]+)"),
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    branch: str
    subject: str
    session: str
    task: str


def parse_entities(
    path: Path, root: Path, branch_regex: re.Pattern[str] | None
) -> dict[str, str]:
    text = path.relative_to(root).as_posix()
    found: dict[str, str] = {}
    for key in ("subject", "session", "task"):
        matches = [match.group(1) for match in ENTITY_PATTERNS[key].finditer(text)]
        if len(set(matches)) != 1:
            raise ValueError(f"Expected one {key} entity in {text}")
        found[key] = matches[0]
    if branch_regex is not None:
        match = branch_regex.search(text)
        if match is None:
            raise ValueError(f"Branch expression did not match {text}")
        if "branch" in match.groupdict():
            branch = match.group("branch")
        elif match.lastindex:
            branch = match.group(1)
        else:
            branch = match.group(0)
    else:
        match = ENTITY_PATTERNS["branch"].search(text)
        if match is not None:
            branch = match.group(1)
        else:
            parts = path.relative_to(root).parts
            subject_index = parts.index(found["subject"])
            branch = parts[subject_index - 1] if subject_index else "default"
    found["branch"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", branch).strip("._-") or "default"
    return found


def contains_dcm(path: Path) -> bool:
    try:
        return "DCM" in {name for name, _, _ in whosmat(path)}
    except NotImplementedError:
        return True


def discover_candidates(
    root: Path,
    filename_regex: re.Pattern[str],
    branch_regex: re.Pattern[str] | None,
) -> list[Candidate]:
    result: list[Candidate] = []
    for path in sorted(root.rglob("*.mat")):
        if not path.is_file() or not filename_regex.search(path.name) or not contains_dcm(path):
            continue
        try:
            entities = parse_entities(path, root, branch_regex)
        except ValueError:
            continue
        result.append(Candidate(path=path.relative_to(root), **entities))
    return result


def build_pairs(
    candidates: Iterable[Candidate], baseline_session: str, drug_session: str
) -> list[dict[str, str | int]]:
    grouped: dict[tuple[str, str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.branch, candidate.task, candidate.subject, candidate.session)].append(
            candidate
        )
    subject_cells = sorted(
        {(item.branch, item.task, item.subject) for item in candidates}
    )
    pairs: list[dict[str, str | int]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for branch, task, subject in subject_cells:
        baseline = grouped[(branch, task, subject, baseline_session)]
        drug = grouped[(branch, task, subject, drug_session)]
        if len(baseline) != 1 or len(drug) != 1 or baseline[0].path == drug[0].path:
            continue
        pair_counts[(branch, task)] += 1
        pairs.append(
            {
                "branch": branch,
                "task": task,
                "subject": subject,
                "pair_index": pair_counts[(branch, task)],
                "baseline_session": baseline_session,
                "drug_session": drug_session,
                "baseline_path": baseline[0].path.as_posix(),
                "drug_path": drug[0].path.as_posix(),
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    fields = [
        "branch",
        "task",
        "subject",
        "pair_index",
        "baseline_session",
        "drug_session",
        "baseline_path",
        "drug_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcm-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filename-regex", default=r"DCM.*\.mat$|.*_DCM\.mat$")
    parser.add_argument("--branch-regex")
    parser.add_argument("--baseline-session", default="ses-01")
    parser.add_argument("--drug-session", default="ses-02")
    args = parser.parse_args()
    if args.baseline_session == args.drug_session:
        parser.error("Session labels must differ")
    root = args.dcm_root.resolve()
    filename_regex = re.compile(args.filename_regex, re.IGNORECASE)
    branch_regex = re.compile(args.branch_regex) if args.branch_regex else None
    candidates = discover_candidates(root, filename_regex, branch_regex)
    pairs = build_pairs(candidates, args.baseline_session, args.drug_session)
    write_csv(args.output.resolve(), pairs)
    counts = Counter((str(row["branch"]), str(row["task"])) for row in pairs)
    summary = {
        "candidate_count": len(candidates),
        "paired_count": len(pairs),
        "pair_counts": {
            f"{branch}/{task}": count
            for (branch, task), count in sorted(counts.items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
