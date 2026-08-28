from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
TASKS = ('rest', 'meditation', 'music', 'movie')
SESSIONS = ('ses-01', 'ses-02')
DEFAULT_MODAL_LENGTHS = {'rest': 499, 'meditation': 400, 'music': 723, 'movie': 367}
MOTION_COLUMNS = ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1', 'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1')

def parse_entities(path: Path) -> tuple[str, str, str]:
    text = '/' + path.as_posix()
    subject_match = re.search('/(sub-PC\\d+)/', text, flags=re.IGNORECASE)
    session_match = re.search('/(ses-\\d+)/', text)
    task_match = re.search('/task-([^/_]+)_', text)
    if not (subject_match and session_match and task_match):
        raise ValueError(f'cannot parse BIDS entities from {path}')
    return (subject_match.group(1), session_match.group(1), task_match.group(1))

def confounds_path(dataset_root: Path, subject: str, session: str, task: str) -> Path | None:
    base = dataset_root / 'derivatives' / 'fmriprep-22.0.2' / subject / session / 'func'
    paths = sorted(base.glob(f'{subject}_{session}_task-{task}_*desc-confounds_timeseries.tsv'))
    if not paths:
        paths = sorted(base.glob(f'*task-{task}_*desc-confounds_timeseries.tsv'))
    return paths[0] if paths else None

def read_confound_summary(path: Path, expected_frames: int) -> dict:
    frame = pd.read_csv(path, sep='\t')
    difference = len(frame) - expected_frames
    if difference < 0 or difference > 10:
        raise RuntimeError(f'cannot align {len(frame)} confound rows to {expected_frames} ROI frames: {path}')
    frame = frame.iloc[difference:].reset_index(drop=True)
    if 'framewise_displacement' in frame.columns:
        fd = pd.to_numeric(frame['framewise_displacement'], errors='coerce').to_numpy(float)
    else:
        fd = np.full(expected_frames, np.nan)
    finite = fd[np.isfinite(fd)]
    available_motion = [column for column in MOTION_COLUMNS if column in frame.columns]
    return {'confounds_source_frames': int(len(frame) + difference), 'confounds_rows_removed_for_alignment': int(difference), 'motion_column_count': len(available_motion), 'mean_fd': float(finite.mean()) if finite.size else math.nan, 'median_fd': float(np.median(finite)) if finite.size else math.nan, 'fraction_fd_gt_0_2': float(np.mean(finite > 0.2)) if finite.size else math.nan, 'fraction_fd_gt_0_5': float(np.mean(finite > 0.5)) if finite.size else math.nan}

def inspect_npz(path: Path, variants: tuple[str, ...]) -> dict:
    with np.load(path) as archive:
        missing = [variant for variant in variants if variant not in archive]
        if missing:
            raise RuntimeError(f'{path} is missing arrays: {missing}')
        shapes = {variant: tuple(np.asarray(archive[variant]).shape) for variant in variants}
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f'variant shapes disagree in {path}: {shapes}')
        shape = next(iter(shapes.values()))
        if len(shape) != 2 or shape[1] != 332:
            raise RuntimeError(f'unexpected ROI shape in {path}: {shape}')
        nonfinite = {variant: int(np.size(archive[variant]) - np.isfinite(archive[variant]).sum()) for variant in variants}
    return {'n_frames': int(shape[0]), 'n_features': int(shape[1]), 'nonfinite_total': int(sum(nonfinite.values())), 'nonfinite_by_variant': json.dumps(nonfinite, sort_keys=True), 'file_bytes': int(path.stat().st_size)}

def parse_modal_lengths(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for token in value.split(','):
        task, length = token.split('=', 1)
        if task not in TASKS:
            raise ValueError(f'unknown task in modal lengths: {task}')
        result[task] = int(length)
    if set(result) != set(TASKS):
        raise ValueError(f'modal lengths must specify {TASKS}, got {result}')
    return result

def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)

def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    csv_path = stem.with_suffix('.csv')
    parquet_path = stem.with_suffix('.parquet')
    temporary_csv = csv_path.with_name(f'.{csv_path.name}.partial')
    temporary_parquet = parquet_path.with_name(f'.{parquet_path.name}.partial')
    frame.to_csv(temporary_csv, index=False)
    frame.to_parquet(temporary_parquet, index=False)
    temporary_csv.replace(csv_path)
    temporary_parquet.replace(parquet_path)

def build_manifest(volume_root: Path, dataset_root: Path, output_dir: Path, variants: tuple[str, ...], modal_lengths: dict[str, int], proxy_size: int, author_seeds: int, blocked_seeds: int, blocked_folds: int, transfer_seeds: int, participant_seeds: int, participant_folds: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(volume_root.glob('sub-*/ses-*/task-*_volume-roi332.npz'))
    if not paths:
        raise RuntimeError(f'no ROI NPZ files found under {volume_root}')
    rows: list[dict] = []
    for path in paths:
        subject, session, task = parse_entities(path)
        if session not in SESSIONS or task not in TASKS:
            continue
        inspected = inspect_npz(path, variants)
        confound = confounds_path(dataset_root, subject, session, task)
        if confound is None:
            raise RuntimeError(f'missing confounds for {subject} {session} {task}')
        confound_summary = read_confound_summary(confound, inspected['n_frames'])
        rows.append({'subject': subject, 'session': session, 'task': task, 'task_order': TASKS.index(task), 'roi_path': path.relative_to(volume_root).as_posix(), 'confounds_path': confound.relative_to(dataset_root).as_posix(), 'modal_expected_frames': modal_lengths[task], 'modal_length_match': inspected['n_frames'] == modal_lengths[task], **inspected, **confound_summary})
    runs = pd.DataFrame(rows).sort_values(['subject', 'session', 'task_order']).reset_index(drop=True)
    duplicate = runs.duplicated(['subject', 'session', 'task'], keep=False)
    if duplicate.any():
        raise RuntimeError('duplicate subject/session/task ROI files:\n' + runs.loc[duplicate, ['subject', 'session', 'task', 'roi_path']].to_string(index=False))
    if runs.nonfinite_total.sum() != 0:
        raise RuntimeError('non-finite values found in CEBRA ROI inputs')
    subjects = sorted(runs.subject.unique())
    membership_rows: list[dict] = []
    for subject in subjects:
        local = runs[runs.subject == subject]
        complete_both = all((len(local[(local.session == session) & local.task.isin(TASKS)]) == len(TASKS) for session in SESSIONS))
        modal_both = complete_both and bool(local.modal_length_match.all())
        drug = local[local.session == 'ses-02']
        psilocybin_mean_fd = float(drug.mean_fd.mean()) if len(drug) == len(TASKS) else math.nan
        membership_rows.append({'subject': subject, 'all_complete_both_sessions': complete_both, 'modal_length_both_sessions': modal_both, 'psilocybin_mean_fd_across_tasks': psilocybin_mean_fd})
    membership = pd.DataFrame(membership_rows)
    modal_candidates = membership[membership.modal_length_both_sessions].sort_values(['psilocybin_mean_fd_across_tasks', 'subject'], na_position='last')
    if len(modal_candidates) < proxy_size:
        raise RuntimeError(f'only {len(modal_candidates)} modal-length paired participants; cannot create n={proxy_size} proxy')
    proxy_subjects = set(modal_candidates.head(proxy_size).subject)
    membership['paper_size_low_fd_proxy'] = membership.subject.isin(proxy_subjects)
    write_frame(runs, output_dir / 'cebra_run_manifest')
    write_frame(membership, output_dir / 'cebra_cohort_membership')
    cohort_counts = {'all_complete': int(membership.all_complete_both_sessions.sum()), 'modal_length': int(membership.modal_length_both_sessions.sum()), 'paper_size_low_fd_proxy': int(membership.paper_size_low_fd_proxy.sum())}
    fit_estimates = {}
    for name, n_subjects in cohort_counts.items():
        author = n_subjects * len(SESSIONS) * author_seeds
        blocked = n_subjects * len(SESSIONS) * blocked_folds * blocked_seeds
        transfer = n_subjects * 2 * transfer_seeds
        participant_heldout = len(SESSIONS) * participant_folds * participant_seeds
        fit_estimates[name] = {'author_like': author, 'blocked_purged': blocked, 'cross_session': transfer, 'participant_heldout_shared': participant_heldout, 'total': author + blocked + transfer + participant_heldout}
    nonmodal = runs[~runs.modal_length_match][['subject', 'session', 'task', 'n_frames', 'modal_expected_frames']].to_dict('records')
    summary = {'schema': 'cebra-manifest-v1', 'variants': list(variants), 'task_order': list(TASKS), 'sessions': list(SESSIONS), 'modal_lengths': modal_lengths, 'run_count': len(runs), 'cohort_counts': cohort_counts, 'proxy_size': proxy_size, 'nonmodal_runs': nonmodal, 'planned_fit_estimates': fit_estimates, 'planned_fit_estimates_all_variants': {name: {branch: value * len(variants) for branch, value in estimates.items()} for name, estimates in fit_estimates.items()}}
    atomic_text(output_dir / 'cebra_manifest_summary.json', json.dumps(summary, indent=2) + '\n')
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume-root', required=True, type=Path)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--variants', nargs='+', default=['author_literal', 'corrected_centered_pc1'])
    parser.add_argument('--modal-lengths', default=','.join((f'{task}={DEFAULT_MODAL_LENGTHS[task]}' for task in TASKS)))
    parser.add_argument('--proxy-size', type=int, default=54)
    parser.add_argument('--author-seeds', type=int, default=10)
    parser.add_argument('--blocked-seeds', type=int, default=3)
    parser.add_argument('--blocked-folds', type=int, default=4)
    parser.add_argument('--transfer-seeds', type=int, default=3)
    parser.add_argument('--participant-seeds', type=int, default=3)
    parser.add_argument('--participant-folds', type=int, default=5)
    args = parser.parse_args()
    summary = build_manifest(args.volume_root, args.dataset_root, args.output_dir, tuple(args.variants), parse_modal_lengths(args.modal_lengths), args.proxy_size, args.author_seeds, args.blocked_seeds, args.blocked_folds, args.transfer_seeds, args.participant_seeds, args.participant_folds)
    print(json.dumps(summary, ensure_ascii=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
