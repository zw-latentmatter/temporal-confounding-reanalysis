from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy import stats
TASKS = ('rest', 'meditation', 'music', 'movie')
SESSIONS = ('ses-01', 'ses-02')
IDENTITY_COLUMNS = ['backend', 'analysis', 'variant', 'cohort', 'subject', 'session', 'train_session', 'test_session', 'direction', 'readout', 'cebra_model_architecture', 'cebra_time_offsets', 'cebra_output_dimension', 'cebra_max_iterations', 'cebra_batch_size', 'cebra_learning_rate', 'cebra_temperature', 'cebra_distance', 'cebra_num_hidden_units', 'svm_C', 'purge_frames']
OUTCOME_COLUMNS = ['accuracy', 'balanced_accuracy', 'balanced_accuracy_reported', 'balanced_accuracy_declared_class_set', 'test_contains_all_declared_classes', *(f'recall_{task}' for task in TASKS), 'silhouette_mean', *(f'silhouette_{task}' for task in TASKS)]
FOUR_CLASS_READOUTS = {'resubstitution', 'random_stratified_frames', 'four_contexts', 'participant_heldout_shared_four_contexts'}

def _chance_level_for_readout(readout: str, metric: str) -> float | None:
    if metric not in {'accuracy', 'balanced_accuracy', *(f'recall_{task}' for task in TASKS)}:
        return None
    if readout == 'movie_vs_closed_contexts':
        return 0.5
    if readout == 'closed_only_three_contexts' or readout.startswith('leave_out_'):
        return 1.0 / 3.0
    if readout in FOUR_CLASS_READOUTS:
        return 0.25
    raise ValueError(f'unknown scored readout has no declared chance level: {readout}')
FIT_BASE_COLUMNS = ['source_cache_id', 'job_key', 'status', 'backend', 'analysis', 'variant', 'cohort', 'job_subject', 'subject', 'session', 'train_session', 'test_session', 'direction', 'fold', 'seed_repeat', 'seed', 'job_index', 'readout', 'split', *OUTCOME_COLUMNS, 'train_frames', 'test_frames', *(f'train_count_{task}' for task in TASKS), *(f'test_count_{task}' for task in TASKS), 'confusion_matrix_row_normalized_json', 'runtime_seconds', 'author_like_nonindependent', 'participant_independent_validation']
PARTICIPANT_BASE_COLUMNS = [*IDENTITY_COLUMNS, 'seed_n', 'split_n', 'fit_job_n', *OUTCOME_COLUMNS, *(f'{metric}_{suffix}' for metric in OUTCOME_COLUMNS for suffix in ('seed_mean', 'seed_sd', 'seed_min', 'seed_max')), 'author_like_nonindependent', 'participant_independent_validation']
GROUP_COLUMNS = ['summary_type', *(column for column in IDENTITY_COLUMNS if column != 'subject'), 'metric', 'n_participants', 'mean', 'sd', 'sem', 'median', 'q25', 'q75', 'minimum', 'maximum', 'ci95_low', 'ci95_high', 'chance_level', 'mean_minus_chance', 't_vs_chance', 'p_two_sided_vs_chance', 'inference_unit', 'author_like_nonindependent', 'participant_independent_validation']
PAIRED_COLUMNS = [*(column for column in IDENTITY_COLUMNS if column not in {'session', 'train_session', 'test_session', 'direction'}), 'metric', 'ses_01_value', 'ses_02_value', 'difference_ses02_minus_ses01', 'ses_01_seed_n', 'ses_02_seed_n', 'complete_pair', 'author_like_nonindependent', 'participant_independent_validation']

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)

def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix('.csv')
    parquet_path = stem.with_suffix('.parquet')
    temporary_csv = csv_path.with_name(f'.{csv_path.name}.partial')
    temporary_parquet = parquet_path.with_name(f'.{parquet_path.name}.partial')
    frame.to_csv(temporary_csv, index=False)
    frame.to_parquet(temporary_parquet, index=False)
    temporary_csv.replace(csv_path)
    temporary_parquet.replace(parquet_path)

def ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    ordered = list(columns) + [column for column in frame.columns if column not in columns]
    return frame.loc[:, ordered]

def compact_json(value) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(',', ':'))

def flatten_mapping(prefix: str, value: Mapping, output: dict) -> None:
    for key, item in value.items():
        name = f'{prefix}_{key}' if prefix else str(key)
        if isinstance(item, Mapping):
            flatten_mapping(name, item, output)
        elif isinstance(item, (list, tuple)):
            output[name] = compact_json(item)
        else:
            output[name] = item

def numeric_structure(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, np.generic)):
        return True
    if isinstance(value, (list, tuple)):
        return all((numeric_structure(item) for item in value))
    if isinstance(value, Mapping):
        return all((numeric_structure(item) for item in value.values()))
    return False

def flatten_numeric_mapping(prefix: str, value: Mapping, output: dict) -> None:
    for key, item in value.items():
        name = f'{prefix}_{key}' if prefix else str(key)
        if isinstance(item, Mapping):
            flatten_numeric_mapping(name, item, output)
        elif numeric_structure(item):
            output[name] = compact_json(item) if isinstance(item, (list, tuple)) else item

def normalize_subject(value) -> str | None:
    match = re.search('(?:sub[-_])?PC[-_]?(\\d+)', str(value), flags=re.IGNORECASE)
    if not match:
        return None
    return f'sub-PC{int(match.group(1)):03d}'

def branch_labels(analysis: str) -> dict:
    analysis = str(analysis or '')
    return {'author_like_nonindependent': analysis == 'author_like', 'participant_independent_validation': analysis == 'participant_heldout'}

def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f'expected a JSON object in {path}')
    return payload

def discover_results(input_dir: Path, allow_failed_jobs: bool) -> tuple[list[tuple[Path, dict]], list[Path]]:
    cache_dir = input_dir / 'cache'
    failure_paths = sorted(cache_dir.glob('*.failed.json'))
    if failure_paths and (not allow_failed_jobs):
        preview = ', '.join((path.name for path in failure_paths[:8]))
        raise RuntimeError(f'found {len(failure_paths)} failed CEBRA cache file(s): {preview}; resolve them or pass --allow-failed-jobs for an explicitly incomplete summary')
    results: list[tuple[Path, dict]] = []
    for path in sorted(cache_dir.glob('*.json')):
        if path.name.endswith('.failed.json'):
            continue
        payload = read_json(path)
        if payload.get('status') != 'completed':
            raise RuntimeError(f'non-completed result stored as a normal cache file: {path}')
        results.append((path, payload))
    keys = [str(payload.get('job_key') or path.stem) for path, payload in results]
    duplicated = pd.Series(keys, dtype='object').duplicated(keep=False)
    if duplicated.any():
        duplicate_keys = sorted(pd.Series(keys, dtype='object')[duplicated].unique())
        raise RuntimeError(f'duplicate completed job keys: {duplicate_keys[:8]}')
    return (results, failure_paths)

def reconcile_job_plans(input_dir: Path) -> dict:
    paths = sorted(input_dir.glob('job_plan_shard-*.json'))
    if not paths:
        return {'plan_file_count': 0, 'unique_shard_plan_count': 0, 'shard_count': None, 'available_shard_indices': [], 'plans_cover_all_shards': False, 'advertised_total_jobs_all_shards': None, 'expected_jobs_from_available_shards': None, 'explicit_unique_planned_jobs': None, 'duplicate_explicit_job_entries_removed': 0, 'planned_job_keys': None}
    by_shard: dict[tuple[int, int], tuple[Path, dict]] = {}
    for path in paths:
        plan = read_json(path)
        shard_count = int(plan['shard_count'])
        shard_index = int(plan['shard_index'])
        key = (shard_count, shard_index)
        if key in by_shard:
            prior_path, prior = by_shard[key]
            if compact_json(prior) != compact_json(plan):
                raise RuntimeError(f'conflicting plans for shard {shard_index}/{shard_count}: {prior_path} and {path}')
            continue
        by_shard[key] = (path, plan)
    shard_counts = {key[0] for key in by_shard}
    if len(shard_counts) != 1:
        raise RuntimeError(f'job plans disagree on shard_count: {sorted(shard_counts)}')
    shard_count = next(iter(shard_counts))
    shard_indices = sorted((key[1] for key in by_shard))
    if any((index < 0 or index >= shard_count for index in shard_indices)):
        raise RuntimeError('job plan has shard_index outside its declared shard_count')
    advertised_totals = {int(plan['total_jobs_all_shards']) for _, plan in by_shard.values()}
    if len(advertised_totals) != 1:
        raise RuntimeError(f'job plans disagree on total_jobs_all_shards: {sorted(advertised_totals)}')
    advertised_total = next(iter(advertised_totals))
    expected_from_shards = sum((int(plan['jobs_this_shard']) for _, plan in by_shard.values()))
    if expected_from_shards > advertised_total:
        raise RuntimeError('sum of unique jobs_this_shard exceeds total_jobs_all_shards; plans are inconsistent')
    explicit_jobs: list[str] = []
    all_have_explicit_jobs = True
    planned_job_keys: list[str] = []
    all_have_job_keys = True
    for _, plan in by_shard.values():
        job_keys = plan.get('job_keys_this_shard')
        if job_keys is None:
            all_have_job_keys = False
        elif not isinstance(job_keys, list):
            raise RuntimeError("job plan 'job_keys_this_shard' must be a list")
        elif len(job_keys) != int(plan['jobs_this_shard']):
            raise RuntimeError('job plan key list does not match jobs_this_shard')
        else:
            planned_job_keys.extend((str(value) for value in job_keys))
        jobs = plan.get('jobs')
        if jobs is None:
            all_have_explicit_jobs = False
            continue
        if not isinstance(jobs, list):
            raise RuntimeError("job plan 'jobs' must be null or a list")
        if len(jobs) != int(plan['jobs_this_shard']):
            raise RuntimeError('job plan jobs list does not match jobs_this_shard')
        explicit_jobs.extend((compact_json(job) for job in jobs))
    explicit_unique = len(set(explicit_jobs)) if all_have_explicit_jobs else None
    duplicate_explicit = len(explicit_jobs) - len(set(explicit_jobs))
    if explicit_unique is not None:
        expected_from_shards = explicit_unique
    planned_keys = sorted(planned_job_keys) if all_have_job_keys else None
    if planned_keys is not None and len(planned_keys) != len(set(planned_keys)):
        raise RuntimeError('job plans contain duplicate job keys across shards')
    if planned_keys is not None:
        expected_from_shards = len(planned_keys)
    return {'plan_file_count': len(paths), 'unique_shard_plan_count': len(by_shard), 'shard_count': shard_count, 'available_shard_indices': shard_indices, 'plans_cover_all_shards': shard_indices == list(range(shard_count)), 'advertised_total_jobs_all_shards': advertised_total, 'expected_jobs_from_available_shards': expected_from_shards, 'explicit_unique_planned_jobs': explicit_unique, 'duplicate_explicit_job_entries_removed': duplicate_explicit, 'planned_job_keys': planned_keys}

def _metric_subject(analysis: str, job: Mapping, metric: Mapping) -> str | None:
    if analysis == 'participant_heldout':
        value = metric.get('test_subject') or metric.get('subject')
    else:
        value = metric.get('subject') or metric.get('test_subject') or job.get('subject') or job.get('test_subject')
    return normalize_subject(value) if value is not None else None

def flatten_results(results: Sequence[tuple[Path, dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    for path, payload in results:
        backend = str(payload.get('backend', ''))
        job = payload.get('job') or {}
        if not isinstance(job, Mapping):
            raise RuntimeError(f'result job is not an object: {path}')
        analysis = str(job.get('analysis') or payload.get('analysis') or '')
        metrics = payload.get('metrics')
        if not isinstance(metrics, list) or not metrics:
            raise RuntimeError(f'completed result has no metric records: {path}')
        common = {'source_cache_id': path.name, 'job_key': str(payload.get('job_key') or path.stem), 'status': payload.get('status'), 'backend': backend, 'analysis': analysis, 'variant': job.get('variant'), 'cohort': job.get('cohort'), 'job_subject': normalize_subject(job.get('subject')) if job.get('subject') else None, 'session': job.get('session'), 'train_session': job.get('train_session'), 'test_session': job.get('test_session'), 'direction': job.get('direction'), 'fold': job.get('fold'), 'seed_repeat': job.get('seed_repeat'), 'seed': job.get('seed'), 'job_index': job.get('job_index'), 'runtime_seconds': payload.get('runtime_seconds'), **branch_labels(analysis)}
        for prefix, key in (('cebra', 'cebra_configuration'), ('svm', 'svm_configuration')):
            value = payload.get(key)
            if isinstance(value, Mapping):
                flatten_mapping(prefix, value, common)
        for prefix, key in (('seed_audit', 'seed_audit'), ('scaler_audit', 'scaler_audit'), ('encoder_audit', 'encoder_audit'), ('split_audit', 'split_audit')):
            value = payload.get(key)
            if isinstance(value, Mapping):
                flatten_numeric_mapping(prefix, value, common)
        if 'split_audit_purge_frames' in common:
            common['purge_frames'] = common['split_audit_purge_frames']
        elif job.get('purge_frames') is not None:
            common['purge_frames'] = job.get('purge_frames')
        else:
            common['purge_frames'] = None
        silhouette = payload.get('silhouette') or payload.get('test_centers_silhouette')
        if isinstance(silhouette, Mapping):
            for key, value in silhouette.items():
                common[str(key)] = value
        for metric in metrics:
            if not isinstance(metric, Mapping):
                raise RuntimeError(f'metric record is not an object: {path}')
            row = dict(common)
            row['subject'] = _metric_subject(analysis, job, metric)
            row['session'] = metric.get('session', row.get('session'))
            row['train_session'] = metric.get('train_session', row.get('train_session'))
            row['test_session'] = metric.get('test_session', row.get('test_session'))
            row['direction'] = metric.get('direction', row.get('direction'))
            row['readout'] = metric.get('readout')
            row['split'] = metric.get('split', job.get('fold', 0))
            for name in ('accuracy', 'train_frames', 'test_frames'):
                row[name] = metric.get(name)
            class_names = metric.get('class_names')
            if not isinstance(class_names, list):
                class_names = list(TASKS)
            recalls = metric.get('class_recall')
            if isinstance(recalls, list):
                for class_name, value in zip(class_names, recalls):
                    row[f'recall_{class_name}'] = value
            row['balanced_accuracy_reported'] = metric.get('balanced_accuracy')
            if isinstance(recalls, list) and len(recalls) == len(class_names):
                declared_values = pd.to_numeric(pd.Series(recalls), errors='coerce').to_numpy(dtype=float)
                row['balanced_accuracy'] = float(np.mean(declared_values)) if np.isfinite(declared_values).all() else metric.get('balanced_accuracy')
                row['balanced_accuracy_declared_class_set'] = True
            else:
                row['balanced_accuracy'] = metric.get('balanced_accuracy')
                row['balanced_accuracy_declared_class_set'] = bool(metric.get('balanced_accuracy_declared_class_set', False))
            for prefix, name in (('train_count', 'train_class_counts'), ('test_count', 'test_class_counts')):
                counts = metric.get(name)
                if isinstance(counts, list):
                    for class_name, value in zip(class_names, counts):
                        row[f'{prefix}_{class_name}'] = value
                    if prefix == 'test_count':
                        row['test_contains_all_declared_classes'] = bool(len(counts) == len(class_names) and all((int(value) > 0 for value in counts)))
            matrix = metric.get('confusion_matrix_row_normalized')
            if matrix is not None:
                row['confusion_matrix_row_normalized_json'] = compact_json(matrix)
                if isinstance(matrix, list):
                    for true_index, true_task in enumerate(class_names):
                        if true_index >= len(matrix) or not isinstance(matrix[true_index], list):
                            continue
                        for predicted_index, predicted_task in enumerate(class_names):
                            if predicted_index < len(matrix[true_index]):
                                row[f'confusion_true_{true_task}_pred_{predicted_task}'] = matrix[true_index][predicted_index]
            extra_metric = {key: value for key, value in metric.items() if key not in {'test_subject', 'subject', 'session', 'train_session', 'test_session', 'direction', 'readout', 'split', 'accuracy', 'balanced_accuracy', 'class_names', 'class_label_values', 'class_recall', 'confusion_matrix_row_normalized', 'train_frames', 'test_frames', 'train_class_counts', 'test_class_counts'}}
            flatten_mapping('metric', extra_metric, row)
            rows.append(row)
    frame = ensure_columns(pd.DataFrame(rows), FIT_BASE_COLUMNS)
    if frame.empty:
        return frame
    if frame.subject.isna().any():
        bad = frame.loc[frame.subject.isna(), ['job_key', 'analysis', 'readout']]
        raise RuntimeError('metric rows lack participant identity:\n' + bad.head().to_string(index=False))
    duplicate_columns = ['job_key', 'subject', 'readout', 'split']
    if 'metric_null_index' in frame.columns:
        duplicate_columns.append('metric_null_index')
    duplicated = frame.duplicated(duplicate_columns, keep=False)
    if duplicated.any():
        raise RuntimeError('duplicate readout metric rows:\n' + frame.loc[duplicated, duplicate_columns].head().to_string(index=False))
    return frame

def _numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors='coerce')
    return float(values.mean()) if values.notna().any() else math.nan

def aggregate_participant_metrics(fit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fit.empty:
        return (pd.DataFrame(), ensure_columns(pd.DataFrame(), PARTICIPANT_BASE_COLUMNS))
    local = fit.copy()
    for column in IDENTITY_COLUMNS:
        if column not in local:
            local[column] = np.nan
    local['seed_repeat_aggregation'] = pd.to_numeric(local.seed_repeat, errors='coerce')
    missing_repeat = local.seed_repeat_aggregation.isna()
    local.loc[missing_repeat, 'seed_repeat_aggregation'] = pd.to_numeric(local.loc[missing_repeat, 'seed'], errors='coerce')
    seed_keys = [*IDENTITY_COLUMNS, 'seed_repeat_aggregation']
    seed_rows: list[dict] = []
    for keys, group in local.groupby(seed_keys, dropna=False, sort=True):
        row = dict(zip(seed_keys, keys))
        for metric in OUTCOME_COLUMNS:
            row[metric] = _numeric_mean(group[metric])
        row['split_n'] = int(len(group))
        row['fit_job_n'] = int(group.job_key.nunique())
        row['seed_values_json'] = compact_json(sorted({int(value) for value in pd.to_numeric(group.seed, errors='coerce').dropna()}))
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    participant_keys = list(IDENTITY_COLUMNS)
    participant_rows: list[dict] = []
    for keys, group in seed_frame.groupby(participant_keys, dropna=False, sort=True):
        row = dict(zip(participant_keys, keys))
        for metric in OUTCOME_COLUMNS:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            row[metric] = float(values.median()) if len(values) else math.nan
            row[f'{metric}_seed_mean'] = float(values.mean()) if len(values) else math.nan
            row[f'{metric}_seed_sd'] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            row[f'{metric}_seed_min'] = float(values.min()) if len(values) else math.nan
            row[f'{metric}_seed_max'] = float(values.max()) if len(values) else math.nan
        row['seed_n'] = int(group.seed_repeat_aggregation.nunique())
        row['split_n'] = int(group.split_n.sum())
        row['fit_job_n'] = int(group.fit_job_n.sum())
        row.update(branch_labels(str(row.get('analysis') or '')))
        participant_rows.append(row)
    participant = ensure_columns(pd.DataFrame(participant_rows), PARTICIPANT_BASE_COLUMNS)
    return (seed_frame, participant)

def _summary_statistics(values: Iterable[float], chance_level: float | None) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = int(len(array))
    result = {'n_participants': n, 'mean': math.nan, 'sd': math.nan, 'sem': math.nan, 'median': math.nan, 'q25': math.nan, 'q75': math.nan, 'minimum': math.nan, 'maximum': math.nan, 'ci95_low': math.nan, 'ci95_high': math.nan, 'chance_level': chance_level, 'mean_minus_chance': math.nan, 't_vs_chance': math.nan, 'p_two_sided_vs_chance': math.nan}
    if not n:
        return result
    mean = float(array.mean())
    result.update({'mean': mean, 'median': float(np.median(array)), 'q25': float(np.quantile(array, 0.25)), 'q75': float(np.quantile(array, 0.75)), 'minimum': float(array.min()), 'maximum': float(array.max())})
    if n > 1:
        sd = float(array.std(ddof=1))
        sem = sd / math.sqrt(n)
        critical = float(stats.t.ppf(0.975, n - 1))
        result.update({'sd': sd, 'sem': sem, 'ci95_low': mean - critical * sem, 'ci95_high': mean + critical * sem})
    if chance_level is not None:
        result['mean_minus_chance'] = mean - chance_level
        if n > 1 and np.std(array, ddof=1) > 0:
            test = stats.ttest_1samp(array, popmean=chance_level)
            result['t_vs_chance'] = float(test.statistic)
            result['p_two_sided_vs_chance'] = float(test.pvalue)
    return result

def summarize_groups(participant: pd.DataFrame) -> pd.DataFrame:
    if participant.empty:
        return ensure_columns(pd.DataFrame(), GROUP_COLUMNS)
    group_keys = [column for column in IDENTITY_COLUMNS if column != 'subject']
    rows: list[dict] = []
    for keys, group in participant.groupby(group_keys, dropna=False, sort=True):
        identifiers = dict(zip(group_keys, keys))
        for metric in OUTCOME_COLUMNS:
            values = pd.to_numeric(group[metric], errors='coerce')
            if not values.notna().any():
                continue
            chance = _chance_level_for_readout(str(identifiers['readout']), metric)
            row = {'summary_type': 'participant_level', **identifiers, 'metric': metric, **_summary_statistics(values, chance), 'inference_unit': 'participant', **branch_labels(str(identifiers.get('analysis') or ''))}
            rows.append(row)
    return ensure_columns(pd.DataFrame(rows), GROUP_COLUMNS)

def paired_session_differences(participant: pd.DataFrame) -> pd.DataFrame:
    if participant.empty:
        return ensure_columns(pd.DataFrame(), PAIRED_COLUMNS)
    local = participant[participant.session.isin(SESSIONS) & participant.train_session.isna() & participant.test_session.isna()].copy()
    if local.empty:
        return ensure_columns(pd.DataFrame(), PAIRED_COLUMNS)
    pair_keys = [column for column in IDENTITY_COLUMNS if column not in {'session', 'train_session', 'test_session', 'direction'}]
    rows: list[dict] = []
    for keys, group in local.groupby(pair_keys, dropna=False, sort=True):
        identifiers = dict(zip(pair_keys, keys))
        analysis = str(identifiers.get('analysis') or '')
        for metric in OUTCOME_COLUMNS:
            by_session = group[['session', metric]].dropna(subset=[metric]).drop_duplicates('session').set_index('session')[metric]
            if by_session.empty:
                continue
            baseline = float(by_session['ses-01']) if 'ses-01' in by_session else math.nan
            drug = float(by_session['ses-02']) if 'ses-02' in by_session else math.nan
            complete = math.isfinite(baseline) and math.isfinite(drug)
            seed_by_session = group.drop_duplicates('session').set_index('session')['seed_n']
            rows.append({**identifiers, 'metric': metric, 'ses_01_value': baseline, 'ses_02_value': drug, 'difference_ses02_minus_ses01': drug - baseline if complete else math.nan, 'ses_01_seed_n': seed_by_session.get('ses-01', math.nan), 'ses_02_seed_n': seed_by_session.get('ses-02', math.nan), 'complete_pair': complete, **branch_labels(analysis)})
    return ensure_columns(pd.DataFrame(rows), PAIRED_COLUMNS)

def append_paired_group_summaries(group_summary: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    complete = paired[paired.complete_pair.astype(bool)].copy() if not paired.empty else paired
    if complete.empty:
        return group_summary
    pair_group_keys = [column for column in IDENTITY_COLUMNS if column not in {'subject', 'session', 'train_session', 'test_session', 'direction'}]
    rows: list[dict] = []
    for keys, local in complete.groupby([*pair_group_keys, 'metric'], dropna=False, sort=True):
        identifiers = dict(zip([*pair_group_keys, 'metric'], keys))
        analysis = str(identifiers.get('analysis') or '')
        rows.append({'summary_type': 'paired_ses-02_minus_ses-01', **identifiers, 'session': 'ses-02_minus_ses-01', 'train_session': None, 'test_session': None, 'direction': None, **_summary_statistics(local.difference_ses02_minus_ses01, None), 'inference_unit': 'participant', **branch_labels(analysis)})
    combined = pd.concat([group_summary, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return ensure_columns(combined, GROUP_COLUMNS)

def cross_session_summary(participant: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    directional = group_summary[(group_summary.analysis == 'cross_session') & (group_summary.summary_type == 'participant_level')].copy()
    if not directional.empty:
        directional['summary_scope'] = 'direction_specific'
    local = participant[participant.analysis == 'cross_session'].copy()
    if local.empty:
        return ensure_columns(directional, [*GROUP_COLUMNS, 'summary_scope'])
    bidirectional_keys = [column for column in IDENTITY_COLUMNS if column not in {'train_session', 'test_session', 'direction', 'subject'}]
    rows: list[dict] = []
    for keys, group in local.groupby(bidirectional_keys, dropna=False, sort=True):
        identifiers = dict(zip(bidirectional_keys, keys))
        for metric in OUTCOME_COLUMNS:
            subject_means = group.groupby('subject', dropna=False)[metric].mean().dropna()
            if subject_means.empty:
                continue
            rows.append({'summary_type': 'participant_level', **identifiers, 'train_session': None, 'test_session': None, 'direction': 'bidirectional_mean', 'metric': metric, **_summary_statistics(subject_means, 0.25 if metric in {'accuracy', 'balanced_accuracy', *(f'recall_{task}' for task in TASKS)} else None), 'inference_unit': 'participant', 'summary_scope': 'mean_of_two_transfer_directions_per_participant', **branch_labels('cross_session')})
    combined = pd.concat([directional, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return ensure_columns(combined, [*GROUP_COLUMNS, 'summary_scope'])

def participant_heldout_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    return group_summary[group_summary.analysis == 'participant_heldout'].copy()

def summarize(args) -> dict:
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    results, failure_paths = discover_results(input_dir, args.allow_failed_jobs)
    reconciliation = reconcile_job_plans(input_dir)
    planned_job_keys = reconciliation.pop('planned_job_keys', None)
    expected = reconciliation['expected_jobs_from_available_shards']
    completed_count = len(results)
    if not reconciliation['plans_cover_all_shards'] and (not args.allow_incomplete):
        raise RuntimeError('job plans do not cover every declared shard')
    if expected is not None and completed_count != expected and (not args.allow_incomplete):
        raise RuntimeError(f'completed cache count ({completed_count}) does not match expected unique jobs from available shard plans ({expected}); pass --allow-incomplete only for an explicitly incomplete audit')
    observed_job_keys = {str(payload.get('job_key') or path.stem) for path, payload in results}
    planned_key_set = set(planned_job_keys) if planned_job_keys is not None else None
    missing_job_keys = sorted(planned_key_set - observed_job_keys) if planned_key_set is not None else []
    extra_job_keys = sorted(observed_job_keys - planned_key_set) if planned_key_set is not None else []
    if (missing_job_keys or extra_job_keys) and (not args.allow_incomplete):
        raise RuntimeError(f'completed cache job keys do not exactly match the declared shard plans: missing={missing_job_keys[:8]}, extra={extra_job_keys[:8]}')
    fit = flatten_results(results)
    ordinary_fit = fit[~fit.readout.eq('circular_block_shift_null_four_contexts')].copy()
    seed_metrics, participant = aggregate_participant_metrics(ordinary_fit)
    group = summarize_groups(participant)
    paired = paired_session_differences(participant)
    group = append_paired_group_summaries(group, paired)
    cross = cross_session_summary(participant, group)
    heldout = participant_heldout_summary(group)
    write_frame(fit, output_dir / 'cebra_fit_metrics')
    write_frame(participant, output_dir / 'cebra_participant_metrics')
    write_frame(group, output_dir / 'cebra_group_summary')
    write_frame(paired, output_dir / 'cebra_paired_session_differences')
    write_frame(cross, output_dir / 'cebra_cross_session_summary')
    write_frame(heldout, output_dir / 'cebra_participant_heldout_summary')
    reconciliation = {**reconciliation, 'completed_cache_file_count': completed_count, 'completed_unique_job_count': int(fit.job_key.nunique()) if not fit.empty else 0, 'failed_cache_file_count': len(failure_paths), 'count_matches_available_plans': expected is None or completed_count == expected, 'exact_job_key_set_available': planned_job_keys is not None, 'missing_planned_job_key_count': len(missing_job_keys), 'missing_planned_job_key_examples': missing_job_keys[:10], 'extra_completed_job_key_count': len(extra_job_keys), 'extra_completed_job_key_examples': extra_job_keys[:10], 'summary_incomplete': bool(failure_paths or not reconciliation['plans_cover_all_shards'] or (expected is not None and completed_count != expected) or missing_job_keys or extra_job_keys)}
    declared_macro = ordinary_fit['balanced_accuracy_declared_class_set'].fillna(False).astype(bool) if not ordinary_fit.empty else pd.Series(dtype=bool)
    all_classes = ordinary_fit['test_contains_all_declared_classes'].fillna(False).astype(bool) if not ordinary_fit.empty else pd.Series(dtype=bool)
    missing_class_rows = ordinary_fit.loc[~all_classes] if not ordinary_fit.empty else ordinary_fit
    summary = {'schema': 'cebra-summary-v2', 'result_reconciliation': reconciliation, 'seed_level_intermediate_rows': len(seed_metrics), 'participant_metric_rows': len(participant), 'group_summary_rows': len(group), 'all_fit_rows_use_declared_class_macro': bool(declared_macro.all()), 'fit_rows_missing_at_least_one_declared_test_class': int(len(missing_class_rows)), 'tables': {'cebra_fit_metrics': len(fit), 'cebra_participant_metrics': len(participant), 'cebra_group_summary': len(group), 'cebra_paired_session_differences': len(paired), 'cebra_cross_session_summary': len(cross), 'cebra_participant_heldout_summary': len(heldout)}, 'balanced_accuracy_group_summary': group[group.metric.eq('balanced_accuracy')].to_dict('records'), 'cross_session_balanced_accuracy_summary': cross[cross.metric.eq('balanced_accuracy')].to_dict('records'), 'participant_heldout_balanced_accuracy_summary': heldout[heldout.metric.eq('balanced_accuracy')].to_dict('records')}
    atomic_text(output_dir / 'cebra_reanalysis_summary.json', json.dumps(json_ready(summary), indent=2, sort_keys=True) + '\n')
    return summary

def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize cached CEBRA jobs using participant-level inference.')
    parser.add_argument('--input-dir', '--results-dir', '--cebra-dir', dest='input_dir', required=True, type=Path, help='Directory containing cache/ and job_plan_shard-*.json files.')
    parser.add_argument('--output-dir', type=Path, help='Summary destination (default: input directory).')
    parser.add_argument('--allow-failed-jobs', action='store_true', help='Produce an explicitly incomplete summary even when *.failed.json files exist.')
    parser.add_argument('--allow-incomplete', action='store_true', help='Permit completed-result counts that do not match the available shard plans.')
    args = parser.parse_args()
    summary = summarize(args)
    print(json.dumps({'completed_jobs': summary['result_reconciliation']['completed_unique_job_count'], 'participant_metric_rows': summary['tables']['cebra_participant_metrics']}))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
