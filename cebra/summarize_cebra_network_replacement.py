from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping
import numpy as np
import pandas as pd
from scipy import stats
from run_cebra_network_replacement import ANALYSIS_SCHEMA, JOINT_TARGET
from summarize_cebra import atomic_text, write_frame
TASKS = ('rest', 'meditation', 'music', 'movie')
DROP_METRICS = ('accuracy_drop_signed', 'balanced_accuracy_drop_signed', *(f'recall_drop_signed_{task}' for task in TASKS))
INTERACTION_METRICS = ('accuracy_interaction_signed', 'balanced_accuracy_interaction_signed', *(f'recall_interaction_signed_{task}' for task in TASKS))

def nested(mapping: Mapping | None, *keys, default=math.nan):
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value

def flatten_drop(drop: Mapping | None) -> dict:
    return {'accuracy_drop_signed': nested(drop, 'accuracy_drop_signed'), 'balanced_accuracy_drop_signed': nested(drop, 'balanced_accuracy_drop_signed'), **{f'recall_drop_signed_{task}': nested(drop, 'recall_drop_signed', task) for task in TASKS}}

def flatten_interaction(interaction: Mapping) -> dict:
    return {'accuracy_interaction_signed': nested(interaction, 'accuracy_interaction_signed'), 'balanced_accuracy_interaction_signed': nested(interaction, 'balanced_accuracy_interaction_signed'), **{f'recall_interaction_signed_{task}': nested(interaction, 'recall_interaction_signed', task) for task in TASKS}}

def flatten_ood(prefix: str, payload: Mapping | None) -> dict:
    result = {}
    if not isinstance(payload, Mapping):
        return result
    for key, value in payload.items():
        if isinstance(value, (int, float, bool)) or value is None:
            result[f'{prefix}_{key}'] = value
    return result

def load_results(input_dir: Path) -> list[tuple[Path, dict]]:
    cache = input_dir / 'cache'
    failures = sorted(cache.glob('*.failed.json'))
    if failures:
        raise RuntimeError(f'found {len(failures)} failed network-replacement jobs')
    results = []
    for path in sorted(cache.glob('network_replacement__*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema') != ANALYSIS_SCHEMA or payload.get('status') != 'completed':
            continue
        artifact_id = payload.get('artifact_id')
        artifact = input_dir / artifact_id if artifact_id else None
        if not artifact or not artifact.exists():
            raise RuntimeError(f'missing job artifact referenced by {path}')
        results.append((path, payload))
    if not results:
        raise RuntimeError(f'no completed network-replacement results in {cache}')
    return results

def reconcile_plans(input_dir: Path, completed_job_keys: set[str], allow_incomplete: bool) -> dict:
    completed_jobs = len(completed_job_keys)
    paths = sorted(input_dir.glob('network_replacement_plan_shard-*.json'))
    if not paths:
        if allow_incomplete:
            return {'available': False, 'completed_jobs': completed_jobs}
        raise RuntimeError('no network-replacement shard plans were found')
    plans = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    expected_values = {int(plan['total_jobs_all_shards']) for plan in plans}
    shard_counts = {int(plan['shard_count']) for plan in plans}
    if len(expected_values) != 1 or len(shard_counts) != 1:
        raise RuntimeError('network-replacement shard plans disagree')
    expected = next(iter(expected_values))
    shard_count = next(iter(shard_counts))
    shard_indices = sorted({int(plan['shard_index']) for plan in plans})
    planned_keys: list[str] = []
    all_have_keys = True
    for plan in plans:
        keys = plan.get('job_keys_this_shard')
        if keys is None:
            all_have_keys = False
            continue
        if not isinstance(keys, list) or len(keys) != int(plan['jobs_this_shard']):
            raise RuntimeError('network plan job key list does not match jobs_this_shard')
        planned_keys.extend((str(value) for value in keys))
    if all_have_keys and len(planned_keys) != len(set(planned_keys)):
        raise RuntimeError('network plans contain duplicate job keys across shards')
    planned_key_set = set(planned_keys) if all_have_keys else None
    missing = sorted(planned_key_set - completed_job_keys) if planned_key_set is not None else []
    extra = sorted(completed_job_keys - planned_key_set) if planned_key_set is not None else []
    complete = shard_indices == list(range(shard_count)) and completed_jobs == expected and (not missing) and (not extra)
    if not complete and (not allow_incomplete):
        raise RuntimeError(f'incomplete network-replacement result set: jobs={completed_jobs}/{expected}, shards={shard_indices}/{list(range(shard_count))}')
    return {'available': True, 'expected_jobs': expected, 'completed_jobs': completed_jobs, 'shard_count': shard_count, 'available_shard_indices': shard_indices, 'exact_job_key_set_available': planned_key_set is not None, 'missing_planned_job_key_count': len(missing), 'missing_planned_job_key_examples': missing[:10], 'extra_completed_job_key_count': len(extra), 'extra_completed_job_key_examples': extra[:10], 'complete': complete}

def flatten_results(results: Iterable[tuple[Path, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    reference_rows = []
    for path, payload in results:
        job = payload['job']
        common = {'source_cache_id': path.name, 'subject': job['subject'], 'seed_repeat': job['seed_repeat'], 'seed': job['seed'], 'backend': payload['backend'], 'cohort': job['cohort'], 'variant': job['variant'], 'preprocessing': payload.get('analysis_signature', {}).get('preprocessing', 'none'), 'runtime_seconds': payload.get('runtime_seconds')}
        for name, reference in payload['reference_models'].items():
            metric = reference.get('metric', {})
            reference_rows.append({**common, 'reference_model': name, 'accuracy': metric.get('accuracy'), 'balanced_accuracy': metric.get('balanced_accuracy'), **{f'recall_{task}': nested(metric, 'recall', task) for task in TASKS}})
        for row in payload['primary_perturbations']:
            metric = row['hybrid_metric']
            fit_rows.append({**common, 'row_kind': 'primary', 'target': row['target'], 'target_kind': row['target_kind'], 'model_mode': row['model_mode'], 'mask_size': row['mask_size'], 'random_mask_id': math.nan, 'hybrid_accuracy': metric['accuracy'], 'hybrid_balanced_accuracy': metric['balanced_accuracy'], **flatten_drop(row), **flatten_ood('input', row.get('input_covariance_ood')), **flatten_ood('embedding', row.get('embedding_ood'))})
        for interaction in payload['dmn_visual_interactions']:
            fit_rows.append({**common, 'row_kind': 'interaction', 'target': JOINT_TARGET, 'target_kind': 'interaction', 'model_mode': interaction['model_mode'], 'mask_size': math.nan, 'random_mask_id': math.nan, **flatten_interaction(interaction)})
        for row in payload['random_mask_nulls_seed0']:
            fixed_metric = row['fixed_hybrid_metric']
            fit_rows.append({**common, 'row_kind': 'fixed_random_null', 'target': row['target_size_matched_to'], 'target_kind': 'size_matched_random_roi', 'model_mode': 'fixed_post_model', 'mask_size': row['mask_size'], 'random_mask_id': row['random_mask_id'], 'hybrid_accuracy': fixed_metric['accuracy'], 'hybrid_balanced_accuracy': fixed_metric['balanced_accuracy'], **flatten_drop(row['fixed_drop']), **flatten_ood('input', row.get('input_covariance_ood')), **flatten_ood('embedding', row.get('fixed_embedding_ood'))})
    return (pd.DataFrame(fit_rows), pd.DataFrame(reference_rows))

def participant_metrics(fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = fit[fit.row_kind == 'primary']
    keys = ['subject', 'target', 'target_kind', 'model_mode', 'mask_size', 'preprocessing']
    for values, group in primary.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update({'row_kind': 'primary', 'seed_n': int(group.seed_repeat.nunique()), 'seed_repeats': json.dumps(sorted(group.seed_repeat.unique().tolist()))})
        for metric in DROP_METRICS:
            numeric = pd.to_numeric(group[metric], errors='coerce')
            row[metric] = float(numeric.median())
            row[f'{metric}_seed_mean'] = float(numeric.mean())
            row[f'{metric}_seed_sd'] = float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan
        rows.append(row)
    interactions = fit[fit.row_kind == 'interaction']
    keys = ['subject', 'target', 'target_kind', 'model_mode', 'preprocessing']
    for values, group in interactions.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update({'row_kind': 'interaction', 'mask_size': math.nan, 'seed_n': int(group.seed_repeat.nunique()), 'seed_repeats': json.dumps(sorted(group.seed_repeat.unique().tolist()))})
        for metric in INTERACTION_METRICS:
            numeric = pd.to_numeric(group[metric], errors='coerce')
            row[metric] = float(numeric.median())
            row[f'{metric}_seed_mean'] = float(numeric.mean())
            row[f'{metric}_seed_sd'] = float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)

def ood_participant_summary(fit: pd.DataFrame) -> pd.DataFrame:
    primary = fit[fit.row_kind == 'primary']
    keys = ['subject', 'target', 'target_kind', 'model_mode', 'mask_size', 'preprocessing']
    metric_columns = [column for column in primary.columns if column.startswith(('input_', 'embedding_'))]
    rows = []
    for values, group in primary.groupby(keys, dropna=False, sort=True):
        common = dict(zip(keys, values))
        for metric in metric_columns:
            numeric = pd.to_numeric(group[metric], errors='coerce').dropna()
            if numeric.empty:
                continue
            rows.append({**common, 'metric': metric, 'seed_n': int(group.seed_repeat.nunique()), 'participant_seed_median': float(numeric.median()), 'participant_seed_mean': float(numeric.mean()), 'participant_seed_sd': float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan})
    return pd.DataFrame(rows)

def ood_group_summary(participant_ood: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if participant_ood.empty:
        return pd.DataFrame(rows)
    keys = ['target', 'target_kind', 'model_mode', 'preprocessing', 'metric']
    for values, group in participant_ood.groupby(keys, dropna=False, sort=True):
        numeric = pd.to_numeric(group.participant_seed_median, errors='coerce').dropna().to_numpy()
        if not len(numeric):
            continue
        summary = one_sample_summary(numeric)
        rows.append({**dict(zip(keys, values)), 'n_participants': summary['n_participants'], 'mean': summary['mean'], 'sd': summary['sd'], 'median': summary['median'], 'ci95_low': summary['ci95_low'], 'ci95_high': summary['ci95_high'], 'minimum': float(np.min(numeric)), 'maximum': float(np.max(numeric)), 'inference_unit': 'participant'})
    return pd.DataFrame(rows)

def fixed_null_participant_summary(fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observed = fit[(fit.row_kind == 'primary') & (fit.model_mode == 'fixed_post_model') & (fit.seed_repeat == 0)]
    nulls = fit[fit.row_kind == 'fixed_random_null']
    for record in observed.to_dict('records'):
        local = nulls[(nulls.subject == record['subject']) & (nulls.target == record['target'])]
        if local.empty:
            continue
        for metric in DROP_METRICS:
            null_values = pd.to_numeric(local[metric], errors='coerce').dropna().to_numpy()
            observed_value = float(record[metric])
            rows.append({'subject': record['subject'], 'target': record['target'], 'preprocessing': record['preprocessing'], 'metric': metric, 'observed_fixed_drop_seed0': observed_value, 'random_mask_n': int(len(null_values)), 'random_mean': float(np.mean(null_values)), 'random_median': float(np.median(null_values)), 'random_sd': float(np.std(null_values, ddof=1)) if len(null_values) > 1 else math.nan, 'observed_minus_random_mean': float(observed_value - np.mean(null_values)), 'empirical_p_random_greater_equal': float((1 + np.sum(null_values >= observed_value)) / (len(null_values) + 1)), 'signed_not_clipped': True})
    return pd.DataFrame(rows)

def bh(values: pd.Series) -> pd.Series:
    array = pd.to_numeric(values, errors='coerce').to_numpy(float)
    result = np.full(len(array), np.nan)
    valid = np.flatnonzero(np.isfinite(array))
    if not len(valid):
        return pd.Series(result, index=values.index)
    order = valid[np.argsort(array[valid])]
    adjusted = array[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)

def one_sample_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    result = {'n_participants': int(n), 'mean': float(np.mean(values)) if n else math.nan, 'sd': float(np.std(values, ddof=1)) if n > 1 else math.nan, 'median': float(np.median(values)) if n else math.nan, 'ci95_low': math.nan, 'ci95_high': math.nan, 't_vs_zero': math.nan, 'p_t_two_sided': math.nan, 'wilcoxon_statistic': math.nan, 'p_wilcoxon_two_sided': math.nan}
    if n > 1:
        sem = stats.sem(values)
        low, high = stats.t.interval(0.95, n - 1, loc=np.mean(values), scale=sem)
        t_result = stats.ttest_1samp(values, 0)
        result.update({'ci95_low': float(low), 'ci95_high': float(high), 't_vs_zero': float(t_result.statistic), 'p_t_two_sided': float(t_result.pvalue)})
        if np.any(values != 0):
            w_result = stats.wilcoxon(values, alternative='two-sided')
            result['wilcoxon_statistic'] = float(w_result.statistic)
            result['p_wilcoxon_two_sided'] = float(w_result.pvalue)
    return result

def group_summary(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record_kind, metrics in (('primary', DROP_METRICS), ('interaction', INTERACTION_METRICS)):
        local = participant[participant.row_kind == record_kind]
        for keys, group in local.groupby(['target', 'target_kind', 'model_mode', 'preprocessing'], dropna=False, sort=True):
            for metric in metrics:
                rows.append({'row_kind': record_kind, 'target': keys[0], 'target_kind': keys[1], 'model_mode': keys[2], 'preprocessing': keys[3], 'metric': metric, **one_sample_summary(pd.to_numeric(group[metric], errors='coerce').to_numpy()), 'inference_unit': 'participant', 'signed_not_clipped': True, 'additivity_not_imposed': True})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['q_t_bh_all_network_replacement_tests'] = bh(frame.p_t_two_sided)
        frame['q_wilcoxon_bh_all_network_replacement_tests'] = bh(frame.p_wilcoxon_two_sided)
    return frame

def null_group_summary(fixed_null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not fixed_null.empty:
        for keys, group in fixed_null.groupby(['target', 'preprocessing', 'metric'], sort=True):
            rows.append({'null_kind': 'fixed_model_20plus_random_masks', 'target': keys[0], 'preprocessing': keys[1], 'metric': keys[2], **one_sample_summary(group.observed_minus_random_mean.to_numpy()), 'inference_unit': 'participant'})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['q_t_bh_within_null_summary'] = bh(frame.p_t_two_sided)
        frame['q_wilcoxon_bh_within_null_summary'] = bh(frame.p_wilcoxon_two_sided)
    return frame

def summarize(args) -> dict:
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(input_dir)
    completed_job_keys = {str(payload.get('job_key') or path.stem) for path, payload in results}
    if len(completed_job_keys) != len(results):
        raise RuntimeError('duplicate completed network-replacement job keys')
    reconciliation = reconcile_plans(input_dir, completed_job_keys, args.allow_incomplete)
    fit, references = flatten_results(results)
    participant = participant_metrics(fit)
    participant_ood = ood_participant_summary(fit)
    fixed_null = fixed_null_participant_summary(fit)
    group = group_summary(participant)
    group_ood = ood_group_summary(participant_ood)
    null_group = null_group_summary(fixed_null)
    write_frame(fit, output_dir / 'cebra_network_replacement_fit_metrics')
    write_frame(references, output_dir / 'cebra_network_replacement_reference_metrics')
    write_frame(participant, output_dir / 'cebra_network_replacement_participant_metrics')
    write_frame(group, output_dir / 'cebra_network_replacement_group_summary')
    write_frame(participant_ood, output_dir / 'cebra_network_replacement_ood_participant_summary')
    write_frame(group_ood, output_dir / 'cebra_network_replacement_ood_group_summary')
    write_frame(fixed_null, output_dir / 'cebra_network_replacement_fixed_null')
    write_frame(null_group, output_dir / 'cebra_network_replacement_null_group_summary')
    summary = {'schema': ANALYSIS_SCHEMA, 'completed_jobs': len(results), 'plan_reconciliation': reconciliation, 'participants': int(participant.subject.nunique()), 'fit_metric_rows': len(fit), 'participant_metric_rows': len(participant), 'group_summary_rows': len(group), 'ood_participant_rows': len(participant_ood), 'ood_group_rows': len(group_ood), 'fixed_null_participant_rows': len(fixed_null), 'fixed_null_group_rows': len(null_group)}
    atomic_text(output_dir / 'cebra_network_replacement_summary.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    summary = summarize(args)
    print(json.dumps(summary))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
