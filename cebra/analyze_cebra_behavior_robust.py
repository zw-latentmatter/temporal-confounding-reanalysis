from __future__ import annotations
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
BRANCHES = (('author_random', 'author_like_raw', 'author_like', 'random_stratified_frames'), ('author_resub', 'author_like_raw', 'author_like', 'resubstitution'), ('blocked_gap50', 'blocked_zscore_gap50', 'blocked', 'four_contexts'), ('cross_session', 'cross_session_zscore', 'cross_session', 'four_contexts'), ('participant_heldout', 'participant_heldout_zscore', 'participant_heldout', 'participant_heldout_shared_four_contexts'))
METRICS = ('accuracy', 'balanced_accuracy', 'silhouette_mean')
OUTCOMES = ('MEQ30_MEAN', 'MINDSET_AVAILABLE_MEAN')

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)

def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({'true', '1', 'yes'})

def load_metric_series(primary_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for branch, directory, analysis, readout in BRANCHES:
        path = primary_root / directory / 'summary' / 'cebra_participant_metrics.csv'
        frame = pd.read_csv(path)
        frame = frame[frame.analysis.eq(analysis) & frame.variant.eq('author_literal') & frame.readout.eq(readout)].copy()
        if analysis == 'cross_session':
            for record in frame.to_dict('records'):
                for metric in METRICS:
                    value = pd.to_numeric(record.get(metric), errors='coerce')
                    if np.isfinite(value):
                        direction = record.get('direction')
                        rows.append({'subject': record['subject'], 'metric_id': f'{branch}__{direction}__{metric}', 'branch': branch, 'analysis': analysis, 'readout': readout, 'scope': str(direction), 'metric': metric, 'metric_value': float(value), 'baseline_metric': math.nan})
            continue
        for metric in METRICS:
            local = frame[['subject', 'session', metric]].copy()
            local[metric] = pd.to_numeric(local[metric], errors='coerce')
            wide = local.pivot_table(index='subject', columns='session', values=metric)
            for subject, record in wide.iterrows():
                baseline = record.get('ses-01', math.nan)
                post = record.get('ses-02', math.nan)
                if np.isfinite(post):
                    rows.append({'subject': subject, 'metric_id': f'{branch}__post__{metric}', 'branch': branch, 'analysis': analysis, 'readout': readout, 'scope': 'ses-02', 'metric': metric, 'metric_value': float(post), 'baseline_metric': float(baseline)})
                if np.isfinite(post) and np.isfinite(baseline):
                    rows.append({'subject': subject, 'metric_id': f'{branch}__delta__{metric}', 'branch': branch, 'analysis': analysis, 'readout': readout, 'scope': 'ses-02_minus_ses-01', 'metric': metric, 'metric_value': float(post - baseline), 'baseline_metric': float(baseline)})
    return pd.DataFrame(rows)

def rank_columns(values: np.ndarray) -> np.ndarray:
    return np.column_stack([stats.rankdata(values[:, index]) for index in range(values.shape[1])])

def residual_pair(x: np.ndarray, y: np.ndarray, covariates: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == 'spearman':
        combined = np.column_stack([x, y, covariates])
        ranked = rank_columns(combined)
        x, y, covariates = (ranked[:, 0], ranked[:, 1], ranked[:, 2:])
    elif method != 'pearson':
        raise ValueError(method)
    if covariates.shape[1]:
        design = np.column_stack([np.ones(len(x)), covariates])
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x = x - x.mean()
    y = y - y.mean()
    return (x, y)

def correlation_from_residuals(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else math.nan

def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, draws: int, rng: np.random.Generator) -> float:
    exceedances = 0
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    for start in range(0, draws, 5000):
        count = min(5000, draws - start)
        permutations = np.argsort(rng.random((count, len(y))), axis=1)
        null = y[permutations] @ x / denominator
        exceedances += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    return float((exceedances + 1) / (draws + 1))

def bootstrap_ci(x: np.ndarray, y: np.ndarray, covariates: np.ndarray, method: str, draws: int, rng: np.random.Generator) -> tuple[float, float, int]:
    estimates = np.empty(draws, dtype=np.float64)
    valid = 0
    for _ in range(draws):
        sample = rng.integers(0, len(x), len(x))
        residual_x, residual_y = residual_pair(x[sample], y[sample], covariates[sample], method)
        estimate = correlation_from_residuals(residual_x, residual_y)
        if np.isfinite(estimate):
            estimates[valid] = estimate
            valid += 1
    if valid < max(100, draws // 2):
        return (math.nan, math.nan, valid)
    low, high = np.quantile(estimates[:valid], [0.025, 0.975])
    return (float(low), float(high), valid)

def association_worker(arguments: tuple[dict, int, int, int]) -> dict:
    spec, permutation_draws, bootstrap_draws, seed = arguments
    frame = pd.DataFrame(spec.pop('records'))
    covariate_names = spec['covariates']
    columns = ['metric_value', 'outcome_value', *covariate_names]
    numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
    keep = np.isfinite(numeric).all(axis=1)
    numeric = numeric.loc[keep]
    spec['n'] = int(len(numeric))
    spec.update({'estimate': math.nan, 'ci95_low': math.nan, 'ci95_high': math.nan, 'permutation_p': math.nan, 'bootstrap_valid_draws': 0, 'status': 'ok'})
    if len(numeric) < max(8, len(covariate_names) + 4):
        spec['status'] = 'insufficient_n'
        return spec
    x = numeric.metric_value.to_numpy(float)
    y = numeric.outcome_value.to_numpy(float)
    covariates = numeric[covariate_names].to_numpy(float) if covariate_names else np.empty((len(numeric), 0), dtype=float)
    residual_x, residual_y = residual_pair(x, y, covariates, spec['method'])
    estimate = correlation_from_residuals(residual_x, residual_y)
    if not np.isfinite(estimate):
        spec['status'] = 'constant_input'
        return spec
    rng = np.random.default_rng(seed)
    spec['estimate'] = estimate
    spec['permutation_p'] = permutation_p(residual_x, residual_y, estimate, permutation_draws, rng)
    low, high, valid = bootstrap_ci(x, y, covariates, spec['method'], bootstrap_draws, rng)
    spec['ci95_low'] = low
    spec['ci95_high'] = high
    spec['bootstrap_valid_draws'] = valid
    return spec

def bh_adjust(series: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=series.index, dtype=float)
    finite = pd.to_numeric(series, errors='coerce').dropna().sort_values()
    if finite.empty:
        return output
    adjusted = finite.to_numpy(float) * len(finite) / np.arange(1, len(finite) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[finite.index] = np.clip(adjusted, 0.0, 1.0)
    return output

def build_specs(metrics: pd.DataFrame, membership: pd.DataFrame, behavior: pd.DataFrame) -> list[dict]:
    merged = metrics.merge(membership, on='subject', how='left').merge(behavior, on='subject', how='left')
    cohorts = {'modal_length_n57': truthy(merged['modal_length_both_sessions']), 'paper_size_low_fd_proxy_n54': truthy(merged['paper_size_low_fd_proxy'])}
    specs: list[dict] = []
    for cohort, cohort_mask in cohorts.items():
        cohort_frame = merged[cohort_mask]
        for metric_id, local_metric in cohort_frame.groupby('metric_id', sort=True):
            descriptor = local_metric.iloc[0]
            for outcome in OUTCOMES:
                outcome_frame = local_metric.copy()
                outcome_frame['outcome_value'] = pd.to_numeric(outcome_frame[outcome], errors='coerce')
                adjustments: list[tuple[str, list[str]]] = [('unadjusted', [])]
                has_baseline = np.isfinite(pd.to_numeric(outcome_frame.baseline_metric, errors='coerce')).sum() >= 8
                if has_baseline:
                    adjustments.append(('baseline_performance_and_post_motion', ['baseline_metric', 'psilocybin_mean_fd_across_tasks']))
                if outcome == 'MINDSET_AVAILABLE_MEAN':
                    adjustments.append(('MEQ30', ['MEQ30_MEAN']))
                    if has_baseline:
                        adjustments.append(('MEQ30_baseline_performance_and_post_motion', ['MEQ30_MEAN', 'baseline_metric', 'psilocybin_mean_fd_across_tasks']))
                for adjustment, covariates in adjustments:
                    for method in ('pearson', 'spearman'):
                        specs.append({'cohort': cohort, 'metric_id': metric_id, 'branch': descriptor.branch, 'analysis': descriptor.analysis, 'readout': descriptor.readout, 'scope': descriptor.scope, 'metric': descriptor.metric, 'outcome': outcome, 'adjustment': adjustment, 'covariates': covariates, 'method': method, 'records': outcome_frame[['subject', 'metric_value', 'outcome_value', *covariates]].to_dict('records')})
    return specs

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-root', required=True, type=Path)
    parser.add_argument('--membership', required=True, type=Path)
    parser.add_argument('--behavior', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--permutation-draws', type=int, default=100000)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10910)
    args = parser.parse_args()
    metrics = load_metric_series(args.primary_root)
    membership = pd.read_csv(args.membership)
    behavior = pd.read_csv(args.behavior).rename(columns={'participant_id': 'subject'})
    specs = build_specs(metrics, membership, behavior)
    tasks = [(spec, args.permutation_draws, args.bootstrap_draws, args.seed + index * 1009) for index, spec in enumerate(specs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(association_worker, tasks))
    result = pd.DataFrame(rows)
    result['covariates'] = result.covariates.map(lambda values: '|'.join(values))
    result['q_bh_within_cohort_method'] = math.nan
    for (_, method), index in result.groupby(['cohort', 'method']).groups.items():
        result.loc[index, 'q_bh_within_cohort_method'] = bh_adjust(result.loc[index, 'permutation_p'])
    result['q_bh_all_gpu_behavior_tests'] = bh_adjust(result['permutation_p'])
    result['inference_unit'] = 'participant'
    metadata = {'schema': 'cebra-behavior-robust-v2', 'permutation_draws': args.permutation_draws, 'bootstrap_draws': args.bootstrap_draws, 'workers': args.workers, 'seed': args.seed, 'metric_series_count': int(metrics.metric_id.nunique()), 'test_row_count': len(result), 'finite_test_count': int(np.isfinite(result.permutation_p).sum()), 'outcomes': list(OUTCOMES)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / 'cebra_behavior_robust.csv', result)
    atomic_text(args.output_dir / 'cebra_behavior_robust.json', json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metadata, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
