from __future__ import annotations
import argparse
import json
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ACTUAL_READOUT = 'four_contexts'
NULL_READOUT = 'circular_block_shift_null_four_contexts'
SESSIONS = ('ses-01', 'ses-02')
METRICS = ('accuracy', 'balanced_accuracy')
VALUE_TYPES = ('actual', 'null', 'excess')

def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)

def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})

def load_fit(stage_dir: Path) -> pd.DataFrame:
    summary = stage_dir / 'summary'
    parquet = summary / 'cebra_fit_metrics.parquet'
    csv = summary / 'cebra_fit_metrics.csv'
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(csv)

def participant_scores(fit: pd.DataFrame, gap_frames: int) -> pd.DataFrame:
    seed = pd.to_numeric(fit['seed_repeat'], errors='coerce')
    actual = fit[fit.readout.eq(ACTUAL_READOUT) & seed.eq(0)].copy()
    null_all = fit[fit.readout.eq(NULL_READOUT) & seed.eq(0)].copy()
    null = null_all[truthy(null_all.test_contains_all_declared_classes)].copy()
    keys = ['subject', 'session', 'fold']
    cells = []
    for metric in METRICS:
        actual[metric] = pd.to_numeric(actual[metric], errors='coerce')
        null[metric] = pd.to_numeric(null[metric], errors='coerce')
        actual_metric = actual[np.isfinite(actual[metric])]
        null_metric = null[np.isfinite(null[metric])]
        actual_fold = actual_metric.groupby(keys, sort=True)[metric].mean()
        null_fold = null_metric.groupby(keys, sort=True)[metric].mean()
        observed = actual_fold.groupby(level=[0, 1], sort=True).mean().rename('actual')
        expected = null_fold.groupby(level=[0, 1], sort=True).mean().rename('null')
        frame = pd.concat([observed, expected], axis=1).reset_index()
        frame['excess'] = frame.actual - frame.null
        frame['metric'] = metric
        frame['gap_frames'] = gap_frames
        actual_fold_n = actual_fold.groupby(level=[0, 1]).size().rename('actual_fold_n').reset_index()
        valid_counts = null_metric.groupby(keys, sort=True).size()
        null_audit = valid_counts.groupby(level=[0, 1]).agg(null_valid_row_n='sum', null_fold_n='size', null_valid_row_min_per_fold='min', null_valid_row_max_per_fold='max').reset_index()
        total_counts = null_all.groupby(['subject', 'session'], sort=True).size().rename('null_total_row_n').reset_index()
        frame = frame.merge(actual_fold_n, on=['subject', 'session'])
        frame = frame.merge(null_audit, on=['subject', 'session'])
        frame = frame.merge(total_counts, on=['subject', 'session'])
        frame['null_excluded_row_n'] = frame.null_total_row_n - frame.null_valid_row_n
        cells.append(frame)
    return pd.concat(cells, ignore_index=True)

def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return (float(low), float(high))

def signflip_p(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    exceedances = 0
    for start in range(0, draws, 10000):
        count = min(10000, draws - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
    return float((exceedances + 1) / (draws + 1))

def inference_row(values: pd.Series, *, bootstrap_draws: int, signflip_draws: int, seed: int) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors='coerce').to_numpy(float)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(seed)
    ci_low, ci_high = bootstrap_ci(array, bootstrap_draws, rng)
    ttest = stats.ttest_1samp(array, 0.0)
    try:
        wilcoxon = stats.wilcoxon(array, alternative='two-sided')
        wilcoxon_stat = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    return {'participant_n': len(array), 'mean': float(array.mean()), 'sd': float(array.std(ddof=1)), 'median': float(np.median(array)), 'ci95_low': ci_low, 'ci95_high': ci_high, 'positive_n': int(np.count_nonzero(array > 0)), 'negative_n': int(np.count_nonzero(array < 0)), 'zero_n': int(np.count_nonzero(array == 0)), 'one_sample_t': float(ttest.statistic), 'one_sample_t_p_two_sided': float(ttest.pvalue), 'wilcoxon_statistic': wilcoxon_stat, 'wilcoxon_p_two_sided': wilcoxon_p, 'signflip_p_two_sided': signflip_p(array, signflip_draws, rng)}

def bh_adjust(series: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=series.index, dtype=float)
    finite = pd.to_numeric(series, errors='coerce').dropna().sort_values()
    if finite.empty:
        return output
    adjusted = finite.to_numpy(float) * len(finite) / np.arange(1, len(finite) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[finite.index] = np.clip(adjusted, 0.0, 1.0)
    return output

def add_family_bh(frame: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    result = frame.copy()
    p_columns = ('one_sample_t_p_two_sided', 'wilcoxon_p_two_sided', 'signflip_p_two_sided')
    for p_column in p_columns:
        q_column = p_column.replace('_p_', '_q_bh_family_')
        result[q_column] = result.groupby(family, sort=False)[p_column].transform(bh_adjust)
    return result

def group_summary(participant: pd.DataFrame, bootstrap_draws: int, signflip_draws: int, seed: int) -> pd.DataFrame:
    rows = []
    row_index = 0
    for (gap, metric), local in participant.groupby(['gap_frames', 'metric'], sort=True):
        for value_type in VALUE_TYPES:
            wide = local.pivot(index='subject', columns='session', values=value_type)
            for contrast, values in (('ses-01', wide['ses-01']), ('ses-02', wide['ses-02']), ('ses-02_minus_ses-01', wide['ses-02'] - wide['ses-01'])):
                row = {'gap_frames': int(gap), 'metric': metric, 'value_type': value_type, 'contrast': contrast, 'inference_unit': 'participant', 'seed_repeat': 0}
                row.update(inference_row(values, bootstrap_draws=bootstrap_draws, signflip_draws=signflip_draws, seed=seed + row_index * 1009))
                rows.append(row)
                row_index += 1
    return add_family_bh(pd.DataFrame(rows), family=['metric', 'value_type', 'contrast'])

def gap_comparisons(participant: pd.DataFrame, bootstrap_draws: int, signflip_draws: int, seed: int) -> pd.DataFrame:
    rows = []
    row_index = 0
    gaps = sorted((int(value) for value in participant.gap_frames.unique()))
    for gap_a, gap_b in combinations(gaps, 2):
        for metric in METRICS:
            local = participant[participant.metric.eq(metric)]
            for value_type in VALUE_TYPES:
                wide = local.pivot(index='subject', columns=['gap_frames', 'session'], values=value_type)
                for contrast, values in (('ses-01', wide[gap_a, 'ses-01'] - wide[gap_b, 'ses-01']), ('ses-02', wide[gap_a, 'ses-02'] - wide[gap_b, 'ses-02']), ('session_delta_difference_in_differences', wide[gap_a, 'ses-02'] - wide[gap_a, 'ses-01'] - (wide[gap_b, 'ses-02'] - wide[gap_b, 'ses-01']))):
                    row = {'gap_a_frames': gap_a, 'gap_b_frames': gap_b, 'comparison': f'gap{gap_a}_minus_gap{gap_b}', 'metric': metric, 'value_type': value_type, 'contrast': contrast, 'inference_unit': 'participant_paired_across_gaps'}
                    row.update(inference_row(values.dropna(), bootstrap_draws=bootstrap_draws, signflip_draws=signflip_draws, seed=seed + 100000 + row_index * 1009))
                    rows.append(row)
                    row_index += 1
    return add_family_bh(pd.DataFrame(rows), family=['metric', 'value_type', 'contrast'])

def integrity_row(fit: pd.DataFrame, participant: pd.DataFrame, gap_frames: int) -> dict[str, object]:
    seed = pd.to_numeric(fit.seed_repeat, errors='coerce')
    actual = fit[fit.readout.eq(ACTUAL_READOUT) & seed.eq(0)]
    null = fit[fit.readout.eq(NULL_READOUT) & seed.eq(0)]
    complete = null[truthy(null.test_contains_all_declared_classes)]
    finite = np.isfinite(complete.loc[:, list(METRICS)].apply(pd.to_numeric, errors='coerce')).all(axis=1)
    status_ok = truthy(fit.production_result).all() if 'production_result' in fit else True
    completed = fit.status.eq('completed').all() if 'status' in fit else True
    zero_overlap = truthy(fit.split_audit_train_test_gap_zero_overlap).all() if 'split_audit_train_test_gap_zero_overlap' in fit else True
    return {'gap_frames': gap_frames, 'fit_metric_rows': len(fit), 'subjects': fit.subject.nunique(), 'sessions': fit.session.nunique(), 'jobs': fit.job_key.nunique() if 'job_key' in fit else len(actual), 'actual_seed0_rows': len(actual), 'null_seed0_rows_total': len(null), 'null_seed0_rows_complete_class': len(complete), 'null_seed0_rows_excluded_missing_class': len(null) - len(complete), 'null_complete_rows_finite_both_metrics': int(finite.sum()), 'all_status_completed': bool(completed), 'all_production_result': bool(status_ok), 'all_train_test_gap_zero_overlap': bool(zero_overlap), 'participant_metric_rows': len(participant), 'actual_fold_n_min': int(participant.actual_fold_n.min()), 'actual_fold_n_max': int(participant.actual_fold_n.max()), 'null_fold_n_min': int(participant.null_fold_n.min()), 'null_fold_n_max': int(participant.null_fold_n.max()), 'null_valid_row_min_per_fold': int(participant.null_valid_row_min_per_fold.min()), 'null_valid_row_max_per_fold': int(participant.null_valid_row_max_per_fold.max())}

def parse_stage(value: str) -> tuple[int, Path]:
    gap, path = value.split('=', 1)
    return (int(gap), Path(path))

def run_analysis(stages: list[tuple[int, Path]], output_dir: Path, *, bootstrap_draws: int=10000, signflip_draws: int=100000, seed: int=10910) -> dict[str, object]:
    participant_frames = []
    integrity_rows = []
    for gap_frames, stage_dir in sorted(stages):
        fit = load_fit(stage_dir)
        participant = participant_scores(fit, gap_frames)
        participant_frames.append(participant)
        integrity_rows.append(integrity_row(fit, participant, gap_frames))
    participants = pd.concat(participant_frames, ignore_index=True)
    integrity = pd.DataFrame(integrity_rows)
    group = group_summary(participants, bootstrap_draws, signflip_draws, seed)
    comparisons = gap_comparisons(participants, bootstrap_draws, signflip_draws, seed)
    atomic_csv(output_dir / 'cebra_gap_sensitivity_participants.csv', participants)
    atomic_csv(output_dir / 'cebra_gap_sensitivity_integrity.csv', integrity)
    atomic_csv(output_dir / 'cebra_gap_sensitivity_group.csv', group)
    atomic_csv(output_dir / 'cebra_gap_sensitivity_gap_comparisons.csv', comparisons)
    metadata = {'schema': 'cebra-purge-gap-temporal-null-v1', 'gap_frames': sorted((int(gap) for gap, _ in stages)), 'metrics': list(METRICS), 'value_types': list(VALUE_TYPES), 'seed_repeat': 0, 'complete_class_null_only': True, 'fold_equal_weighting': True, 'bootstrap_draws': bootstrap_draws, 'signflip_draws': signflip_draws, 'participant_rows': len(participants), 'group_rows': len(group), 'gap_comparison_rows': len(comparisons)}
    atomic_text(output_dir / 'cebra_gap_sensitivity.json', json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', action='append', required=True, metavar='GAP=BLOCKED_DIR', help='repeat for every purge-gap blocked result')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--signflip-draws', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=10910)
    args = parser.parse_args()
    metadata = run_analysis([parse_stage(value) for value in args.stage], args.output_dir, bootstrap_draws=args.bootstrap_draws, signflip_draws=args.signflip_draws, seed=args.seed)
    print(json.dumps(metadata, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
