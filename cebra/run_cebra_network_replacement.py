from __future__ import annotations
import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from run_cebra_jobs import CebraConfig, TASKS, atomic_json, atomic_npz, fit_encoder, linear_svm, load_session, numeric_subject, safe_transform, segment_centers, set_all_seeds
NETWORK_ORDER = ('subcortical', 'visual', 'somatomotor', 'dorsal_attention', 'limbic', 'salience_ventral_attention', 'default_mode', 'control')
JOINT_TARGET = 'default_mode_plus_visual'
ANALYSIS_SCHEMA = 'cebra-network-baseline-substitution-v1'
NETWORK_PATTERNS = (('visual', re.compile('(?:^|_)vis(?:_|$)', re.IGNORECASE)), ('somatomotor', re.compile('(?:^|_)sommot(?:_|$)', re.IGNORECASE)), ('dorsal_attention', re.compile('(?:^|_)dorsattn(?:_|$)', re.IGNORECASE)), ('limbic', re.compile('(?:^|_)limbic(?:_|$)', re.IGNORECASE)), ('salience_ventral_attention', re.compile('(?:^|_)salventattn(?:_|$)', re.IGNORECASE)), ('default_mode', re.compile('(?:^|_)default(?:_|$)', re.IGNORECASE)), ('control', re.compile('(?:^|_)cont(?:_|$)', re.IGNORECASE)))

def read_roi_labels(path: Path) -> list[str]:
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, Mapping):
            payload = payload.get('labels')
        labels = [str(value) for value in payload]
    else:
        lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        labels = lines[0::2] if len(lines) == 664 else lines
    if len(labels) != 332:
        raise RuntimeError(f'expected 332 atlas labels, found {len(labels)} in {path}')
    return labels

def parse_network_indices(labels: Sequence[str]) -> dict[str, np.ndarray]:
    if len(labels) != 332:
        raise ValueError(f'expected 332 labels, got {len(labels)}')
    assignments = ['subcortical'] * 32
    for roi_index, label in enumerate(labels[32:], start=32):
        matches = [name for name, pattern in NETWORK_PATTERNS if pattern.search(label)]
        if len(matches) != 1:
            raise RuntimeError(f'cortical ROI {roi_index} label maps to {len(matches)} networks: {label!r}')
        assignments.append(matches[0])
    result = {network: np.flatnonzero(np.asarray(assignments) == network).astype(np.int16) for network in NETWORK_ORDER}
    if len(result['subcortical']) != 32:
        raise RuntimeError('the first 32 ROI indices must be subcortical')
    if sum((len(result[name]) for name in NETWORK_ORDER[1:])) != 300:
        raise RuntimeError('the Schaefer seven-network mapping does not cover exactly 300 ROIs')
    if any((len(result[name]) == 0 for name in NETWORK_ORDER)):
        raise RuntimeError('at least one atlas network has no ROI')
    return result

def target_masks(networks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {name: np.asarray(networks[name], dtype=np.int16) for name in NETWORK_ORDER}
    result[JOINT_TARGET] = np.sort(np.concatenate([result['default_mode'], result['visual']])).astype(np.int16)
    return result

def fixed_random_masks(targets: Mapping[str, np.ndarray], count: int, seed: int) -> dict[str, list[np.ndarray]]:
    masks: dict[str, list[np.ndarray]] = {}
    for target_index, target in enumerate(targets):
        rng = np.random.default_rng(seed + (target_index + 1) * 100003)
        masks[target] = [np.sort(rng.choice(332, size=len(targets[target]), replace=False)).astype(np.int16) for _ in range(count)]
    return masks

def verify_paired_sessions(post: Mapping, baseline: Mapping) -> None:
    if post['run_slices'] != baseline['run_slices']:
        raise RuntimeError('baseline and post task lengths differ; exact task/frame substitution is impossible')
    if post['x'].shape != baseline['x'].shape:
        raise RuntimeError('baseline and post arrays have different shapes')
    if not np.array_equal(post['y'], baseline['y']):
        raise RuntimeError('baseline and post task labels are not frame-aligned')

def preprocess_sessions(post: np.ndarray, baseline: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if mode == 'none':
        return (np.asarray(post, dtype=np.float32), np.asarray(baseline, dtype=np.float32), {'mode': 'none', 'separately_fit_by_session': False})
    if mode != 'separate_session_zscore':
        raise ValueError(mode)
    outputs = []
    audits = {}
    for name, values in (('post', post), ('baseline', baseline)):
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        zero_scale = scale <= 1e-12
        safe_scale = scale.copy()
        safe_scale[zero_scale] = 1.0
        transformed = ((values - mean) / safe_scale).astype(np.float32)
        outputs.append(transformed)
        audits[name] = {'fit_frames': int(len(values)), 'roi_count': int(values.shape[1]), 'zero_scale_roi_count': int(zero_scale.sum()), 'pre_mean_abs_max': float(np.max(np.abs(mean))), 'pre_scale_min_nonzero': float(np.min(scale[~zero_scale])) if np.any(~zero_scale) else math.nan, 'post_mean_abs_max': float(np.max(np.abs(transformed.mean(axis=0)))), 'post_scale_abs_error_max': float(np.max(np.abs(transformed.std(axis=0, ddof=0)[~zero_scale] - 1.0))) if np.any(~zero_scale) else math.nan}
    return (outputs[0], outputs[1], {'mode': mode, 'separately_fit_by_session': True, 'post': audits['post'], 'baseline': audits['baseline']})

def replace_rois_by_task(post: np.ndarray, baseline: np.ndarray, post_run_slices: Sequence[tuple[int, int]], baseline_run_slices: Sequence[tuple[int, int]], roi_indices: Sequence[int]) -> tuple[np.ndarray, dict]:
    if post.shape[1] != 332 or baseline.shape[1] != 332:
        raise ValueError('network substitution expects 332 ROI columns')
    roi_indices = np.unique(np.asarray(roi_indices, dtype=np.int64))
    if len(roi_indices) == 0 or roi_indices[0] < 0 or roi_indices[-1] >= 332:
        raise ValueError('replacement ROI indices are empty or out of bounds')
    if len(post_run_slices) != len(baseline_run_slices):
        raise RuntimeError('baseline and post have different task counts')
    hybrid = np.array(post, dtype=np.float32, copy=True)
    per_task = []
    for task, post_bounds, baseline_bounds in zip(TASKS, post_run_slices, baseline_run_slices):
        post_start, post_stop = post_bounds
        base_start, base_stop = baseline_bounds
        post_length = post_stop - post_start
        baseline_length = base_stop - base_start
        if post_length != baseline_length:
            raise RuntimeError(f'task {task} has {post_length} post frames and {baseline_length} baseline frames')
        hybrid[post_start:post_stop, roi_indices] = baseline[base_start:base_stop, roi_indices]
        per_task.append({'task': task, 'post_half_open': [int(post_start), int(post_stop)], 'baseline_half_open': [int(base_start), int(base_stop)], 'within_task_frame_index_start': 0, 'within_task_frame_index_stop_exclusive': int(post_length)})
    return (hybrid, {'roi_indices': roi_indices.tolist(), 'roi_count': int(len(roi_indices)), 'task_frame_mapping': per_task, 'replaced_cell_count': int(sum((stop - start for start, stop in post_run_slices)) * len(roi_indices))})

def stratified_ood_indices(run_slices: Sequence[tuple[int, int]], max_frames: int) -> np.ndarray:
    per_run = max(2, max_frames // len(run_slices))
    parts = []
    for start, stop in run_slices:
        take = min(per_run, stop - start)
        parts.append(np.linspace(start, stop - 1, num=take, dtype=np.int64))
    return np.unique(np.concatenate(parts))

@dataclass
class InputCovarianceReference:
    post: np.ndarray
    baseline: np.ndarray
    sample_indices: np.ndarray

    def __post_init__(self):
        self.post_sample = np.asarray(self.post[self.sample_indices], dtype=np.float64)
        self.baseline_sample = np.asarray(self.baseline[self.sample_indices], dtype=np.float64)
        self.post_centered = self.post_sample - self.post_sample.mean(axis=0, keepdims=True)
        self.baseline_centered = self.baseline_sample - self.baseline_sample.mean(axis=0, keepdims=True)
        denominator = max(1, len(self.sample_indices) - 1)
        self.post_covariance = self.post_centered.T @ self.post_centered / denominator
        self.baseline_covariance = self.baseline_centered.T @ self.baseline_centered / denominator

    def metrics(self, mask: Sequence[int]) -> dict:
        mask = np.asarray(mask, dtype=np.int64)
        retained = np.setdiff1d(np.arange(332), mask, assume_unique=False)
        denominator = max(1, len(self.sample_indices) - 1)
        mixed_cross = self.baseline_centered[:, mask].T @ self.post_centered[:, retained] / denominator
        hybrid_covariance = self.post_covariance.copy()
        hybrid_covariance[np.ix_(mask, mask)] = self.baseline_covariance[np.ix_(mask, mask)]
        hybrid_covariance[np.ix_(mask, retained)] = mixed_cross
        hybrid_covariance[np.ix_(retained, mask)] = mixed_cross.T
        post_cross = self.post_covariance[np.ix_(mask, retained)]
        baseline_cross = self.baseline_covariance[np.ix_(mask, retained)]
        difference = self.baseline_sample[:, mask] - self.post_sample[:, mask]
        whole_rmse = math.sqrt(float(np.square(difference).sum()) / (len(difference) * 332))

        def relative(left: np.ndarray, right: np.ndarray) -> float:
            return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12))
        return {'ood_sample_frames': int(len(self.sample_indices)), 'input_rmse_replaced_rois_from_post': float(np.sqrt(np.mean(np.square(difference)))), 'input_rmse_whole_brain_from_post': whole_rmse, 'input_mean_shift_l2': float(np.linalg.norm(self.baseline_sample[:, mask].mean(axis=0) - self.post_sample[:, mask].mean(axis=0))), 'hybrid_covariance_relative_frobenius_to_post': relative(hybrid_covariance, self.post_covariance), 'hybrid_covariance_relative_frobenius_to_baseline': relative(hybrid_covariance, self.baseline_covariance), 'mixed_replaced_retained_cross_covariance_relative_to_post': relative(mixed_cross, post_cross), 'mixed_replaced_retained_cross_covariance_relative_to_baseline': relative(mixed_cross, baseline_cross)}

def embedding_ood_metrics(reference: np.ndarray, query: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    mean = reference.mean(axis=0)
    centered = reference - mean
    covariance = np.cov(reference, rowvar=False)
    inverse = np.linalg.pinv(covariance)
    query_centered = query - mean
    mahalanobis_squared = np.einsum('ni,ij,nj->n', query_centered, inverse, query_centered)
    reference_covariance = np.cov(reference, rowvar=False)
    query_covariance = np.cov(query, rowvar=False)
    nearest = NearestNeighbors(n_neighbors=2).fit(reference)
    self_distances = nearest.kneighbors(reference, return_distance=True)[0][:, 1]
    query_distances = nearest.kneighbors(query, n_neighbors=1, return_distance=True)[0][:, 0]
    lower = np.quantile(reference, 0.01, axis=0)
    upper = np.quantile(reference, 0.99, axis=0)
    reference_scale = float(np.sqrt(np.mean(np.square(centered))))
    return {'mean_squared_mahalanobis_to_post': float(np.mean(mahalanobis_squared)), 'median_squared_mahalanobis_to_post': float(np.median(mahalanobis_squared)), 'centroid_shift_over_post_rms_scale': float(np.linalg.norm(query.mean(axis=0) - mean) / max(reference_scale, 1e-12)), 'covariance_relative_frobenius_to_post': float(np.linalg.norm(query_covariance - reference_covariance) / max(np.linalg.norm(reference_covariance), 1e-12)), 'nearest_post_distance_mean': float(query_distances.mean()), 'nearest_post_distance_over_post_self_nn': float(query_distances.mean() / max(self_distances.mean(), 1e-12)), 'fraction_outside_post_1_99_percentile_box': float(np.mean(np.any((query < lower) | (query > upper), axis=1)))}

def classification_metrics(model, embedding: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[dict, np.ndarray]:
    true = np.asarray(labels[centers], dtype=np.int8)
    prediction = np.asarray(model.predict(embedding[centers]), dtype=np.int8)
    matrix = confusion_matrix(true, prediction, labels=np.arange(len(TASKS)), normalize='true')
    return ({'accuracy': float(accuracy_score(true, prediction)), 'balanced_accuracy': float(balanced_accuracy_score(true, prediction)), 'recall': {task: float(matrix[index, index]) for index, task in enumerate(TASKS)}, 'confusion_matrix_row_normalized': matrix.tolist(), 'score_frame_count': int(len(centers)), 'class_counts': np.bincount(true, minlength=len(TASKS)).tolist()}, prediction)

def signed_drop(reference: Mapping, perturbed: Mapping) -> dict:
    return {'accuracy_drop_signed': float(reference['accuracy'] - perturbed['accuracy']), 'balanced_accuracy_drop_signed': float(reference['balanced_accuracy'] - perturbed['balanced_accuracy']), 'recall_drop_signed': {task: float(reference['recall'][task] - perturbed['recall'][task]) for task in TASKS}, 'negative_values_retained': True, 'clipped_or_normalized': False}

def signed_interaction(joint: Mapping, default: Mapping, visual: Mapping) -> dict:
    return {'accuracy_interaction_signed': float(joint['accuracy_drop_signed'] - default['accuracy_drop_signed'] - visual['accuracy_drop_signed']), 'balanced_accuracy_interaction_signed': float(joint['balanced_accuracy_drop_signed'] - default['balanced_accuracy_drop_signed'] - visual['balanced_accuracy_drop_signed']), 'recall_interaction_signed': {task: float(joint['recall_drop_signed'][task] - default['recall_drop_signed'][task] - visual['recall_drop_signed'][task]) for task in TASKS}, 'negative_values_retained': True, 'additivity_not_imposed': True}

def fit_session_model(x: np.ndarray, labels: np.ndarray, run_slices: Sequence[tuple[int, int]], config: CebraConfig, seed: int, svm_c: float) -> dict:
    seed_audit = set_all_seeds(seed)
    indices = np.arange(len(labels), dtype=np.int64)
    estimator, embedding, encoder_audit = fit_encoder(np.array(x, dtype=np.float32, copy=True), indices, [(0, len(labels))], run_slices, config, seed)
    left = int(estimator.model_.get_offset().left)
    right = int(estimator.model_.get_offset().right)
    centers = segment_centers(run_slices, left, right)
    svm = linear_svm(svm_c)
    svm.fit(embedding[centers], labels[centers])
    metric, prediction = classification_metrics(svm, embedding, labels, centers)
    return {'estimator': estimator, 'svm': svm, 'embedding': np.asarray(embedding, dtype=np.float32), 'centers': centers, 'prediction': prediction, 'metric': metric, 'seed_audit': seed_audit, 'encoder_audit': encoder_audit, 'receptive_left': left, 'receptive_right_exclusive': right}

def transform_model(bundle: Mapping, x: np.ndarray, labels: np.ndarray) -> dict:
    embedding = safe_transform(bundle['estimator'], np.asarray(x, dtype=np.float32))
    metric, prediction = classification_metrics(bundle['svm'], embedding, labels, bundle['centers'])
    return {'embedding': embedding, 'metric': metric, 'prediction': prediction}

def release_estimator(bundle: dict) -> None:
    bundle.pop('estimator', None)
    bundle.pop('svm', None)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def source_signature(path: Path, archive_id: str) -> dict:
    stat = path.stat()
    return {'archive_id': archive_id, 'size': int(stat.st_size)}

def job_input_signature(manifest: pd.DataFrame, subject: str, volume_root: Path, label_path: Path) -> dict:
    local = manifest[(manifest.subject == subject) & manifest.session.isin(['ses-01', 'ses-02'])].sort_values(['session', 'task_order'])
    return {'subject': subject, 'variant': 'author_literal', 'roi_sources': [source_signature(volume_root / path, str(path)) for path in local.roi_path], 'atlas_labels': source_signature(label_path, label_path.name)}

def analysis_signature(args, config: CebraConfig, masks: Mapping, random_masks: Mapping) -> dict:
    return {'schema': ANALYSIS_SCHEMA, 'cohort': 'paper_size_low_fd_proxy', 'variant': 'author_literal', 'post_session': 'ses-02', 'baseline_session': 'ses-01', 'cebra_configuration': config.as_dict(), 'preprocessing': args.preprocessing, 'svm_c': args.svm_c, 'seed_repeats': args.seeds, 'fixed_random_masks_per_target_seed0': args.fixed_random_masks, 'random_mask_seed': args.random_mask_seed, 'ood_max_frames': args.ood_max_frames, 'network_masks': {name: np.asarray(mask).tolist() for name, mask in masks.items()}, 'random_masks': {name: [np.asarray(mask).tolist() for mask in values] for name, values in random_masks.items()}}

def job_key(job: Mapping) -> str:
    return f'network_replacement__author_literal__{job['subject']}__seed-repeat-{job['seed_repeat']:02d}__seed-{job['seed']}'

def reusable_result(path: Path, analysis: Mapping, inputs: Mapping) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    artifact_id = payload.get('artifact_id')
    artifact = path.parent.parent / artifact_id if artifact_id else None
    reusable = bool(payload.get('status') == 'completed' and payload.get('analysis_signature') == analysis and (payload.get('input_signature') == inputs) and artifact and artifact.exists())
    if not reusable:
        return False
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            if not archive.files:
                return False
            for name in archive.files:
                archive[name]
    except (OSError, ValueError):
        return False
    return True

def run_job(job: Mapping, args, manifest: pd.DataFrame, config: CebraConfig, masks: Mapping[str, np.ndarray], random_masks: Mapping[str, list[np.ndarray]], analysis: Mapping, inputs: Mapping) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.monotonic()
    subject = job['subject']
    post = load_session(manifest, subject, 'ses-02', 'author_literal', args.volume_root, args.session_cache_dir)
    baseline = load_session(manifest, subject, 'ses-01', 'author_literal', args.volume_root, args.session_cache_dir)
    verify_paired_sessions(post, baseline)
    labels = np.asarray(post['y'], dtype=np.int8)
    post_x, baseline_x, preprocessing_audit = preprocess_sessions(np.asarray(post['x'], dtype=np.float32), np.asarray(baseline['x'], dtype=np.float32), args.preprocessing)
    component_seed = int(job['seed'])
    post_bundle = fit_session_model(post_x, labels, post['run_slices'], config, component_seed, args.svm_c)
    baseline_bundle = fit_session_model(baseline_x, labels, baseline['run_slices'], config, component_seed, args.svm_c)
    if not np.array_equal(post_bundle['centers'], baseline_bundle['centers']):
        raise RuntimeError('baseline and post safe score centers differ')
    centers = post_bundle['centers']
    post_reference_embedding = post_bundle['embedding'][centers]
    post_fixed_on_baseline = transform_model(post_bundle, baseline_x, labels)
    covariance_reference = InputCovarianceReference(post_x, baseline_x, stratified_ood_indices(post['run_slices'], args.ood_max_frames))
    artifacts: dict[str, np.ndarray] = {'labels': labels, 'score_centers': centers.astype(np.int32), 'post_embedding_float32': post_bundle['embedding'].astype(np.float32), 'post_prediction': post_bundle['prediction'].astype(np.int8), 'baseline_embedding_float32': baseline_bundle['embedding'].astype(np.float32), 'baseline_prediction': baseline_bundle['prediction'].astype(np.int8), 'ood_sample_indices': covariance_reference.sample_indices.astype(np.int32)}
    release_estimator(baseline_bundle)
    primary_rows = []
    primary_by_target_mode: dict[tuple[str, str], dict] = {}
    for target, mask in masks.items():
        hybrid, replacement_audit = replace_rois_by_task(post_x, baseline_x, post['run_slices'], baseline['run_slices'], mask)
        input_ood = covariance_reference.metrics(mask)
        fixed = transform_model(post_bundle, hybrid, labels)
        fixed_drop = signed_drop(post_bundle['metric'], fixed['metric'])
        fixed_row = {'target': target, 'target_kind': 'joint' if target == JOINT_TARGET else 'single_network', 'model_mode': 'fixed_post_model', 'mask_indices': np.asarray(mask).tolist(), 'mask_size': int(len(mask)), 'replacement_audit': replacement_audit, 'input_covariance_ood': input_ood, 'embedding_ood': embedding_ood_metrics(post_reference_embedding, fixed['embedding'][centers]), 'hybrid_metric': fixed['metric'], **fixed_drop}
        primary_rows.append(fixed_row)
        primary_by_target_mode[target, 'fixed_post_model'] = fixed_drop
        artifacts[f'fixed__{target}__embedding_float32'] = fixed['embedding'].astype(np.float32)
        artifacts[f'fixed__{target}__prediction'] = fixed['prediction'].astype(np.int8)
        retrained_bundle = fit_session_model(hybrid, labels, post['run_slices'], config, component_seed, args.svm_c)
        retrained_drop = signed_drop(post_bundle['metric'], retrained_bundle['metric'])
        retrained_row = {'target': target, 'target_kind': 'joint' if target == JOINT_TARGET else 'single_network', 'model_mode': 'retrained_hybrid_model', 'mask_indices': np.asarray(mask).tolist(), 'mask_size': int(len(mask)), 'replacement_audit': replacement_audit, 'input_covariance_ood': input_ood, 'embedding_ood': {'comparable_to_post_fixed_coordinates': False}, 'hybrid_metric': retrained_bundle['metric'], **retrained_drop}
        primary_rows.append(retrained_row)
        primary_by_target_mode[target, 'retrained_hybrid_model'] = retrained_drop
        artifacts[f'retrained__{target}__embedding_float32'] = retrained_bundle['embedding'].astype(np.float32)
        artifacts[f'retrained__{target}__prediction'] = retrained_bundle['prediction'].astype(np.int8)
        release_estimator(retrained_bundle)
    interactions = []
    for mode in ('fixed_post_model', 'retrained_hybrid_model'):
        interactions.append({'model_mode': mode, **signed_interaction(primary_by_target_mode[JOINT_TARGET, mode], primary_by_target_mode['default_mode', mode], primary_by_target_mode['visual', mode])})
    random_rows = []
    if job['seed_repeat'] == 0:
        for target, target_random_masks in random_masks.items():
            for mask_id, mask in enumerate(target_random_masks):
                hybrid, replacement_audit = replace_rois_by_task(post_x, baseline_x, post['run_slices'], baseline['run_slices'], mask)
                fixed = transform_model(post_bundle, hybrid, labels)
                row = {'target_size_matched_to': target, 'random_mask_id': mask_id, 'mask_indices': np.asarray(mask).tolist(), 'mask_size': int(len(mask)), 'overlap_with_named_target_count': int(len(np.intersect1d(mask, masks[target]))), 'overlap_with_named_target_fraction': float(len(np.intersect1d(mask, masks[target])) / len(mask)), 'replacement_audit': replacement_audit, 'input_covariance_ood': covariance_reference.metrics(mask), 'fixed_embedding_ood': embedding_ood_metrics(post_reference_embedding, fixed['embedding'][centers]), 'fixed_hybrid_metric': fixed['metric'], 'fixed_drop': signed_drop(post_bundle['metric'], fixed['metric'])}
                artifacts[f'random_mask__{target}__{mask_id:03d}'] = np.asarray(mask, dtype=np.int16)
                random_rows.append(row)
    release_estimator(post_bundle)
    return ({'status': 'completed', 'schema': ANALYSIS_SCHEMA, 'job': dict(job), 'analysis_signature': analysis, 'input_signature': inputs, 'backend': 'cebra', 'reference_models': {'post_original': {'metric': post_bundle['metric'], 'encoder_audit': post_bundle['encoder_audit'], 'seed_audit': post_bundle['seed_audit']}, 'baseline_original': {'metric': baseline_bundle['metric'], 'encoder_audit': baseline_bundle['encoder_audit'], 'seed_audit': baseline_bundle['seed_audit']}, 'fixed_post_model_on_complete_baseline': {'metric': post_fixed_on_baseline['metric'], 'accuracy_drop_signed': float(post_bundle['metric']['accuracy'] - post_fixed_on_baseline['metric']['accuracy']), 'embedding_ood': embedding_ood_metrics(post_reference_embedding, post_fixed_on_baseline['embedding'][centers])}}, 'reference_contrasts': {'independently_trained_post_minus_baseline': signed_drop(post_bundle['metric'], baseline_bundle['metric']), 'fixed_post_model_post_minus_complete_baseline_input': signed_drop(post_bundle['metric'], post_fixed_on_baseline['metric'])}, 'session_alignment': {'post_session': 'ses-02', 'baseline_session': 'ses-01', 'task_order': list(TASKS), 'run_slices': post['run_slices'], 'same_task_same_frame_index': True, 'score_only_receptive_window_inside_run': True, 'score_centers': int(len(centers))}, 'session_cache_audit': {'post': post['session_cache_audit'], 'baseline': baseline['session_cache_audit']}, 'input_preprocessing_audit': preprocessing_audit, 'primary_perturbations': primary_rows, 'dmn_visual_interactions': interactions, 'random_mask_nulls_seed0': random_rows, 'null_policy': {'fixed_random_masks_per_target': args.fixed_random_masks, 'fixed_random_masks_run_only_seed_repeat_zero': True, 'random_masks_fixed_across_subjects': True}, 'runtime_seconds': time.monotonic() - started}, artifacts)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True, type=Path)
    parser.add_argument('--volume-root', required=True, type=Path)
    parser.add_argument('--roi-labels', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--session-cache-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--base-seed', type=int, default=20260826)
    parser.add_argument('--random-mask-seed', type=int, default=20260826)
    parser.add_argument('--fixed-random-masks', type=int, default=20)
    parser.add_argument('--ood-max-frames', type=int, default=512)
    parser.add_argument('--svm-c', type=float, default=1.0)
    parser.add_argument('--preprocessing', choices=['none', 'separate_session_zscore'], default='none')
    parser.add_argument('--model-architecture', default='offset10-model')
    parser.add_argument('--time-offsets', type=int, default=10)
    parser.add_argument('--output-dimension', type=int, default=3)
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--temperature-mode', choices=['constant', 'auto'], default='constant')
    parser.add_argument('--min-temperature', type=float, default=0.1)
    parser.add_argument('--distance', choices=['cosine', 'euclidean'], default='cosine')
    parser.add_argument('--num-hidden-units', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=8)
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--list-jobs', action='store_true')
    args = parser.parse_args()
    if args.fixed_random_masks < 20:
        raise ValueError('production analysis requires at least 20 fixed-model random masks')
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('shard-index must be in [0, shard-count)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.session_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.manifest_dir / 'cebra_run_manifest.parquet')
    membership = pd.read_parquet(args.manifest_dir / 'cebra_cohort_membership.parquet')
    subjects = sorted(membership.loc[membership.paper_size_low_fd_proxy.astype(bool), 'subject'].tolist())
    labels = read_roi_labels(args.roi_labels)
    networks = parse_network_indices(labels)
    masks = target_masks(networks)
    random_masks = fixed_random_masks(masks, args.fixed_random_masks, args.random_mask_seed)
    config = CebraConfig(model_architecture=args.model_architecture, time_offsets=args.time_offsets, output_dimension=args.output_dimension, max_iterations=args.max_iterations, batch_size=args.batch_size, learning_rate=args.learning_rate, temperature=args.temperature, temperature_mode=args.temperature_mode, min_temperature=args.min_temperature, distance=args.distance, num_hidden_units=args.num_hidden_units, device=args.device)
    analysis = analysis_signature(args, config, masks, random_masks)
    jobs = []
    for subject in subjects:
        for seed_repeat in range(args.seeds):
            jobs.append({'analysis': 'cebra_network_baseline_substitution', 'cohort': 'paper_size_low_fd_proxy', 'variant': 'author_literal', 'subject': subject, 'seed_repeat': seed_repeat, 'seed': args.base_seed + numeric_subject(subject) * 100000 + seed_repeat})
    jobs.sort(key=lambda item: (item['subject'], item['seed_repeat']))
    for index, job in enumerate(jobs):
        job['job_index'] = index
    shard_jobs = [job for job in jobs if job['job_index'] % args.shard_count == args.shard_index]
    input_signatures = {job['subject']: job_input_signature(manifest, job['subject'], args.volume_root, args.roi_labels) for job in shard_jobs}
    pending = []
    for job in shard_jobs:
        key = job_key(job)
        result_path = cache_dir / f'{key}.json'
        reusable = not args.overwrite and reusable_result(result_path, analysis, input_signatures[job['subject']])
        if reusable:
            failure_path = cache_dir / f'{key}.failed.json'
            if failure_path.exists():
                failure_path.unlink()
        else:
            pending.append(job)
    if args.max_jobs is not None:
        pending = pending[:args.max_jobs]
    plan = {'schema': ANALYSIS_SCHEMA, 'total_jobs_all_shards': len(jobs), 'jobs_this_shard': len(shard_jobs), 'pending_selected_this_invocation': len(pending), 'shard_index': args.shard_index, 'shard_count': args.shard_count, 'participant_count': len(subjects), 'fit_count_per_seed': 2 + len(masks), 'total_cebra_fits_all_shards': len(subjects) * args.seeds * (2 + len(masks)), 'optimizer_steps_all_shards': len(subjects) * args.seeds * (2 + len(masks)) * config.max_iterations, 'analysis_signature': analysis, 'job_keys_this_shard': [job_key(job) for job in shard_jobs], 'jobs': pending if args.list_jobs else None}
    atomic_json(args.output_dir / f'network_replacement_plan_shard-{args.shard_index:02d}.json', plan)
    print(json.dumps({key: value for key, value in plan.items() if key != 'jobs'}))
    if args.list_jobs:
        return 0
    completed = 0
    failed = 0
    for ordinal, job in enumerate(pending, start=1):
        key = job_key(job)
        result_path = cache_dir / f'{key}.json'
        failure_path = cache_dir / f'{key}.failed.json'
        print(f'[{ordinal}/{len(pending)}] START {key}', flush=True)
        try:
            result, artifacts = run_job(job, args, manifest, config, masks, random_masks, analysis, input_signatures[job['subject']])
            artifact_path = args.output_dir / 'artifacts' / f'{key}.npz'
            atomic_npz(artifact_path, artifacts)
            result['artifact_id'] = artifact_path.relative_to(args.output_dir).as_posix()
            result['artifact_arrays'] = sorted(artifacts)
            atomic_json(result_path, result)
            if failure_path.exists():
                failure_path.unlink()
            completed += 1
            print(f'[{ordinal}/{len(pending)}] DONE {key} seconds={result['runtime_seconds']:.3f}', flush=True)
        except Exception as exc:
            atomic_json(failure_path, {'status': 'failed', 'schema': ANALYSIS_SCHEMA, 'job_key': key, 'job': job, 'analysis_signature': analysis, 'input_signature': input_signatures[job['subject']], 'error_type': type(exc).__name__})
            failed += 1
            print(f'[{ordinal}/{len(pending)}] FAILED {key}: {type(exc).__name__}', flush=True)
    print(json.dumps({'completed': completed, 'failed': failed, 'selected': len(pending)}))
    return 1 if failed else 0
if __name__ == '__main__':
    raise SystemExit(main())
