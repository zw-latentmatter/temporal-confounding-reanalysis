from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
TASKS = ('rest', 'meditation', 'music', 'movie')
SESSIONS = ('ses-01', 'ses-02')
TASK_TO_LABEL = {task: index for index, task in enumerate(TASKS)}
ANALYSIS_OFFSETS = {'author_like': 0, 'blocked': 30000, 'cross_session': 60000, 'participant_heldout': 90000}
SESSION_CACHE_SCHEMA = 'cebra-session-cache-v1'

def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.partial')
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)

def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.partial.npy')
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)

def numeric_subject(subject: str) -> int:
    match = re.search('(\\d+)$', subject)
    if not match:
        raise ValueError(f'cannot parse numeric participant id from {subject}')
    return int(match.group(1))

def set_all_seeds(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    result = {'python_random_seed': seed, 'numpy_global_seed': seed, 'restricted_loader_torch_generator_seed': seed, 'torch_seed': None, 'torch_cuda_seed_all': None, 'torch_deterministic_algorithms': None, 'torch_deterministic_algorithms_warn_only': None, 'cudnn_deterministic': None, 'cudnn_benchmark': None}
    try:
        import torch
        torch.manual_seed(seed)
        result['torch_seed'] = seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            result['torch_cuda_seed_all'] = seed
        torch.use_deterministic_algorithms(True)
        result['torch_deterministic_algorithms'] = True
        result['torch_deterministic_algorithms_warn_only'] = False
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            result['cudnn_deterministic'] = True
            result['cudnn_benchmark'] = False
    except ImportError:
        pass
    return result

def mask_to_segments(mask: np.ndarray, bounds: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for bound_start, bound_stop in bounds:
        local = np.flatnonzero(mask[bound_start:bound_stop]) + bound_start
        if not len(local):
            continue
        starts = np.r_[local[0], local[1:][np.diff(local) > 1]]
        stops = np.r_[local[:-1][np.diff(local) > 1] + 1, local[-1] + 1]
        segments.extend(((int(start), int(stop)) for start, stop in zip(starts, stops)))
    return segments

def segment_centers(segments: Sequence[tuple[int, int]], left: int, right: int) -> np.ndarray:
    parts = []
    for start, stop in segments:
        first = start + left
        last_inclusive = stop - right
        if first <= last_inclusive:
            parts.append(np.arange(first, last_inclusive + 1, dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

def make_blocked_split(n_samples: int, run_slices: Sequence[tuple[int, int]], fold: int, folds: int, purge_frames: int) -> dict:
    if not 0 <= fold < folds:
        raise ValueError(f'fold {fold} outside [0,{folds})')
    test = np.zeros(n_samples, dtype=bool)
    excluded = np.zeros(n_samples, dtype=bool)
    test_segments: list[tuple[int, int]] = []
    for start, stop in run_slices:
        length = stop - start
        test_start = start + fold * length // folds
        test_stop = start + (fold + 1) * length // folds
        test[test_start:test_stop] = True
        excluded[max(start, test_start - purge_frames):min(stop, test_stop + purge_frames)] = True
        test_segments.append((test_start, test_stop))
    train = ~excluded
    purge = excluded & ~test
    train_segments = mask_to_segments(train, run_slices)
    if np.any(train & test) or np.any(train & purge) or np.any(test & purge):
        raise RuntimeError('blocked split masks overlap')
    return {'train_mask': train, 'test_mask': test, 'purge_mask': purge, 'train_segments': train_segments, 'test_segments': test_segments}

def window_inside_one_segment(centers: np.ndarray, segments: Sequence[tuple[int, int]], left: int, right: int) -> np.ndarray:
    allowed = np.zeros(len(centers), dtype=bool)
    for start, stop in segments:
        allowed |= (centers - left >= start) & (centers + right <= stop)
    return allowed

class RestrictedTimeLoader:

    def __init__(self, dataset, allowed_segments: Sequence[tuple[int, int]], *, time_offset: int, num_steps: int, batch_size: int, seed: int, run_slices: Sequence[tuple[int, int]] | None=None):
        import torch
        self.dataset = dataset
        self.device = str(dataset.device)
        self.time_offset = int(time_offset)
        self.num_steps = int(num_steps)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.allowed_segments = [(int(start), int(stop)) for start, stop in allowed_segments]
        self.run_slices = [(int(start), int(stop)) for start, stop in run_slices or []]
        self.left = int(dataset.offset.left)
        self.right = int(dataset.offset.right)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)
        reference_parts = []
        reference_segment_parts = []
        negative_parts = []
        negative_segment_parts = []
        for segment_id, (start, stop) in enumerate(self.allowed_segments):
            negative_first = start + self.left
            negative_last = stop - self.right
            reference_last = negative_last - self.time_offset
            if negative_first <= negative_last:
                centers = torch.arange(negative_first, negative_last + 1, dtype=torch.long, device=self.device)
                negative_parts.append(centers)
                negative_segment_parts.append(torch.full_like(centers, segment_id))
            if negative_first <= reference_last:
                centers = torch.arange(negative_first, reference_last + 1, dtype=torch.long, device=self.device)
                reference_parts.append(centers)
                reference_segment_parts.append(torch.full_like(centers, segment_id))
        if not reference_parts or not negative_parts:
            raise RuntimeError('no valid restricted CEBRA centers; increase segment length or reduce offset/receptive field')
        self.reference_pool = torch.cat(reference_parts)
        self.reference_segment_ids = torch.cat(reference_segment_parts)
        self.negative_pool = torch.cat(negative_parts)
        self.negative_segment_ids = torch.cat(negative_segment_parts)
        self.sampled_reference_by_segment = torch.zeros(len(self.allowed_segments), dtype=torch.long, device=self.device)
        self.sampled_negative_by_segment = torch.zeros(len(self.allowed_segments), dtype=torch.long, device=self.device)
        self.sampled_positive_pairs = 0
        self.boundary_crossings = torch.zeros(1, dtype=torch.long, device=self.device)
        allowed_segment_id = torch.full((len(dataset),), -1, dtype=torch.long, device=self.device)
        for segment_id, (start, stop) in enumerate(self.allowed_segments):
            allowed_segment_id[start:stop] = segment_id
        self._allowed_segment_id = allowed_segment_id
        self.run_boundary_crossings = torch.zeros(1, dtype=torch.long, device=self.device)
        self._first_samples: dict[str, list[int]] | None = None
        self._run_id = None
        if self.run_slices:
            run_id = torch.full((len(dataset),), -1, dtype=torch.long, device=self.device)
            for local_id, (start, stop) in enumerate(self.run_slices):
                run_id[start:stop] = local_id
            self._run_id = run_id

    def __len__(self) -> int:
        return self.num_steps

    def get_indices(self, num_samples: int):
        import torch
        from cebra.data.datatypes import BatchIndex
        ref_pick = torch.randint(len(self.reference_pool), (num_samples,), generator=self._generator, device=self.device)
        neg_pick = torch.randint(len(self.negative_pool), (num_samples,), generator=self._generator, device=self.device)
        reference = self.reference_pool[ref_pick]
        positive = reference + self.time_offset
        negative = self.negative_pool[neg_pick]
        ref_segment = self.reference_segment_ids[ref_pick]
        neg_segment = self.negative_segment_ids[neg_pick]
        self.sampled_reference_by_segment += torch.bincount(ref_segment, minlength=len(self.allowed_segments))
        self.sampled_negative_by_segment += torch.bincount(neg_segment, minlength=len(self.allowed_segments))
        self.sampled_positive_pairs += int(num_samples)
        self.boundary_crossings += torch.sum((self._allowed_segment_id[reference] < 0) | (self._allowed_segment_id[positive] < 0) | (self._allowed_segment_id[reference] != self._allowed_segment_id[positive]))
        if self._run_id is not None:
            self.run_boundary_crossings += torch.sum(self._run_id[reference] != self._run_id[positive])
        if self._first_samples is None:
            keep = min(64, num_samples)
            self._first_samples = {'reference': reference[:keep].detach().cpu().tolist(), 'positive': positive[:keep].detach().cpu().tolist(), 'negative': negative[:keep].detach().cpu().tolist()}
        return BatchIndex(reference=reference, positive=positive, negative=negative)

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self.dataset.load_batch(self.get_indices(self.batch_size))

    def audit(self) -> dict:
        ref_counts = self.sampled_reference_by_segment.detach().cpu().tolist()
        neg_counts = self.sampled_negative_by_segment.detach().cpu().tolist()
        return {'loader': 'RestrictedTimeLoader', 'seed': self.seed, 'num_steps': self.num_steps, 'batch_size': self.batch_size, 'time_offset': self.time_offset, 'receptive_window_left': self.left, 'receptive_window_right_exclusive': self.right, 'allowed_segments': self.allowed_segments, 'reference_pool_size': int(len(self.reference_pool)), 'negative_pool_size': int(len(self.negative_pool)), 'sampled_reference_by_segment': ref_counts, 'sampled_negative_by_segment': neg_counts, 'every_training_segment_sampled_as_reference': all((count > 0 for count in ref_counts)), 'every_training_segment_sampled_as_negative': all((count > 0 for count in neg_counts)), 'sampled_positive_pairs': self.sampled_positive_pairs, 'artificial_temporal_adjacencies': int(self.boundary_crossings.detach().cpu().item()), 'sampled_pairs_crossing_declared_run_boundaries': int(self.run_boundary_crossings.detach().cpu().item()), 'first_sampled_centers': self._first_samples}

@dataclass(frozen=True)
class CebraConfig:
    model_architecture: str = 'offset10-model'
    conditional: str = 'time'
    time_offsets: int = 10
    output_dimension: int = 3
    max_iterations: int = 2000
    batch_size: int = 512
    learning_rate: float = 0.0003
    temperature: float = 1.0
    temperature_mode: str = 'constant'
    min_temperature: float = 0.1
    distance: str = 'cosine'
    num_hidden_units: int = 32
    device: str = 'cuda'

    def as_dict(self) -> dict:
        return dict(self.__dict__)

def make_cebra_estimator(config: CebraConfig):
    import cebra
    if cebra.__version__ != '0.6.1':
        raise RuntimeError(f'this restricted-loader integration is pinned to CEBRA 0.6.1, got {cebra.__version__}')
    return cebra.CEBRA(model_architecture=config.model_architecture, conditional=config.conditional, time_offsets=config.time_offsets, output_dimension=config.output_dimension, max_iterations=config.max_iterations, batch_size=config.batch_size, learning_rate=config.learning_rate, temperature_mode=config.temperature_mode, temperature=config.temperature, min_temperature=config.min_temperature, distance=config.distance, num_hidden_units=config.num_hidden_units, pad_before_transform=True, device=config.device, verbose=False)

def fit_restricted_cebra(x: np.ndarray, allowed_segments: Sequence[tuple[int, int]], run_slices: Sequence[tuple[int, int]], config: CebraConfig, seed: int):
    estimator = make_cebra_estimator(config)
    state = estimator._prepare_fit(x)
    solver, model, original_loader, is_multisession = state
    if is_multisession:
        raise RuntimeError('restricted loader requires a single encoder/session')
    loader = RestrictedTimeLoader(original_loader.dataset, allowed_segments, time_offset=config.time_offsets, num_steps=config.max_iterations, batch_size=config.batch_size, seed=seed, run_slices=run_slices)
    estimator._partial_fit(solver, model, loader, is_multisession)
    audit = loader.audit()
    if not audit['every_training_segment_sampled_as_reference']:
        raise RuntimeError('at least one allowed training segment was never sampled as reference')
    if audit['artificial_temporal_adjacencies'] != 0:
        raise RuntimeError('restricted loader sampled a temporal pair across an allowed-segment boundary')
    audit['final_loss'] = float(estimator.solver_.history[-1])
    audit['loss_history_length'] = len(estimator.solver_.history)
    return (estimator, audit)

def safe_transform(estimator, x: np.ndarray) -> np.ndarray:
    embedding = np.asarray(estimator.transform(x), dtype=np.float32)
    if embedding.shape[0] != x.shape[0]:
        raise RuntimeError(f'CEBRA transform returned {embedding.shape[0]} rows for {x.shape[0]} input frames')
    return embedding

def linear_svm(c_value: float) -> Pipeline:
    return Pipeline([('scale_embedding', StandardScaler()), ('svm', SVC(kernel='linear', C=c_value, class_weight='balanced', cache_size=512))])

def score_readout(embedding: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, c_value: float, *, include_labels: Sequence[int] | None=None, movie_vs_closed: bool=False) -> dict:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    original_label_values = np.asarray(sorted(include_labels), dtype=np.int64) if include_labels is not None else np.arange(len(TASKS), dtype=np.int64)
    if include_labels is not None:
        train_indices = train_indices[np.isin(labels[train_indices], original_label_values)]
        test_indices = test_indices[np.isin(labels[test_indices], original_label_values)]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    if movie_vs_closed:
        train_labels = (train_labels == TASK_TO_LABEL['movie']).astype(np.int8)
        test_labels = (test_labels == TASK_TO_LABEL['movie']).astype(np.int8)
        score_labels = np.arange(2)
        class_names = ['closed_contexts', 'movie']
        reported_label_values = ['closed_contexts', TASK_TO_LABEL['movie']]
    else:
        score_labels = original_label_values
        class_names = [TASKS[int(label)] for label in score_labels]
        reported_label_values = score_labels.tolist()
    if len(np.unique(train_labels)) != len(score_labels):
        raise RuntimeError('SVM training split does not contain every requested class')
    if len(test_indices) == 0:
        raise RuntimeError('SVM test split is empty after class restriction')
    model = linear_svm(c_value)
    model.fit(embedding[train_indices], train_labels)
    prediction = model.predict(embedding[test_indices])
    matrix = confusion_matrix(test_labels, prediction, labels=score_labels, normalize='true')
    test_class_counts = np.asarray([np.sum(test_labels == label) for label in score_labels], dtype=np.int64)
    declared_class_balanced_accuracy = float(np.mean(np.diag(matrix)))
    present_classes = test_class_counts > 0
    observed_class_balanced_accuracy = float(np.mean(np.diag(matrix)[present_classes]))
    return {'accuracy': float(accuracy_score(test_labels, prediction)), 'balanced_accuracy': declared_class_balanced_accuracy, 'balanced_accuracy_observed_test_classes_only': observed_class_balanced_accuracy, 'balanced_accuracy_declared_class_set': True, 'test_contains_all_declared_classes': bool(present_classes.all()), 'class_names': class_names, 'class_label_values': reported_label_values, 'class_recall': np.diag(matrix).tolist(), 'confusion_matrix_row_normalized': matrix.tolist(), 'train_frames': int(len(train_indices)), 'test_frames': int(len(test_indices)), 'train_class_counts': [int(np.sum(train_labels == label)) for label in score_labels], 'test_class_counts': test_class_counts.astype(int).tolist()}

def context_readout_suite(embedding: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, c_value: float, split: int) -> list[dict]:
    specifications = [('four_contexts', None, False), ('closed_only_three_contexts', (0, 1, 2), False), ('movie_vs_closed_contexts', None, True)]
    specifications.extend(((f'leave_out_{task}_three_contexts', tuple((label for label in range(4) if label != omitted)), False) for omitted, task in enumerate(TASKS)))
    metrics = []
    for readout, include_labels, collapse in specifications:
        metrics.append({'readout': readout, 'split': split, **score_readout(embedding, labels, train_indices, test_indices, c_value, include_labels=include_labels, movie_vs_closed=collapse)})
    return metrics

def circular_block_shift_null(embedding: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, c_value: float, *, seed: int, repeats: int, block_size: int, minimum_circular_distance: int, split: int) -> tuple[list[dict], dict]:
    if repeats <= 0:
        return ([], {'requested_repeats': 0, 'candidate_shifts': [], 'actual_shifts': [], 'minimum_circular_distance': int(minimum_circular_distance)})
    possible = np.arange(block_size, len(labels), block_size, dtype=np.int64)
    possible = possible[np.minimum(possible, len(labels) - possible) >= minimum_circular_distance]
    if not len(possible):
        raise RuntimeError('session is too short for the requested circular block-shift null')
    eligible = []
    excluded_missing_training_classes = []
    declared_labels = np.arange(len(TASKS), dtype=np.int64)
    for shift in possible:
        shifted = np.roll(labels, int(shift))
        train_counts = [int(np.sum(shifted[train_indices] == label)) for label in declared_labels]
        if all((count > 0 for count in train_counts)):
            eligible.append(int(shift))
        else:
            excluded_missing_training_classes.append({'shift': int(shift), 'train_class_counts': train_counts})
    if not eligible:
        raise RuntimeError('no circular block shift leaves every requested class in the SVM training split')
    rng = np.random.default_rng(seed + 7919)
    shifts = rng.choice(eligible, size=repeats, replace=len(eligible) < repeats)
    metrics = []
    for null_index, shift in enumerate(shifts):
        shifted = np.roll(labels, int(shift))
        metrics.append({'readout': 'circular_block_shift_null_four_contexts', 'split': split, 'null_index': null_index, 'label_shift_frames': int(shift), 'null_block_size_frames': int(block_size), **score_readout(embedding, shifted, train_indices, test_indices, c_value)})
    return (metrics, {'requested_repeats': int(repeats), 'block_size_frames': int(block_size), 'minimum_circular_distance': int(minimum_circular_distance), 'candidate_shifts': [int(value) for value in possible], 'training_class_complete_candidate_shifts': eligible, 'candidate_shifts_excluded_missing_training_classes': excluded_missing_training_classes, 'actual_shifts': [int(value) for value in shifts]})

def load_session(run_manifest: pd.DataFrame, subject: str, session: str, variant: str, volume_root: Path, cache_dir: Path | None=None) -> dict:
    local = run_manifest[(run_manifest.subject == subject) & (run_manifest.session == session)]
    by_task = {row.task: row for row in local.itertuples()}
    if set(by_task) != set(TASKS):
        raise RuntimeError(f'incomplete session for {subject} {session}: {sorted(by_task)}')
    source_ids = [str(by_task[task].roi_path) for task in TASKS]
    source_signature = []
    for source_id in source_ids:
        stat = (volume_root / source_id).stat()
        source_signature.append({'archive_id': source_id, 'size': int(stat.st_size)})
    signature = {'schema': SESSION_CACHE_SCHEMA, 'subject': subject, 'session': session, 'variant': variant, 'sources': source_signature, 'config': {'task_order': list(TASKS), 'feature_count': 332, 'x_dtype': 'float32', 'y_dtype': 'int8'}}
    cache_paths = None
    if cache_dir is not None:
        safe_variant = ''.join((character if character.isalnum() or character in '_.-' else f'_u{ord(character):04x}_' for character in variant))
        prefix = cache_dir / subject / session / safe_variant
        cache_paths = {'x': prefix / 'x.npy', 'y': prefix / 'y.npy', 'metadata': prefix / 'metadata.json'}
        if all((path.exists() for path in cache_paths.values())):
            metadata = json.loads(cache_paths['metadata'].read_text(encoding='utf-8'))
            if metadata.get('signature') == signature:
                return {'x': np.load(cache_paths['x'], mmap_mode='r', allow_pickle=False), 'y': np.load(cache_paths['y'], mmap_mode='r', allow_pickle=False), 'run_slices': [tuple(item) for item in metadata['run_slices']], 'source_ids': source_ids, 'task_lengths': metadata['task_lengths'], 'session_cache_audit': {'cache_hit': True, 'schema': SESSION_CACHE_SCHEMA, 'signature': signature}}
    parts = []
    labels = []
    run_slices = []
    offset = 0
    for task in TASKS:
        row = by_task[task]
        path = volume_root / row.roi_path
        with np.load(path) as archive:
            array = np.asarray(archive[variant], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 332 or (not np.isfinite(array).all()):
            raise RuntimeError(f'invalid {variant} array in {path}: {array.shape}')
        parts.append(array)
        labels.append(np.full(len(array), TASK_TO_LABEL[task], dtype=np.int8))
        run_slices.append((offset, offset + len(array)))
        offset += len(array)
    x = np.concatenate(parts)
    y = np.concatenate(labels)
    task_lengths = {task: stop - start for task, (start, stop) in zip(TASKS, run_slices)}
    cache_audit = {'cache_hit': False, 'schema': SESSION_CACHE_SCHEMA, 'signature': signature}
    if cache_paths is not None:
        atomic_npy(cache_paths['x'], x)
        atomic_npy(cache_paths['y'], y)
        atomic_json(cache_paths['metadata'], {'signature': signature, 'run_slices': run_slices, 'task_lengths': task_lengths})
        x = np.load(cache_paths['x'], mmap_mode='r', allow_pickle=False)
        y = np.load(cache_paths['y'], mmap_mode='r', allow_pickle=False)
    return {'x': x, 'y': y, 'run_slices': run_slices, 'source_ids': source_ids, 'task_lengths': task_lengths, 'session_cache_audit': cache_audit}

def scale_train_only(x: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    scaler.fit(x[fit_indices])
    transformed = scaler.transform(x).astype(np.float32)
    return (transformed, {'scaler_fit_frames': int(len(fit_indices)), 'scaler_mean_min': float(np.min(scaler.mean_)), 'scaler_mean_max': float(np.max(scaler.mean_)), 'scaler_scale_min': float(np.min(scaler.scale_)), 'scaler_scale_max': float(np.max(scaler.scale_))})

def preprocess_input(x: np.ndarray, fit_indices: np.ndarray, preprocessing: str) -> tuple[np.ndarray, dict]:
    if preprocessing == 'none':
        return (np.array(x, dtype=np.float32, copy=True), {'preprocessing': 'none', 'scaler_was_fit': False, 'scaler_fit_frames': 0})
    if preprocessing == 'train_zscore':
        transformed, audit = scale_train_only(x, fit_indices)
        return (transformed, {'preprocessing': 'train_zscore', 'scaler_was_fit': True, **audit})
    raise ValueError(preprocessing)

def fit_encoder(x_scaled: np.ndarray, fit_indices: np.ndarray, allowed_segments: Sequence[tuple[int, int]], run_slices: Sequence[tuple[int, int]], config: CebraConfig, seed: int):
    estimator, audit = fit_restricted_cebra(x_scaled, allowed_segments, run_slices, config, seed)
    return (estimator, safe_transform(estimator, x_scaled), audit)

def maybe_silhouette(embedding: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> dict:
    from sklearn.metrics import silhouette_samples
    rng = np.random.default_rng(seed)
    selected = []
    per_class = max(2, max_samples // len(TASKS))
    for label in range(len(TASKS)):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) > per_class:
            candidates = np.sort(rng.choice(candidates, size=per_class, replace=False))
        selected.append(candidates)
    indices = np.concatenate(selected)
    values = silhouette_samples(embedding[indices], labels[indices], metric='euclidean')
    result = {'silhouette_mean': float(values.mean()), 'silhouette_sample_n': int(len(values))}
    for label, task in enumerate(TASKS):
        result[f'silhouette_{task}'] = float(values[labels[indices] == label].mean())
    return result

def run_author_like(job: dict, args, run_manifest: pd.DataFrame, config: CebraConfig) -> tuple[dict, dict]:
    data = load_session(run_manifest, job['subject'], job['session'], job['variant'], args.volume_root, args.session_cache_dir)
    n = len(data['y'])
    all_indices = np.arange(n, dtype=np.int64)
    x_prepared, scaler_audit = preprocess_input(data['x'], all_indices, args.preprocessing)
    estimator, embedding, encoder_audit = fit_encoder(x_prepared, all_indices, [(0, n)], data['run_slices'], config, job['seed'])
    encoder_audit['author_like_full_concatenated_session'] = True
    encoder_audit['run_boundary_pairs_intentionally_allowed'] = True
    metrics = []
    resub = score_readout(embedding, data['y'], all_indices, all_indices, args.svm_c)
    metrics.append({'readout': 'resubstitution', 'split': 0, **resub})
    split_arrays: dict[str, np.ndarray] = {'encoder_fit_indices': all_indices.astype(np.int32), 'embedding_float32': embedding.astype(np.float32), 'embedding_labels': np.asarray(data['y'], dtype=np.int8)}
    splitter = StratifiedShuffleSplit(n_splits=args.random_splits, test_size=args.test_fraction, random_state=job['seed'])
    for split, (train, test) in enumerate(splitter.split(embedding, data['y'])):
        train = train.astype(np.int64)
        test = test.astype(np.int64)
        scored = score_readout(embedding, data['y'], train, test, args.svm_c)
        metrics.append({'readout': 'random_stratified_frames', 'split': split, **scored})
        split_arrays[f'random_{split}_train'] = train.astype(np.int32)
        split_arrays[f'random_{split}_test'] = test.astype(np.int32)
    return ({'task_lengths': data['task_lengths'], 'session_cache_audit': data['session_cache_audit'], 'scaler_audit': scaler_audit, 'encoder_audit': encoder_audit, 'silhouette': maybe_silhouette(embedding, data['y'], args.silhouette_max_samples, job['seed']), 'metrics': metrics}, split_arrays)

def run_blocked(job: dict, args, run_manifest: pd.DataFrame, config: CebraConfig) -> tuple[dict, dict]:
    data = load_session(run_manifest, job['subject'], job['session'], job['variant'], args.volume_root, args.session_cache_dir)
    split = make_blocked_split(len(data['y']), data['run_slices'], job['fold'], args.blocked_folds, args.purge_frames)
    train_frames = np.flatnonzero(split['train_mask'])
    x_prepared, scaler_audit = preprocess_input(data['x'], train_frames, args.preprocessing)
    estimator, embedding, encoder_audit = fit_encoder(x_prepared, train_frames, split['train_segments'], data['run_slices'], config, job['seed'])
    if encoder_audit.get('artificial_temporal_adjacencies', 0) != 0:
        raise RuntimeError('blocked encoder sampled across an allowed training segment')
    if encoder_audit.get('sampled_pairs_crossing_declared_run_boundaries', 0) != 0:
        raise RuntimeError('blocked encoder sampled across an fMRI run boundary')
    left = int(estimator.model_.get_offset().left)
    right = int(estimator.model_.get_offset().right)
    train_centers = segment_centers(split['train_segments'], left, right)
    test_centers = segment_centers(split['test_segments'], left, right)
    if np.intersect1d(train_centers, test_centers).size:
        raise RuntimeError('safe train/test centers overlap')
    if not window_inside_one_segment(train_centers, split['train_segments'], left, right).all():
        raise RuntimeError('a training convolution window crosses a train-segment boundary')
    if not window_inside_one_segment(test_centers, split['test_segments'], left, right).all():
        raise RuntimeError('a test convolution window crosses a test-segment boundary')
    metrics = context_readout_suite(embedding, data['y'], train_centers, test_centers, args.svm_c, job['fold'])
    null_repeats = args.null_shifts if job['seed_repeat'] == 0 else 0
    null_metrics, null_audit = circular_block_shift_null(embedding, data['y'], train_centers, test_centers, args.svm_c, seed=job['seed'], repeats=null_repeats, block_size=args.null_block_size, minimum_circular_distance=max(100, args.purge_frames), split=job['fold'])
    metrics.extend(null_metrics)
    return ({'task_lengths': data['task_lengths'], 'session_cache_audit': data['session_cache_audit'], 'scaler_audit': scaler_audit, 'encoder_audit': encoder_audit, 'test_centers_silhouette': maybe_silhouette(embedding[test_centers], data['y'][test_centers], args.silhouette_max_samples, job['seed']), 'split_audit': {'fold': job['fold'], 'folds': args.blocked_folds, 'purge_frames': args.purge_frames, 'train_segments': split['train_segments'], 'test_segments': split['test_segments'], 'train_frames': int(split['train_mask'].sum()), 'test_frames': int(split['test_mask'].sum()), 'purge_only_frames': int(split['purge_mask'].sum()), 'safe_train_centers': int(len(train_centers)), 'safe_test_centers': int(len(test_centers)), 'receptive_left': left, 'receptive_right_exclusive': right, 'train_test_gap_zero_overlap': True, 'train_and_test_transform_scored_only_on_segment_interior': True, 'circular_block_shift_null': null_audit, 'null_run_only_for_seed_repeat_zero': True}, 'metrics': metrics}, {'encoder_fit_indices': train_frames.astype(np.int32), 'train_mask': split['train_mask'].astype(np.uint8), 'test_mask': split['test_mask'].astype(np.uint8), 'purge_mask': split['purge_mask'].astype(np.uint8), 'svm_train_centers': train_centers.astype(np.int32), 'svm_test_centers': test_centers.astype(np.int32), 'test_embedding_float32': embedding[test_centers].astype(np.float32), 'test_embedding_labels': np.asarray(data['y'][test_centers], dtype=np.int8), 'test_embedding_centers': test_centers.astype(np.int32)})

def run_cross_session(job: dict, args, run_manifest: pd.DataFrame, config: CebraConfig) -> tuple[dict, dict]:
    train_data = load_session(run_manifest, job['subject'], job['train_session'], job['variant'], args.volume_root, args.session_cache_dir)
    test_data = load_session(run_manifest, job['subject'], job['test_session'], job['variant'], args.volume_root, args.session_cache_dir)
    train_indices = np.arange(len(train_data['y']), dtype=np.int64)
    if args.preprocessing == 'none':
        train_prepared = np.array(train_data['x'], dtype=np.float32, copy=True)
        test_prepared = np.array(test_data['x'], dtype=np.float32, copy=True)
        scaler_audit = {'preprocessing': 'none', 'scaler_was_fit': False, 'scaler_fit_frames': 0, 'fit_session': job['train_session'], 'test_session_used_in_scaler_fit': False}
    else:
        scaler = StandardScaler().fit(train_data['x'])
        train_prepared = scaler.transform(train_data['x']).astype(np.float32)
        test_prepared = scaler.transform(test_data['x']).astype(np.float32)
        scaler_audit = {'preprocessing': 'train_zscore', 'scaler_was_fit': True, 'scaler_fit_frames': len(train_data['x']), 'fit_session': job['train_session'], 'test_session_used_in_scaler_fit': False, 'scaler_mean_min': float(np.min(scaler.mean_)), 'scaler_mean_max': float(np.max(scaler.mean_)), 'scaler_scale_min': float(np.min(scaler.scale_)), 'scaler_scale_max': float(np.max(scaler.scale_))}
    estimator, train_embedding, encoder_audit = fit_encoder(train_prepared, train_indices, train_data['run_slices'], train_data['run_slices'], config, job['seed'])
    if encoder_audit.get('artificial_temporal_adjacencies', 0) != 0:
        raise RuntimeError('cross-session encoder sampled across an allowed training segment')
    if encoder_audit.get('sampled_pairs_crossing_declared_run_boundaries', 0) != 0:
        raise RuntimeError('cross-session encoder sampled across an fMRI run boundary')
    test_embedding = safe_transform(estimator, test_prepared)
    left = int(estimator.model_.get_offset().left)
    right = int(estimator.model_.get_offset().right)
    train_centers = segment_centers(train_data['run_slices'], left, right)
    test_centers = segment_centers(test_data['run_slices'], left, right)
    joined_embedding = np.concatenate([train_embedding, test_embedding], axis=0)
    joined_labels = np.concatenate([train_data['y'], test_data['y']], axis=0)
    joined_test_centers = test_centers + len(train_embedding)
    metrics = context_readout_suite(joined_embedding, joined_labels, train_centers, joined_test_centers, args.svm_c, 0)
    for metric in metrics:
        metric['validation'] = 'cross_session_independent_acquisition_fixed_order'
    return ({'train_task_lengths': train_data['task_lengths'], 'test_task_lengths': test_data['task_lengths'], 'session_cache_audit': {'train': train_data['session_cache_audit'], 'test': test_data['session_cache_audit']}, 'scaler_audit': scaler_audit, 'encoder_audit': encoder_audit, 'split_audit': {'train_session': job['train_session'], 'test_session': job['test_session'], 'same_participant': True, 'test_session_used_in_encoder_or_svm_fit': False, 'score_centers_have_receptive_window_inside_individual_run': True, 'receptive_left': left, 'receptive_right_exclusive': right}, 'metrics': metrics}, {'train_encoder_fit_indices': train_indices.astype(np.int32), 'svm_train_centers': train_centers.astype(np.int32), 'svm_test_centers': test_centers.astype(np.int32)})

def concatenate_participant_sessions(run_manifest: pd.DataFrame, subjects: Sequence[str], session: str, variant: str, volume_root: Path, cache_dir: Path | None) -> dict:
    x_parts = []
    y_parts = []
    allowed_segments = []
    participant_slices = {}
    per_subject = {}
    offset = 0
    for subject in subjects:
        data = load_session(run_manifest, subject, session, variant, volume_root, cache_dir)
        start = offset
        x_parts.append(np.asarray(data['x'], dtype=np.float32))
        y_parts.append(np.asarray(data['y'], dtype=np.int8))
        allowed_segments.extend(((offset + run_start, offset + run_stop) for run_start, run_stop in data['run_slices']))
        offset += len(data['y'])
        participant_slices[subject] = (start, offset)
        per_subject[subject] = data
    return {'x': np.concatenate(x_parts, axis=0), 'y': np.concatenate(y_parts, axis=0), 'allowed_segments': allowed_segments, 'participant_slices': participant_slices, 'per_subject': per_subject}

def run_participant_heldout(job: dict, args, run_manifest: pd.DataFrame, config: CebraConfig) -> tuple[dict, dict]:
    train = concatenate_participant_sessions(run_manifest, job['train_subjects'], job['session'], job['variant'], args.volume_root, args.session_cache_dir)
    train_indices = np.arange(len(train['y']), dtype=np.int64)
    if args.preprocessing == 'none':
        train_prepared = np.array(train['x'], dtype=np.float32, copy=True)
        scaler = None
        scaler_audit = {'preprocessing': 'none', 'scaler_was_fit': False, 'scaler_fit_frames': 0, 'held_out_participants_used_in_scaler_fit': False}
    else:
        scaler = StandardScaler().fit(train['x'])
        train_prepared = scaler.transform(train['x']).astype(np.float32)
        scaler_audit = {'preprocessing': 'train_zscore', 'scaler_was_fit': True, 'scaler_fit_frames': int(len(train['x'])), 'held_out_participants_used_in_scaler_fit': False, 'scaler_mean_min': float(np.min(scaler.mean_)), 'scaler_mean_max': float(np.max(scaler.mean_)), 'scaler_scale_min': float(np.min(scaler.scale_)), 'scaler_scale_max': float(np.max(scaler.scale_))}
    estimator, train_embedding, encoder_audit = fit_encoder(train_prepared, train_indices, train['allowed_segments'], train['allowed_segments'], config, job['seed'])
    if encoder_audit.get('artificial_temporal_adjacencies', 0) != 0:
        raise RuntimeError('participant-held-out encoder crossed an allowed participant/run segment')
    if encoder_audit.get('sampled_pairs_crossing_declared_run_boundaries', 0) != 0:
        raise RuntimeError('participant-held-out encoder crossed a participant or run boundary')
    left = int(estimator.model_.get_offset().left)
    right = int(estimator.model_.get_offset().right)
    train_centers = segment_centers(train['allowed_segments'], left, right)
    svm = linear_svm(args.svm_c)
    svm.fit(train_embedding[train_centers], train['y'][train_centers])
    metrics = []
    cache_audit = {'train': {subject: train['per_subject'][subject]['session_cache_audit'] for subject in job['train_subjects']}, 'test': {}}
    test_artifact_subject = []
    test_artifact_embedding = []
    test_artifact_labels = []
    for subject in job['test_subjects']:
        data = load_session(run_manifest, subject, job['session'], job['variant'], args.volume_root, args.session_cache_dir)
        if scaler is None:
            test_prepared = np.array(data['x'], dtype=np.float32, copy=True)
        else:
            test_prepared = scaler.transform(data['x']).astype(np.float32)
        test_embedding = safe_transform(estimator, test_prepared)
        test_centers = segment_centers(data['run_slices'], left, right)
        prediction = svm.predict(test_embedding[test_centers])
        true_labels = np.asarray(data['y'][test_centers], dtype=np.int8)
        matrix = confusion_matrix(true_labels, prediction, labels=np.arange(len(TASKS)), normalize='true')
        metrics.append({'test_subject': subject, 'readout': 'participant_heldout_shared_four_contexts', 'split': job['fold'], 'accuracy': float(accuracy_score(true_labels, prediction)), 'balanced_accuracy': float(balanced_accuracy_score(true_labels, prediction)), 'class_names': list(TASKS), 'class_label_values': list(range(len(TASKS))), 'class_recall': np.diag(matrix).tolist(), 'confusion_matrix_row_normalized': matrix.tolist(), 'train_frames': int(len(train_centers)), 'test_frames': int(len(test_centers)), 'train_class_counts': np.bincount(train['y'][train_centers], minlength=len(TASKS)).tolist(), 'test_class_counts': np.bincount(true_labels, minlength=len(TASKS)).tolist()})
        cache_audit['test'][subject] = data['session_cache_audit']
        test_artifact_subject.append(np.full(len(test_centers), subject))
        test_artifact_embedding.append(test_embedding[test_centers].astype(np.float32))
        test_artifact_labels.append(true_labels)
    return ({'session_cache_audit': cache_audit, 'scaler_audit': scaler_audit, 'encoder_audit': encoder_audit, 'split_audit': {'participant_fold': job['fold'], 'participant_folds': args.participant_folds, 'participant_split_seed': args.participant_split_seed, 'train_subjects': job['train_subjects'], 'test_subjects': job['test_subjects'], 'train_subject_count': len(job['train_subjects']), 'test_subject_count': len(job['test_subjects']), 'held_out_participants_used_in_encoder_or_svm_fit': False, 'each_training_participant_run_is_an_independent_allowed_segment': True, 'score_centers_have_receptive_window_inside_individual_run': True, 'receptive_left': left, 'receptive_right_exclusive': right}, 'metrics': metrics}, {'svm_train_centers': train_centers.astype(np.int32), 'test_embedding_float32': np.concatenate(test_artifact_embedding), 'test_embedding_labels': np.concatenate(test_artifact_labels), 'test_embedding_subject': np.concatenate(test_artifact_subject)})

def cohort_column(name: str) -> str:
    mapping = {'all_complete': 'all_complete_both_sessions', 'modal_length': 'modal_length_both_sessions', 'paper_size_low_fd_proxy': 'paper_size_low_fd_proxy'}
    return mapping[name]

def build_jobs(args, membership: pd.DataFrame) -> list[dict]:
    selected = membership[membership[cohort_column(args.cohort)].astype(bool)]
    subjects = sorted(selected.subject.tolist())
    jobs: list[dict] = []
    analyses = ('author_like', 'blocked', 'cross_session', 'participant_heldout') if args.analysis == 'all' else (args.analysis,)
    for variant in args.variants:
        for subject in subjects:
            subject_base = args.base_seed + numeric_subject(subject) * 100000
            if 'author_like' in analyses:
                for session_index, session in enumerate(SESSIONS):
                    for repeat in range(args.author_seeds):
                        jobs.append({'analysis': 'author_like', 'subject': subject, 'session': session, 'variant': variant, 'seed_repeat': repeat, 'seed': subject_base + ANALYSIS_OFFSETS['author_like'] + session_index * 1000 + repeat})
            if 'blocked' in analyses:
                for session_index, session in enumerate(SESSIONS):
                    for fold in range(args.blocked_folds):
                        for repeat in range(args.blocked_seeds):
                            jobs.append({'analysis': 'blocked', 'subject': subject, 'session': session, 'variant': variant, 'fold': fold, 'seed_repeat': repeat, 'seed': subject_base + ANALYSIS_OFFSETS['blocked'] + session_index * 5000 + fold * 100 + repeat})
            if 'cross_session' in analyses:
                for direction, (train_session, test_session) in enumerate((('ses-01', 'ses-02'), ('ses-02', 'ses-01'))):
                    for repeat in range(args.transfer_seeds):
                        jobs.append({'analysis': 'cross_session', 'subject': subject, 'train_session': train_session, 'test_session': test_session, 'variant': variant, 'direction': f'{train_session}_to_{test_session}', 'seed_repeat': repeat, 'seed': subject_base + ANALYSIS_OFFSETS['cross_session'] + direction * 1000 + repeat})
        if 'participant_heldout' in analyses:
            splitter = KFold(n_splits=args.participant_folds, shuffle=True, random_state=args.participant_split_seed)
            subject_array = np.asarray(subjects)
            for session_index, session in enumerate(SESSIONS):
                for fold, (train_index, test_index) in enumerate(splitter.split(subject_array)):
                    train_subjects = sorted(subject_array[train_index].tolist())
                    test_subjects = sorted(subject_array[test_index].tolist())
                    for repeat in range(args.participant_seeds):
                        jobs.append({'analysis': 'participant_heldout', 'session': session, 'variant': variant, 'fold': fold, 'train_subjects': train_subjects, 'test_subjects': test_subjects, 'seed_repeat': repeat, 'seed': args.base_seed + ANALYSIS_OFFSETS['participant_heldout'] + session_index * 1000 + fold * 100 + repeat})
    jobs.sort(key=lambda job: (job['analysis'], job['variant'], job.get('subject', ''), job.get('session', ''), job.get('direction', ''), job.get('fold', -1), job['seed_repeat']))
    for index, job in enumerate(jobs):
        job['job_index'] = index
        job['cohort'] = args.cohort
    return jobs

def job_key(job: dict) -> str:
    parts = [job['analysis'], job['variant']]
    if 'subject' in job:
        parts.append(job['subject'])
    if 'session' in job:
        parts.append(job['session'])
    if 'direction' in job:
        parts.append(job['direction'])
    if 'fold' in job:
        parts.append(f'fold-{job['fold']:02d}')
    parts.append(f'seed-{job['seed']}')
    return '__'.join(parts)

def analysis_signature(args, config: CebraConfig) -> dict:
    return {'schema': 'cebra-analysis-signature-v2', 'cohort': args.cohort, 'preprocessing': args.preprocessing, 'cebra_configuration': config.as_dict(), 'svm_c': args.svm_c, 'random_splits': args.random_splits, 'test_fraction': args.test_fraction, 'blocked_folds': args.blocked_folds, 'purge_frames': args.purge_frames, 'null_shifts_seed_repeat_zero': args.null_shifts, 'null_block_size': args.null_block_size, 'participant_folds': args.participant_folds, 'participant_split_seed': args.participant_split_seed, 'silhouette_max_samples': args.silhouette_max_samples}

def job_input_signature(job: dict, run_manifest: pd.DataFrame, volume_root: Path) -> dict:
    if job['analysis'] == 'cross_session':
        subjects = [job['subject']]
        sessions = [job['train_session'], job['test_session']]
    elif job['analysis'] == 'participant_heldout':
        subjects = sorted(job['train_subjects'] + job['test_subjects'])
        sessions = [job['session']]
    else:
        subjects = [job['subject']]
        sessions = [job['session']]
    local = run_manifest[run_manifest.subject.isin(subjects) & run_manifest.session.isin(sessions)].sort_values(['subject', 'session', 'task_order'])
    sources = []
    for path_text in local.roi_path:
        path = volume_root / path_text
        stat = path.stat()
        sources.append({'archive_id': str(path_text), 'size': int(stat.st_size)})
    return {'variant': job['variant'], 'task_order': list(TASKS), 'sources': sources}

def completed_result_reusable(result_path: Path, expected_analysis_signature: dict, expected_input_signature: dict) -> bool:
    if not result_path.exists():
        return False
    try:
        existing = json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    artifact_id = existing.get('split_indices_id')
    artifact_path = result_path.parent.parent / artifact_id if artifact_id else None
    reusable = bool(existing.get('status') == 'completed' and existing.get('analysis_signature') == expected_analysis_signature and (existing.get('input_signature') == expected_input_signature) and artifact_path and artifact_path.exists())
    if not reusable:
        return False
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            if not archive.files:
                return False
            for name in archive.files:
                archive[name]
    except (OSError, ValueError):
        return False
    return True

def run_one_job(job: dict, args, run_manifest: pd.DataFrame, config: CebraConfig, *, expected_input_signature: dict | None=None) -> dict:
    started = time.monotonic()
    expected_input_signature = expected_input_signature or job_input_signature(job, run_manifest, args.volume_root)
    seed_audit = set_all_seeds(job['seed'])
    if job['analysis'] == 'author_like':
        payload, split_arrays = run_author_like(job, args, run_manifest, config)
    elif job['analysis'] == 'blocked':
        payload, split_arrays = run_blocked(job, args, run_manifest, config)
    elif job['analysis'] == 'cross_session':
        payload, split_arrays = run_cross_session(job, args, run_manifest, config)
    elif job['analysis'] == 'participant_heldout':
        payload, split_arrays = run_participant_heldout(job, args, run_manifest, config)
    else:
        raise ValueError(job['analysis'])
    key = job_key(job)
    split_path = args.output_dir / 'split_indices' / f'{key}.npz'
    atomic_npz(split_path, split_arrays)
    contains_embedding = any(('embedding' in name for name in split_arrays))
    split_id = split_path.relative_to(args.output_dir).as_posix()
    result = {'status': 'completed', 'job_key': key, 'job': job, 'backend': 'cebra', 'analysis_signature': analysis_signature(args, config), 'input_signature': expected_input_signature, 'cebra_configuration': config.as_dict(), 'input_preprocessing': args.preprocessing, 'svm_configuration': {'kernel': 'linear', 'C': args.svm_c, 'class_weight': 'balanced', 'embedding_scaler_fit_inside_svm_training_boundary': True}, 'seed_audit': seed_audit, 'split_indices_id': split_id, 'embedding_artifact_id': split_id if contains_embedding else None, 'embedding_artifact_arrays': [name for name in split_arrays if 'embedding' in name], 'runtime_seconds': time.monotonic() - started, **payload}
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True, type=Path)
    parser.add_argument('--volume-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--analysis', choices=['author_like', 'blocked', 'cross_session', 'participant_heldout', 'all'], default='all')
    parser.add_argument('--cohort', choices=['all_complete', 'modal_length', 'paper_size_low_fd_proxy'], default='modal_length')
    parser.add_argument('--variants', nargs='+', default=['author_literal'])
    parser.add_argument('--base-seed', type=int, default=20260826)
    parser.add_argument('--author-seeds', type=int, default=10)
    parser.add_argument('--blocked-seeds', type=int, default=3)
    parser.add_argument('--transfer-seeds', type=int, default=3)
    parser.add_argument('--participant-seeds', type=int, default=3)
    parser.add_argument('--blocked-folds', type=int, default=4)
    parser.add_argument('--participant-folds', type=int, default=5)
    parser.add_argument('--participant-split-seed', type=int, default=20260826)
    parser.add_argument('--purge-frames', type=int, default=50)
    parser.add_argument('--null-shifts', type=int, default=20)
    parser.add_argument('--null-block-size', type=int, default=50)
    parser.add_argument('--random-splits', type=int, default=5)
    parser.add_argument('--test-fraction', type=float, default=0.25)
    parser.add_argument('--svm-c', type=float, default=1.0)
    parser.add_argument('--silhouette-max-samples', type=int, default=800)
    parser.add_argument('--model-architecture', default='offset10-model')
    parser.add_argument('--preprocessing', choices=['none', 'train_zscore'], default='none', help='ROI preprocessing before CEBRA; train_zscore is fit only within the training boundary')
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
    parser.add_argument('--session-cache-dir', type=Path, required=True, help='deletable system-disk cache for concatenated session .npy arrays')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--list-jobs', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('shard-index must be in [0, shard-count)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.session_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = pd.read_parquet(args.manifest_dir / 'cebra_run_manifest.parquet')
    membership = pd.read_parquet(args.manifest_dir / 'cebra_cohort_membership.parquet')
    jobs = build_jobs(args, membership)
    shard_jobs = [job for job in jobs if job['job_index'] % args.shard_count == args.shard_index]
    config = CebraConfig(model_architecture=args.model_architecture, time_offsets=args.time_offsets, output_dimension=args.output_dimension, max_iterations=args.max_iterations, batch_size=args.batch_size, learning_rate=args.learning_rate, temperature=args.temperature, temperature_mode=args.temperature_mode, min_temperature=args.min_temperature, distance=args.distance, num_hidden_units=args.num_hidden_units, device=args.device)
    expected_analysis_signature = analysis_signature(args, config)
    input_signatures: dict[str, dict] = {}
    selected_jobs = list(shard_jobs)
    if args.max_jobs is not None:
        if args.overwrite:
            selected_jobs = selected_jobs[:args.max_jobs]
        else:
            pending_jobs = []
            for job in selected_jobs:
                key = job_key(job)
                expected_input = job_input_signature(job, run_manifest, args.volume_root)
                input_signatures[key] = expected_input
                reusable = completed_result_reusable(cache_dir / f'{key}.json', expected_analysis_signature, expected_input)
                if reusable:
                    failure_path = cache_dir / f'{key}.failed.json'
                    if failure_path.exists():
                        failure_path.unlink()
                else:
                    pending_jobs.append(job)
            selected_jobs = pending_jobs[:args.max_jobs]
    plan = {'total_jobs_all_shards': len(jobs), 'jobs_this_shard': len(shard_jobs), 'jobs_selected_this_invocation': len(selected_jobs), 'shard_index': args.shard_index, 'shard_count': args.shard_count, 'analysis_counts_all_shards': pd.Series([job['analysis'] for job in jobs]).value_counts().to_dict(), 'optimizer_steps_all_shards': len(jobs) * config.max_iterations, 'analysis_signature': expected_analysis_signature, 'job_keys_this_shard': [job_key(job) for job in shard_jobs], 'jobs': selected_jobs if args.list_jobs else None}
    atomic_json(args.output_dir / f'job_plan_shard-{args.shard_index:02d}.json', plan)
    print(json.dumps({key: value for key, value in plan.items() if key != 'jobs'}))
    if args.list_jobs:
        return 0
    completed = 0
    skipped = 0
    failed = 0
    for ordinal, job in enumerate(selected_jobs, start=1):
        key = job_key(job)
        result_path = cache_dir / f'{key}.json'
        failure_path = cache_dir / f'{key}.failed.json'
        expected_input_signature = input_signatures.get(key) or job_input_signature(job, run_manifest, args.volume_root)
        if not args.overwrite:
            if completed_result_reusable(result_path, expected_analysis_signature, expected_input_signature):
                if failure_path.exists():
                    failure_path.unlink()
                skipped += 1
                continue
            if result_path.exists():
                print(f'[{ordinal}/{len(selected_jobs)}] STALE {key}; rerunning', flush=True)
        print(f'[{ordinal}/{len(selected_jobs)}] START {key}', flush=True)
        try:
            result = run_one_job(job, args, run_manifest, config, expected_input_signature=expected_input_signature)
            atomic_json(result_path, result)
            if failure_path.exists():
                failure_path.unlink()
            completed += 1
            print(f'[{ordinal}/{len(selected_jobs)}] DONE {key} seconds={result['runtime_seconds']:.3f}', flush=True)
        except Exception as exc:
            atomic_json(failure_path, {'status': 'failed', 'job_key': key, 'job': job, 'analysis_signature': expected_analysis_signature, 'input_signature': expected_input_signature, 'error_type': type(exc).__name__})
            failed += 1
            print(f'[{ordinal}/{len(selected_jobs)}] FAILED {key}: {type(exc).__name__}', flush=True)
    print(json.dumps({'completed': completed, 'skipped': skipped, 'failed': failed, 'selected': len(selected_jobs)}))
    return 1 if failed else 0
if __name__ == '__main__':
    raise SystemExit(main())
