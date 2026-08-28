from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import VariantConfig


@dataclass
class GraphSequence:
    features: torch.Tensor
    edge_indices: list[torch.Tensor]
    targets: list[torch.Tensor]

    def to(self, device: torch.device | str) -> GraphSequence:
        return GraphSequence(
            features=self.features.to(device),
            edge_indices=[edge.to(device) for edge in self.edge_indices],
            targets=[target.to(device) for target in self.targets],
        )


def _validate_fc_sequence(fc: np.ndarray) -> np.ndarray:
    fc = np.asarray(fc)
    if fc.ndim != 3 or fc.shape[0] != 4 or fc.shape[1] != fc.shape[2]:
        raise ValueError(f"expected [4,N,N] FC sequence, got {fc.shape}")
    if not np.isfinite(fc).all():
        raise ValueError("FC sequence contains non-finite values")
    if not np.allclose(fc, np.swapaxes(fc, 1, 2), atol=1e-5, rtol=1e-5):
        raise ValueError("FC matrices must be symmetric")
    if not np.allclose(np.diagonal(fc, axis1=1, axis2=2), 0.0, atol=1e-6):
        raise ValueError("FC matrices must have a zero diagonal")
    return fc.astype(np.float32, copy=False)


def _feature_matrix(fc: np.ndarray, mode: str) -> np.ndarray:
    if mode == "signed":
        return fc.copy()
    if mode == "positive":
        return np.maximum(fc, 0.0)
    if mode == "absolute":
        return np.abs(fc)
    raise ValueError(f"unknown feature mode: {mode}")


def fixed_density_adjacency(
    fc: np.ndarray, density: float, topology_score: str
) -> np.ndarray:
    if not 0.0 < density < 1.0:
        raise ValueError(f"density must be in (0,1), got {density}")
    n_nodes = fc.shape[0]
    rows, cols = np.triu_indices(n_nodes, k=1)
    n_select = int(np.floor(density * rows.size))
    if n_select < 1:
        raise ValueError("density selects no edges")
    raw = fc[rows, cols]
    if topology_score == "absolute":
        score = np.abs(raw)
        eligible = np.ones(raw.shape, dtype=bool)
    elif topology_score == "positive":
        score = raw
        eligible = raw > 0
        if int(eligible.sum()) < n_select:
            raise ValueError(
                f"only {int(eligible.sum())} positive edges available for fixed K={n_select}"
            )
    else:
        raise ValueError(f"unknown topology score: {topology_score}")
    eligible_index = np.flatnonzero(eligible)
    order = np.lexsort(
        (cols[eligible_index], rows[eligible_index], -score[eligible_index])
    )
    chosen = eligible_index[order[:n_select]]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adjacency[rows[chosen], cols[chosen]] = 1.0
    adjacency[cols[chosen], rows[chosen]] = 1.0
    return adjacency


def adjacency_to_edge_index(adjacency: np.ndarray) -> torch.Tensor:
    source, target = np.nonzero(adjacency)
    return torch.from_numpy(np.vstack([source, target]).astype(np.int64))


def build_graph_sequence(fc: np.ndarray, variant: VariantConfig) -> GraphSequence:
    signed_fc = _validate_fc_sequence(fc)
    features = []
    edge_indices = []
    targets = []
    for snapshot in signed_fc:
        adjacency = fixed_density_adjacency(
            snapshot, variant.density, variant.topology_score
        )
        features.append(_feature_matrix(snapshot, variant.feature_mode))
        edge_indices.append(adjacency_to_edge_index(adjacency))
        targets.append(torch.from_numpy(adjacency))
    return GraphSequence(
        features=torch.from_numpy(np.stack(features).astype(np.float32)),
        edge_indices=edge_indices,
        targets=targets,
    )
