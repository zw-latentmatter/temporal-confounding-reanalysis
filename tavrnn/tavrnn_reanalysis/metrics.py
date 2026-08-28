from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score


NETWORK_ORDER = (
    "subcortical",
    "visual",
    "somatomotor",
    "dorsal_attention",
    "limbic",
    "salience_ventral_attention",
    "default_mode",
    "control",
)

NETWORK_TOKENS = {
    "visual": "_Vis_",
    "somatomotor": "_SomMot_",
    "dorsal_attention": "_DorsAttn_",
    "limbic": "_Limbic_",
    "salience_ventral_attention": "_SalVentAttn_",
    "default_mode": "_Default_",
    "control": "_Cont_",
}


def roi_network_indices(labels: list[str]) -> dict[str, np.ndarray]:
    if len(labels) != 332:
        raise ValueError(f"expected 332 ROI labels, got {len(labels)}")
    networks = {"subcortical": np.arange(32, dtype=np.int64)}
    for network, token in NETWORK_TOKENS.items():
        networks[network] = np.asarray(
            [index for index, label in enumerate(labels) if token in label],
            dtype=np.int64,
        )
    combined = np.concatenate([networks[name] for name in NETWORK_ORDER])
    if len(combined) != 332 or len(np.unique(combined)) != 332:
        raise RuntimeError("ROI labels do not form a complete network partition")
    return networks


def network_embedding_metric_rows(
    embeddings: np.ndarray,
    task_order: tuple[str, ...],
    network_indices: dict[str, np.ndarray],
) -> list[dict]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.shape[:2] != (len(task_order), 332):
        raise ValueError(
            f"expected embeddings [{len(task_order)},332,d], got {embeddings.shape}"
        )
    network_vector = np.empty(332, dtype=np.int8)
    for network_index, network in enumerate(NETWORK_ORDER):
        network_vector[network_indices[network]] = network_index
    rows = []
    for task_index, condition in enumerate(task_order):
        points = embeddings[task_index]
        global_distance = float(pdist(points, metric="euclidean").mean())
        rows.extend(
            [
                {
                    "scope": "condition",
                    "condition_a": condition,
                    "condition_b": "",
                    "metric": "latent_global_mean_pairwise_distance",
                    "value": global_distance,
                },
                {
                    "scope": "condition",
                    "condition_a": condition,
                    "condition_b": "",
                    "metric": "latent_network_silhouette",
                    "value": float(silhouette_score(points, network_vector)),
                },
            ]
        )
        for network in NETWORK_ORDER:
            local = points[network_indices[network]]
            within_distance = float(pdist(local, metric="euclidean").mean())
            rows.extend(
                [
                    {
                        "scope": "condition_network",
                        "condition_a": condition,
                        "condition_b": network,
                        "metric": "latent_within_network_mean_pairwise_distance",
                        "value": within_distance,
                    },
                    {
                        "scope": "condition_network",
                        "condition_a": condition,
                        "condition_b": network,
                        "metric": "latent_within_to_global_distance_ratio",
                        "value": within_distance / global_distance,
                    },
                ]
            )
    return rows
