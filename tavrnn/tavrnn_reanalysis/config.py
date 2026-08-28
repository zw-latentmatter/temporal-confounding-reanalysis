from __future__ import annotations

from dataclasses import asdict, dataclass


CANONICAL_TASKS = ("rest", "meditation", "music", "movie")


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    feature_mode: str
    topology_score: str
    density: float

    def as_dict(self) -> dict:
        return asdict(self)


CORE_VARIANTS = (
    VariantConfig("signed-d10", "signed", "absolute", 0.10),
    VariantConfig("signed-d05", "signed", "absolute", 0.05),
    VariantConfig("signed-d20", "signed", "absolute", 0.20),
    VariantConfig("positive-d10", "positive", "positive", 0.10),
    VariantConfig("absolute-d10", "absolute", "absolute", 0.10),
)


@dataclass(frozen=True)
class ModelConfig:
    x_dim: int
    h_dim: int = 32
    z_dim: int = 8
    n_layers: int = 1
    eps: float = 1e-10
    bias: bool = True
    attention_width: int = 3
    loss_mode: str = "notebook_full_matrix"

    def as_dict(self) -> dict:
        return asdict(self)
