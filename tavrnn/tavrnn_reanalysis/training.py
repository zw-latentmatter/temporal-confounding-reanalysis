from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, VariantConfig
from .data import load_fc_sequence
from .graph import build_graph_sequence
from .io import atomic_csv, atomic_json, atomic_npz, atomic_torch_save, read_json
from .model import build_model
from .randomness import capture_rng_state, restore_rng_state, seed_everything


def _plain_spec(row: pd.Series | dict) -> dict:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    result = {}
    for key, value in source.items():
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _checkpoint(
    path: Path,
    model,
    optimizer,
    generator: torch.Generator,
    epoch: int,
    best_loss: float,
    history: list[dict],
) -> None:
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "rng": capture_rng_state(),
            "epoch": epoch,
            "best_loss": best_loss,
            "history": history,
        },
    )


def execute_run(
    row: pd.Series | dict,
    output_root: Path,
    resume: bool = True,
    checkpoint_interval: int = 25,
    device_override: str | None = None,
) -> str:
    spec = _plain_spec(row)
    run_dir = output_root / "runs" / str(spec["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    embeddings_path = run_dir / "embeddings.npz"
    if state_path.exists() and embeddings_path.exists():
        state = read_json(state_path)
        if state.get("status") == "completed":
            return "skipped_completed"
    device_name = device_override or str(spec["requested_device"])
    if device_name == "cuda":
        device_name = "cuda:0"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested run")
    started = time.monotonic()
    state = {
        "status": "running",
        "run_id": str(spec["run_id"]),
        "subject": str(spec["subject"]),
        "session": str(spec["session"]),
        "variant_id": str(spec["variant_id"]),
        "seed": int(spec["seed"]),
        "epochs": int(spec["epochs"]),
    }
    atomic_json(state_path, state)
    try:
        seed = int(spec["seed"])
        seed_everything(seed)
        variant = VariantConfig(
            variant_id=str(spec["variant_id"]),
            feature_mode=str(spec["feature_mode"]),
            topology_score=str(spec["topology_score"]),
            density=float(spec["density"]),
        )
        fc = load_fc_sequence(
            str(spec["input_paths_json"]),
            str(spec["fc_key"]),
            int(spec["expected_nodes"]),
        )
        graph = build_graph_sequence(fc, variant).to(device)
        config = ModelConfig(
            x_dim=graph.features.shape[-1],
            h_dim=32,
            z_dim=min(8, graph.features.shape[-1]),
            n_layers=1,
            eps=1e-10,
            bias=True,
            attention_width=3,
            loss_mode=str(spec["loss_mode"]),
        )
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 100000)
        latest_path = run_dir / "checkpoint_latest.pt"
        best_path = run_dir / "checkpoint_best.pt"
        start_epoch = 0
        best_loss = float("inf")
        history = []
        if resume and latest_path.exists():
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            model.load_state_dict(latest["model"])
            optimizer.load_state_dict(latest["optimizer"])
            _optimizer_to(optimizer, device)
            generator.set_state(latest["generator"].cpu())
            restore_rng_state(latest["rng"])
            start_epoch = int(latest["epoch"])
            best_loss = float(latest["best_loss"])
            history = list(latest["history"])
        model.train()
        for epoch in range(start_epoch, int(spec["epochs"])):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                graph.features,
                graph.edge_indices,
                graph.targets,
                latent_mode="sample",
                generator=generator,
            )
            output.loss.backward()
            optimizer.step()
            loss_value = float(output.loss.detach().cpu())
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "kld": float(output.kld_loss.detach().cpu()),
                    "reconstruction": float(output.reconstruction_loss.detach().cpu()),
                }
            )
            if loss_value < best_loss:
                best_loss = loss_value
                _checkpoint(
                    best_path,
                    model,
                    optimizer,
                    generator,
                    epoch + 1,
                    best_loss,
                    history,
                )
            if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == int(
                spec["epochs"]
            ):
                _checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    generator,
                    epoch + 1,
                    best_loss,
                    history,
                )
                state["epoch_completed"] = epoch + 1
                atomic_json(state_path, state)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        model.eval()
        with torch.no_grad():
            evaluation = model(
                graph.features,
                graph.edge_indices,
                graph.targets,
                latent_mode="mean",
            )
        embeddings = evaluation.encoder_means.detach().cpu().numpy()
        atomic_npz(
            embeddings_path,
            embeddings=embeddings.astype(np.float32),
            task_order=np.asarray(str(spec["task_order"]).split("|")),
        )
        atomic_csv(run_dir / "training_history.csv", pd.DataFrame(history))
        atomic_json(
            run_dir / "run_summary.json",
            {
                "run_id": str(spec["run_id"]),
                "best_epoch": int(best["epoch"]),
                "best_training_loss": float(best["best_loss"]),
                "evaluation_kld_loss": float(evaluation.kld_loss.cpu()),
                "evaluation_reconstruction_loss": float(
                    evaluation.reconstruction_loss.cpu()
                ),
            },
        )
        state.update(
            {
                "status": "completed",
                "epoch_completed": int(spec["epochs"]),
                "runtime_seconds": time.monotonic() - started,
            }
        )
        atomic_json(state_path, state)
        return "completed"
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "runtime_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
            }
        )
        atomic_json(state_path, state)
        raise
