from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value


class PlainGCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        edge_weight = torch.ones(source.numel(), dtype=x.dtype, device=x.device)
        degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        degree.index_add_(0, source, edge_weight)
        degree_inverse = degree.pow(-0.5)
        degree_inverse.masked_fill_(torch.isinf(degree_inverse), 0.0)
        norm = degree_inverse[source] * edge_weight * degree_inverse[target]
        transformed = x @ self.weight
        messages = norm[:, None] * transformed[source]
        output = torch.zeros(
            (x.shape[0], self.out_channels), dtype=x.dtype, device=x.device
        )
        output.index_add_(0, target, messages)
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output)


class NotebookAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, all_hidden: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        final_layer = torch.stack([tensor[-1] for tensor in all_hidden], dim=0)
        weights_at_step = []
        if final_layer.shape[0] >= 2:
            query = self.query(final_layer[-1, -1].reshape(1, -1))
            keys = self.key(final_layer[1:, -1])
            scores = query @ keys.transpose(-2, -1) / math.sqrt(self.hidden_size)
            weights = F.softmax(scores, dim=1)
            weights_at_step.append(weights)
            values = self.value(torch.stack(all_hidden[1:], dim=0))
            expanded = weights.reshape(1, weights.shape[1], 1, 1, 1)
            all_hidden[-1] = torch.sum(values * expanded, dim=1).squeeze(0)
        return all_hidden, weights_at_step


class GraphGRUAttention(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, n_layers: int, bias: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.attention = NotebookAttention(hidden_size)
        layer_inputs = [input_size] + [hidden_size] * (n_layers - 1)
        self.weight_xz = nn.ModuleList(
            [PlainGCNConv(size, hidden_size, _identity, bias) for size in layer_inputs]
        )
        self.weight_hz = nn.ModuleList(
            [PlainGCNConv(hidden_size, hidden_size, _identity, bias) for _ in layer_inputs]
        )
        self.weight_xr = nn.ModuleList(
            [PlainGCNConv(size, hidden_size, _identity, bias) for size in layer_inputs]
        )
        self.weight_hr = nn.ModuleList(
            [PlainGCNConv(hidden_size, hidden_size, _identity, bias) for _ in layer_inputs]
        )
        self.weight_xh = nn.ModuleList(
            [PlainGCNConv(size, hidden_size, _identity, bias) for size in layer_inputs]
        )
        self.weight_hh = nn.ModuleList(
            [PlainGCNConv(hidden_size, hidden_size, _identity, bias) for _ in layer_inputs]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        all_hidden: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        hidden_hat = torch.zeros_like(all_hidden[-1])
        for layer in range(self.n_layers):
            layer_input = inputs if layer == 0 else hidden_hat[layer - 1]
            previous = all_hidden[-1][layer]
            update = torch.sigmoid(
                self.weight_xz[layer](layer_input, edge_index)
                + self.weight_hz[layer](previous, edge_index)
            )
            reset = torch.sigmoid(
                self.weight_xr[layer](layer_input, edge_index)
                + self.weight_hr[layer](previous, edge_index)
            )
            candidate = torch.tanh(
                self.weight_xh[layer](layer_input, edge_index)
                + self.weight_hh[layer](reset * previous, edge_index)
            )
            hidden_hat[layer] = update * previous + (1.0 - update) * candidate
        all_hidden.append(hidden_hat)
        all_hidden, attention_weights = self.attention(all_hidden)
        return all_hidden, all_hidden[-1], attention_weights


@dataclass
class TAVRNNOutput:
    kld_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    encoder_means: torch.Tensor
    logits: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return self.kld_loss + self.reconstruction_loss


class TAVRNN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.phi_x = nn.Sequential(nn.Linear(config.x_dim, config.h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(config.z_dim, config.h_dim), nn.ReLU())
        self.encoder = PlainGCNConv(config.h_dim * 2, config.h_dim)
        self.encoder_mean = PlainGCNConv(config.h_dim, config.z_dim, _identity)
        self.encoder_std = PlainGCNConv(config.h_dim, config.z_dim, F.softplus)
        self.prior = nn.Sequential(nn.Linear(config.h_dim, config.h_dim), nn.ReLU())
        self.prior_mean = nn.Linear(config.h_dim, config.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(config.h_dim, config.z_dim), nn.Softplus()
        )
        self.rnn = GraphGRUAttention(
            config.h_dim * 2, config.h_dim, config.n_layers, config.bias
        )

    def _sample(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        latent_mode: str,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if latent_mode == "mean":
            return mean
        if latent_mode != "sample":
            raise ValueError(f"unknown latent mode: {latent_mode}")
        noise = torch.empty(std.size(), dtype=std.dtype, device="cpu").normal_(
            generator=generator
        )
        return noise.to(std.device) * std + mean

    def _kl(
        self,
        mean_1: torch.Tensor,
        std_1: torch.Tensor,
        mean_2: torch.Tensor,
        std_2: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = mean_1.shape[0]
        element = (
            2.0 * torch.log(std_2 + self.config.eps)
            - 2.0 * torch.log(std_1 + self.config.eps)
            + (
                torch.pow(std_1 + self.config.eps, 2)
                + torch.pow(mean_1 - mean_2, 2)
            )
            / torch.pow(std_2 + self.config.eps, 2)
            - 1.0
        )
        return (0.5 / n_nodes) * torch.mean(torch.sum(element, dim=1), dim=0)

    def _reconstruction_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.config.loss_mode != "notebook_full_matrix":
            raise ValueError(f"unknown loss mode: {self.config.loss_mode}")
        selected_logits = logits.reshape(-1)
        selected_target = target.reshape(-1)
        positives = selected_target.sum()
        total = selected_target.numel()
        negatives = total - positives
        positive_weight = negatives / positives
        norm = total / (negatives * 2.0)
        return norm * F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_target,
            pos_weight=positive_weight,
            reduction="mean",
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: list[torch.Tensor],
        targets: list[torch.Tensor],
        latent_mode: str = "sample",
        generator: torch.Generator | None = None,
    ) -> TAVRNNOutput:
        if len(edge_indices) != len(targets) or len(targets) != x.shape[0]:
            raise ValueError("features, edge indices and targets must have equal T")
        hidden = x.new_zeros((self.config.n_layers, x.shape[1], self.config.h_dim))
        kld_loss = x.new_zeros(())
        reconstruction_loss = x.new_zeros(())
        encoder_means = []
        logits_history = []
        all_hidden = [hidden]
        for time_index in range(x.shape[0]):
            phi_x = self.phi_x(x[time_index])
            encoded = self.encoder(
                torch.cat([phi_x, hidden[-1]], dim=1), edge_indices[time_index]
            )
            encoder_mean = self.encoder_mean(encoded, edge_indices[time_index])
            encoder_std = self.encoder_std(encoded, edge_indices[time_index])
            prior = self.prior(hidden[-1])
            prior_mean = self.prior_mean(prior)
            prior_std = self.prior_std(prior)
            z = self._sample(encoder_mean, encoder_std, latent_mode, generator)
            phi_z = self.phi_z(z)
            logits = z @ z.transpose(0, 1)
            all_hidden, hidden, _ = self.rnn(
                torch.cat([phi_x, phi_z], dim=1), edge_indices[time_index], all_hidden
            )
            kld_loss = kld_loss + self._kl(
                encoder_mean, encoder_std, prior_mean, prior_std
            )
            reconstruction_loss = reconstruction_loss + self._reconstruction_loss(
                logits, targets[time_index]
            )
            encoder_means.append(encoder_mean)
            logits_history.append(logits)
        return TAVRNNOutput(
            kld_loss=kld_loss,
            reconstruction_loss=reconstruction_loss,
            encoder_means=torch.stack(encoder_means),
            logits=torch.stack(logits_history),
        )


def build_model(config: ModelConfig) -> TAVRNN:
    return TAVRNN(config)
