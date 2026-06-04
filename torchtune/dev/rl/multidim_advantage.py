# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-dimensional group-relative advantages for GRPO.

Adapted from *General Preference Reinforcement Learning* (GPRL),
https://arxiv.org/abs/2605.18721. Standard GRPO collapses a vector of
per-reward-function scores into a single scalar (typically by summing over
reward functions) *before* computing the group-relative advantage. Because a
scalar score is an incomplete proxy for multi-dimensional quality, the policy
can collapse onto whichever axis the scalar is most sensitive to -- i.e. reward
hacking along a single dimension.

GPRL instead carries the per-dimension structure through to the policy update:
each reward dimension is normalized *on its own scale* within the GRPO group so
that no axis can dominate purely because of its raw magnitude, and the
normalized advantages are then aggregated across dimensions. This module
provides that computation as a drop-in replacement for the scalar advantage
step, plus a cheap single-axis-exploitation diagnostic that mirrors GPRL's
drift monitor.
"""

from typing import NamedTuple, Optional

import torch

__all__ = [
    "AdvantageDiagnostics",
    "group_relative_advantages",
]


class AdvantageDiagnostics(NamedTuple):
    """Per-dimension diagnostics for the GPRL drift monitor.

    Attributes:
        per_dimension_advantages (torch.Tensor): normalized advantages with
            shape ``[B, G, D]`` (before aggregation across dimensions).
        dimension_weights (torch.Tensor): the weights applied to each of the
            ``D`` dimensions, shape ``[D]``.
        dominance (torch.Tensor): scalar in ``[1 / D, 1]`` giving the fraction
            of total advantage magnitude carried by the single most influential
            dimension. Values near ``1`` indicate single-axis exploitation;
            values near ``1 / D`` indicate balanced use of all axes.
    """

    per_dimension_advantages: torch.Tensor
    dimension_weights: torch.Tensor
    dominance: torch.Tensor


def _normalize_within_group(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-mean, unit-std normalization over the GRPO group dimension (dim=1).

    Args:
        rewards (torch.Tensor): rewards with shape ``[B, G, D]``.
        eps (float): numerical stabilizer added to the per-group std.

    Returns:
        torch.Tensor: group-relative advantages with shape ``[B, G, D]``.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def group_relative_advantages(
    rewards: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    return_diagnostics: bool = False,
):
    """Compute per-dimension group-relative advantages and aggregate them.

    This is the GPRL replacement for the standard GRPO advantage step. Given a
    reward tensor with one score per reward dimension, each dimension is
    normalized independently across the GRPO group so the dimensions live on a
    common scale, and the result is aggregated with ``weights``.

    Args:
        rewards (torch.Tensor): per-dimension rewards. Accepts ``[B, G, D]``
            (one column per reward function) or ``[B, G]`` (already scalar, in
            which case ``D`` is treated as 1 and behavior matches standard
            GRPO). ``B`` is the prompt batch, ``G`` the GRPO group size.
        weights (Optional[torch.Tensor]): non-negative per-dimension weights of
            shape ``[D]``. They are renormalized to sum to 1. Defaults to a
            uniform average over dimensions (GPRL's context-dependent
            eigenvalue weighting is intentionally out of scope here -- uniform
            normalization already removes the raw-magnitude domination that
            scalar GRPO suffers from).
        eps (float): numerical stabilizer for the per-dimension std. Matches the
            ``1e-4`` used by the existing scalar GRPO advantage step.
        return_diagnostics (bool): if ``True``, also return an
            :class:`AdvantageDiagnostics` for the drift monitor.

    Returns:
        torch.Tensor: aggregated advantages flattened to shape ``[B * G]``,
        ready to drop into ``GRPOTrajectory.advantages``. If
        ``return_diagnostics`` is ``True``, returns a tuple of
        ``(advantages, AdvantageDiagnostics)``.

    Raises:
        ValueError: if ``rewards`` does not have 2 or 3 dimensions, or if
            ``weights`` has the wrong shape.
    """
    if rewards.dim() == 2:
        rewards = rewards.unsqueeze(-1)  # [B, G] -> [B, G, 1]
    elif rewards.dim() != 3:
        raise ValueError(
            f"rewards must have shape [B, G] or [B, G, D], got {tuple(rewards.shape)}"
        )

    batch_size, grpo_size, num_dims = rewards.shape

    # Per-dimension group-relative normalization: each axis on its own scale.
    per_dim = _normalize_within_group(rewards, eps)  # [B, G, D]

    if weights is None:
        weights = rewards.new_full((num_dims,), 1.0 / num_dims)
    else:
        weights = weights.to(device=rewards.device, dtype=rewards.dtype)
        if weights.shape != (num_dims,):
            raise ValueError(
                f"weights must have shape ({num_dims},), got {tuple(weights.shape)}"
            )
        weights = weights / (weights.sum() + eps)

    advantages = (per_dim * weights).sum(dim=-1)  # [B, G]
    advantages = advantages.reshape(batch_size * grpo_size)

    if not return_diagnostics:
        return advantages

    diagnostics = AdvantageDiagnostics(
        per_dimension_advantages=per_dim,
        dimension_weights=weights,
        dominance=_single_axis_dominance(per_dim, weights),
    )
    return advantages, diagnostics


def _single_axis_dominance(
    per_dim_advantages: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Fraction of total advantage magnitude carried by the top dimension.

    A cheap proxy for GPRL's drift monitor: it summarizes how concentrated the
    weighted advantage signal is on a single axis. ``1.0`` means one dimension
    accounts for all of the magnitude (single-axis exploitation); ``1 / D``
    means perfectly balanced.

    Args:
        per_dim_advantages (torch.Tensor): normalized advantages ``[B, G, D]``.
        weights (torch.Tensor): per-dimension weights ``[D]``.

    Returns:
        torch.Tensor: scalar dominance value.
    """
    # Mean absolute weighted contribution per dimension across the batch/group.
    contribution = (per_dim_advantages * weights).abs().mean(dim=(0, 1))  # [D]
    total = contribution.sum()
    if total <= 0:
        # No signal in any dimension; report balanced usage.
        num_dims = per_dim_advantages.shape[-1]
        return per_dim_advantages.new_tensor(1.0 / num_dims)
    return contribution.max() / total
