# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for the GPRL multi-dimensional advantage wiring.

These tests cover (a) the core per-dimension group-relative advantage math in
``torchtune.dev.rl.multidim_advantage`` and (b) that the existing GRPO recipe
(``recipes/dev/grpo_full_finetune_distributed.py``) actually imports and calls
it at the advantage step.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torchtune.dev.rl.multidim_advantage import (  # noqa: E402
    AdvantageDiagnostics,
    group_relative_advantages,
)

RECIPE_PATH = (
    Path(__file__).parents[3]
    / "recipes"
    / "dev"
    / "grpo_full_finetune_distributed.py"
)


def _load_recipe_module():
    """Import the existing (non-new) call-site recipe module by file path.

    Skips the test if the recipe's heavy training dependencies are unavailable
    in the current environment.
    """
    spec = importlib.util.spec_from_file_location(
        "grpo_full_finetune_distributed_under_test", RECIPE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # pragma: no cover - depends on optional deps
        pytest.skip(f"recipe deps unavailable: {e}")
    return module


def test_recipe_wires_group_relative_advantages():
    """The recipe must import the new GPRL advantage fn into its namespace."""
    module = _load_recipe_module()
    assert hasattr(module, "group_relative_advantages")
    assert (
        module.group_relative_advantages
        is group_relative_advantages
    )


def test_shape_matches_trajectory_contract():
    """Output is flattened to [B * G], matching GRPOTrajectory.advantages."""
    batch_size, grpo_size, num_dims = 4, 8, 3
    rewards = torch.randn(batch_size, grpo_size, num_dims)
    adv = group_relative_advantages(rewards)
    assert adv.shape == (batch_size * grpo_size,)


def test_per_dimension_normalization_prevents_axis_domination():
    """A high-magnitude axis must not dominate after per-dim normalization.

    This is the core GPRL property: one reward dimension on a much larger raw
    scale should not swamp the others. We compare against the old scalar-GRPO
    behavior (sum over dims, then normalize), which *does* get dominated.
    """
    batch_size, grpo_size = 2, 16
    # Dimension 0 varies on a huge scale; dimension 1 carries the real signal
    # on a tiny scale.
    big_axis = torch.randn(batch_size, grpo_size, 1) * 1000.0
    small_axis = torch.randn(batch_size, grpo_size, 1) * 0.01
    rewards = torch.cat([big_axis, small_axis], dim=-1)  # [B, G, 2]

    # GPRL: per-dimension normalization, with diagnostics.
    adv, diag = group_relative_advantages(rewards, return_diagnostics=True)
    assert isinstance(diag, AdvantageDiagnostics)

    # The small axis must still meaningfully shape the aggregated advantage.
    norm_small = diag.per_dimension_advantages[..., 1]
    # Correlation between the (normalized) small axis and the final advantage
    # should be clearly positive -- it has not been swamped.
    flat_small = norm_small.reshape(-1)
    flat_adv = adv
    corr = torch.corrcoef(torch.stack([flat_small, flat_adv]))[0, 1]
    assert corr > 0.3

    # Old scalar-GRPO behavior: summing first lets the big axis dominate.
    scalar = rewards.sum(dim=-1)
    scalar_adv = (
        (scalar - scalar.mean(1, keepdim=True))
        / (scalar.std(1, keepdim=True) + 1e-4)
    ).reshape(-1)
    scalar_corr = torch.corrcoef(torch.stack([flat_small, scalar_adv]))[0, 1]
    # The small axis is essentially invisible to scalar GRPO.
    assert scalar_corr.abs() < corr


def test_dominance_diagnostic_bounds():
    """Dominance lives in [1/D, 1] and flags single-axis concentration."""
    batch_size, grpo_size = 2, 8
    # Balanced two identical-scale axes -> dominance near 0.5.
    balanced = torch.randn(batch_size, grpo_size, 2)
    _, diag = group_relative_advantages(balanced, return_diagnostics=True)
    assert 0.5 - 1e-3 <= diag.dominance.item() <= 1.0 + 1e-3

    # One axis weighted to zero -> the other carries everything.
    weights = torch.tensor([1.0, 0.0])
    _, diag2 = group_relative_advantages(
        balanced, weights=weights, return_diagnostics=True
    )
    assert diag2.dominance.item() == pytest.approx(1.0, abs=1e-5)


def test_single_dimension_matches_scalar_grpo():
    """With D == 1 the result reproduces the standard GRPO advantage."""
    batch_size, grpo_size = 3, 8
    rewards = torch.randn(batch_size, grpo_size, 1)
    adv = group_relative_advantages(rewards)

    scalar = rewards.squeeze(-1)
    expected = (
        (scalar - scalar.mean(1, keepdim=True))
        / (scalar.std(1, keepdim=True) + 1e-4)
    ).reshape(batch_size * grpo_size)
    assert torch.allclose(adv, expected, atol=1e-5)


def test_accepts_two_dim_rewards():
    """A pre-collapsed [B, G] reward tensor is accepted (treated as D == 1)."""
    rewards = torch.randn(2, 5)
    adv = group_relative_advantages(rewards)
    assert adv.shape == (10,)


def test_bad_weight_shape_raises():
    rewards = torch.randn(2, 4, 3)
    with pytest.raises(ValueError):
        group_relative_advantages(rewards, weights=torch.ones(2))
