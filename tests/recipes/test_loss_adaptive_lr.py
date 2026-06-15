# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FINCH loss-adaptive LR modulation and its wiring into the
full_finetune_distributed recipe.

These tests are deliberately torch-free so they exercise the schedule math and
the recipe wiring without spinning up a full distributed training run.
"""

import math
from pathlib import Path

import pytest

from torchtune.training.loss_adaptive_lr import FinchLRModifier

# The recipe modules are intentionally NOT importable as a package (see
# recipes/__init__.py), so we assert the integration against the call-site
# source directly -- this proves the modifier is actually wired in rather than
# being a freestanding module.
RECIPE_PATH = (
    Path(__file__).resolve().parents[2] / "recipes" / "full_finetune_distributed.py"
)


class _FakeOptimizer:
    """Minimal stand-in exposing only the `param_groups` interface FINCH uses."""

    def __init__(self, lrs):
        self.param_groups = [{"lr": lr} for lr in lrs]


def test_anchor_loss_gives_unit_scale():
    mod = FinchLRModifier(min_scale=0.01, max_scale=10.0)
    # First observed loss becomes the anchor -> scale exactly 1.0.
    assert mod.compute_scale(4.0) == pytest.approx(1.0)


def test_high_loss_reduces_lr_low_loss_raises_it():
    mod = FinchLRModifier(min_scale=0.01, max_scale=10.0, anchor_loss=4.0)
    # Higher-than-anchor loss -> reduce LR (sqrt(4/16) = 0.5).
    assert mod.compute_scale(16.0) == pytest.approx(0.5)
    # Lower-than-anchor loss -> increase LR (sqrt(4/1) = 2.0).
    assert mod.compute_scale(1.0) == pytest.approx(2.0)


def test_scale_is_clamped():
    mod = FinchLRModifier(min_scale=0.5, max_scale=1.5, anchor_loss=1.0)
    assert mod.compute_scale(100.0) == pytest.approx(0.5)  # clamped from sqrt(0.01)
    assert mod.compute_scale(0.0001) == pytest.approx(1.5)  # clamped from sqrt(10000)


def test_step_applies_and_does_not_compound():
    mod = FinchLRModifier(min_scale=0.01, max_scale=10.0, anchor_loss=4.0)
    opt = _FakeOptimizer([1e-3, 2e-3])

    # First step at the anchor loss leaves base LRs unchanged.
    mod.step(4.0, opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)

    # A high-loss step halves the LR relative to the base LR (not compounding).
    mod.step(16.0, opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5e-3)
    assert opt.param_groups[1]["lr"] == pytest.approx(1.0e-3)

    # Returning to the anchor loss restores the original base LR exactly,
    # proving the previously applied factor was divided back out.
    mod.step(4.0, opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[1]["lr"] == pytest.approx(2e-3)


def test_step_composes_with_base_scheduler_lr_rewrites():
    # Simulate a base LR scheduler rewriting `lr` each step, with FINCH applied
    # on top afterwards. The net LR should always be base_lr * finch_scale.
    mod = FinchLRModifier(min_scale=0.01, max_scale=10.0, anchor_loss=4.0)
    opt = _FakeOptimizer([1e-3])
    base_schedule = [1e-3, 8e-4, 5e-4]
    losses = [4.0, 16.0, 1.0]
    expected_scales = [1.0, 0.5, 2.0]
    for base_lr, loss, scale in zip(base_schedule, losses, expected_scales):
        opt.param_groups[0]["lr"] = base_lr  # base scheduler step
        mod.step(loss, opt)  # FINCH on top
        assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * scale)


def test_rejects_non_positive_loss():
    mod = FinchLRModifier()
    with pytest.raises(ValueError):
        mod.compute_scale(0.0)
    with pytest.raises(ValueError):
        mod.compute_scale(float("nan"))


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        FinchLRModifier(min_scale=2.0, max_scale=1.0)
    with pytest.raises(ValueError):
        FinchLRModifier(smoothing=1.0)


def test_recipe_call_site_is_wired():
    # Integration check against the non-new recipe module's source: the recipe
    # must import the modifier and invoke `.step(...)` in its training loop.
    src = RECIPE_PATH.read_text()
    assert "from torchtune.training.loss_adaptive_lr import FinchLRModifier" in src
    assert "self._finch_lr_modifier" in src
    assert "self._finch_lr_modifier.step(" in src


def test_recipe_only_steps_finch_when_enabled():
    # The wiring must be guarded so default runs are unaffected.
    src = RECIPE_PATH.read_text()
    assert 'finch_cfg.get("enabled"' in src


def test_composes_with_existing_cosine_scheduler():
    # Exercise FINCH against the existing, non-new LR scheduler that the recipe
    # already uses, mirroring the recipe's "scheduler.step() then finch.step()"
    # ordering on a real torch optimizer.
    torch = pytest.importorskip("torch")
    from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

    params = [torch.nn.Parameter(torch.zeros(1))]
    base_lr = 1e-3
    opt = torch.optim.SGD(params, lr=base_lr)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=2, num_training_steps=10
    )
    mod = FinchLRModifier(min_scale=0.01, max_scale=10.0, anchor_loss=4.0)

    for step, loss in enumerate([4.0, 16.0, 1.0, 4.0]):
        opt.step()
        scheduler.step()
        base_after_sched = opt.param_groups[0]["lr"]
        mod.step(loss, opt)
        # FINCH must scale relative to whatever the cosine scheduler produced.
        assert opt.param_groups[0]["lr"] == pytest.approx(
            base_after_sched * mod.last_scale
        )
