# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss-adaptive learning-rate modulation (FINCH).

Adapted from "Fine-Tuning Without Forgetting via Loss-Adaptive Learning
Rates" (https://arxiv.org/abs/2605.20005). The paper observes that per-step
catastrophic forgetting during fine-tuning is bounded by the product of the
learning rate and the square root of the current training loss. FINCH exploits
this by *reducing* the learning rate on high-loss batches (which are the most
prone to inducing forgetting) and *increasing* it back as the model converges
and the loss drops -- all while leaving the fine-tuning objective untouched.

This module implements the schedule as a thin multiplier that rides on top of
whatever base learning rate the recipe's existing optimizer / LR scheduler
already produces, so it composes with cosine warmup, constant LR, etc.
"""

import math
from typing import Any, Optional


class FinchLRModifier:
    """Scale an optimizer's learning rate based on the current training loss.

    The schedule keeps the forgetting proxy ``lr * sqrt(loss)`` close to its
    value at an anchor loss. Concretely, the per-step multiplier applied on top
    of the base learning rate is::

        scale = clamp(sqrt(anchor_loss / loss), min_scale, max_scale)

    where ``anchor_loss`` is, by default, the first observed loss. Because high
    loss yields ``scale < 1`` and a converged (lower) loss yields ``scale > 1``,
    this both dampens forgetting on hard batches and restores learning speed as
    training stabilizes.

    The modifier is intentionally idempotent across calls: it tracks the factor
    it last applied to each param group and divides it back out before applying
    a new one, so repeatedly calling :meth:`step` does not compound, and it
    composes correctly with a base LR scheduler that rewrites ``lr`` each step.

    Args:
        min_scale (float): Lower bound on the LR multiplier. Defaults to 0.1.
        max_scale (float): Upper bound on the LR multiplier. Defaults to 1.0,
            which means FINCH only ever reduces the base LR (the conservative,
            forgetting-first setting from the paper). Set above 1.0 to also
            speed up learning once the loss falls below the anchor.
        anchor_loss (Optional[float]): Reference loss at which the multiplier is
            1.0. If ``None`` (default) the first loss passed to :meth:`step` is
            used as the anchor.
        smoothing (float): EMA coefficient in ``[0, 1)`` applied to the loss
            before computing the scale, to avoid jerking the LR around on noisy
            single-batch losses. ``0.0`` disables smoothing. Defaults to 0.0.
    """

    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 1.0,
        anchor_loss: Optional[float] = None,
        smoothing: float = 0.0,
    ) -> None:
        if not 0.0 < min_scale <= max_scale:
            raise ValueError(
                f"Require 0 < min_scale <= max_scale, got "
                f"min_scale={min_scale}, max_scale={max_scale}."
            )
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}.")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.anchor_loss = anchor_loss
        self.smoothing = smoothing

        self._ema_loss: Optional[float] = None
        self._last_scales: list[float] = []
        self._last_written: list[Optional[float]] = []
        self.last_scale: float = 1.0

    @classmethod
    def from_config(cls, cfg: Any) -> "FinchLRModifier":
        """Build a modifier from an (omegaconf) mapping, ignoring ``enabled``."""
        get = cfg.get
        return cls(
            min_scale=float(get("min_scale", 0.1)),
            max_scale=float(get("max_scale", 1.0)),
            anchor_loss=(
                float(get("anchor_loss")) if get("anchor_loss", None) is not None else None
            ),
            smoothing=float(get("smoothing", 0.0)),
        )

    def compute_scale(self, loss: float) -> float:
        """Return the LR multiplier for ``loss`` and update internal state.

        Raises:
            ValueError: If ``loss`` is not strictly positive.
        """
        if not math.isfinite(loss) or loss <= 0.0:
            raise ValueError(f"FINCH requires a finite positive loss, got {loss}.")

        if self.smoothing > 0.0:
            self._ema_loss = (
                loss
                if self._ema_loss is None
                else self.smoothing * self._ema_loss + (1.0 - self.smoothing) * loss
            )
            effective_loss = self._ema_loss
        else:
            effective_loss = loss

        if self.anchor_loss is None:
            self.anchor_loss = effective_loss

        scale = math.sqrt(self.anchor_loss / effective_loss)
        scale = max(self.min_scale, min(self.max_scale, scale))
        self.last_scale = scale
        return scale

    def step(self, loss: float, optimizer: Any) -> float:
        """Apply the loss-adaptive multiplier to ``optimizer``'s param groups.

        The multiplier is applied relative to the base LR for the step.
        ``optimizer`` only needs a ``param_groups`` attribute, so this works for
        both plain optimizers and any wrapper that exposes the same interface.

        FINCH composes with a base LR scheduler regardless of call order: if the
        ``lr`` was changed since FINCH last wrote it (i.e. a base scheduler
        rewrote it for this step), that value is taken as the base. Otherwise --
        the constant-LR / no-scheduler case -- the previously applied FINCH
        factor is divided back out so repeated calls never compound.

        Returns:
            float: The multiplier that was applied this step.
        """
        scale = self.compute_scale(loss)
        groups = optimizer.param_groups
        if len(self._last_scales) != len(groups):
            self._last_scales = [1.0] * len(groups)
            self._last_written = [None] * len(groups)
        for i, group in enumerate(groups):
            cur = group["lr"]
            if self._last_written[i] is not None and cur == self._last_written[i]:
                base_lr = cur / self._last_scales[i]
            else:
                base_lr = cur
            new_lr = base_lr * scale
            group["lr"] = new_lr
            self._last_written[i] = new_lr
            self._last_scales[i] = scale
        return scale
