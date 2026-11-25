"""
ema.py

Exponential moving average of the model weights along the training pass.

The training loop makes one pass over the stream in time order, so the final weights reflect
the last blocks seen (and the optimizer noise) more than the pass as a whole; a learner on the
same inputs trained in random order does markedly better (learner audit of 17 October 2026).
A stateful model cannot shuffle its stream, but it can average its weights along the pass:
after every optimizer step the average is updated with decay ``d``; before the validation hours
the bias-corrected average (the EMA is accumulated from zero, so it is divided by 1 - d^n) is
copied into the model, which is then what the validation measures and what the final checkpoint
contains.  Horizon of the average in optimizer steps: about 1 / (1 - d).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class WeightEMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.n = 0
        self.applied = False
        self._avg = [torch.zeros_like(p, memory_format=torch.contiguous_format) for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call after every optimizer step."""
        self.n += 1
        for a, p in zip(self._avg, model.parameters()):
            a.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def correction(self) -> float:
        return 1.0 - self.decay ** self.n

    @torch.no_grad()
    def average(self):
        c = self.correction()
        return [a / c for a in self._avg]

    @torch.no_grad()
    def apply(self, model: nn.Module) -> Dict[str, float]:
        """Copy the bias-corrected average into the model (once; a no-op before the first update)."""
        if self.n == 0:
            return {"decay": self.decay, "steps": 0, "correction": 0.0}
        for a, p in zip(self.average(), model.parameters()):
            p.copy_(a)
        self.applied = True
        return {"decay": self.decay, "steps": self.n, "correction": self.correction()}
