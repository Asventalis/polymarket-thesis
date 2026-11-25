"""
objectives.py

Training objectives and imbalance handling for the 3-class direction task
(0 = DOWN, 1 = CONSTANT, 2 = UP), plus the regression formulation suggested by
the supervisor: predict the signed mid-price change and turn it into a class
with a threshold calibrated on TRAINING data only.

Everything here is a pure function / small stateful helper so it can be unit
tested without the data pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polymarket_gnn.evaluation.metrics import N_CLASSES, cm_to_metrics

DOWN, CONST, UP = 0, 1, 2


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights.  ``reduction='none'`` returns per-sample losses."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss


def parse_alpha(spec: str, device) -> Optional[torch.Tensor]:
    """'2,1,2' → tensor; 'none' → None (unweighted); 'auto' → None here (weights are set online)."""
    spec = spec.strip().lower()
    if spec in ("none", "auto"):
        return None
    vals = [float(x) for x in spec.split(",")]
    if len(vals) != N_CLASSES:
        raise ValueError(f"--focal-alpha needs {N_CLASSES} comma-separated values, 'auto' or 'none'; got {spec!r}")
    return torch.tensor(vals, dtype=torch.float32, device=device)


def inverse_frequency_weights(counts: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    """
    Class weights ∝ 1 / frequency, normalised to mean 1.  ``counts`` are the
    class counts seen so far in training (causal: computed before the current
    block is added).  With no data yet all weights are 1.
    """
    c = counts.to(torch.float32) + smoothing
    w = c.sum() / c
    return w / w.mean()


def undersample_mask(
    labels: torch.Tensor,
    counts: torch.Tensor,
    ratio: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Random undersampling of the frequent classes for the LOSS only.

    Each sample of class c is kept with probability min(1, ratio · f_min / f_c),
    where f are the running training class frequencies and f_min the smallest
    one.  ratio = 1 gives (in expectation) a balanced loss; ratio = 2 lets the
    most frequent class appear at most twice as often as the rarest; a class
    never seen so far is always kept.  Node-state updates and metrics still use
    every sample.
    """
    if ratio <= 0 or labels.numel() == 0:
        return torch.ones_like(labels, dtype=torch.bool)
    c = counts.to(torch.float32)
    if (c > 0).sum() < 2:
        return torch.ones_like(labels, dtype=torch.bool)
    f_min = c[c > 0].min()
    keep_prob = torch.where(c > 0, torch.clamp(ratio * f_min / c.clamp(min=1.0), max=1.0), torch.ones_like(c))
    u = torch.rand(labels.shape, generator=generator, device=labels.device)
    return u < keep_prob[labels]


def classes_from_delta(delta: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """True classes from the realised signed change (same rule as MidPriceIndex.labels)."""
    delta = np.asarray(delta, dtype=np.float64)
    return np.where(delta > eps, UP, np.where(delta < -eps, DOWN, CONST)).astype(np.int64)


def classify_delta(pred: torch.Tensor, tau: float) -> torch.Tensor:
    """Predicted class from a predicted signed change: UP if > tau, DOWN if < -tau, else CONSTANT."""
    return (1 + (pred > tau).long() - (pred < -tau).long())


def default_thresholds() -> List[float]:
    """Candidate thresholds in raw mid-price units (0 … 0.2).  The mid moves in half-ticks of 0.0005."""
    return [0.0] + np.logspace(-4, np.log10(0.2), 30).round(6).tolist()


class ThresholdCalibrator:
    """
    Accumulates, for every candidate threshold, the 3x3 confusion matrix of
    ``classify_delta(pred, tau)`` against the true classes.

    Used twice: on the last training day(s) to choose tau (``best``), and on
    validation as a diagnostic sweep.  The validation sweep must NOT be used
    to pick tau.
    """

    def __init__(self, thresholds: Sequence[float], device="cpu"):
        self.thresholds = torch.tensor(list(thresholds), dtype=torch.float32, device=device)
        self.cms = torch.zeros(len(thresholds) * N_CLASSES * N_CLASSES, dtype=torch.long, device=device)
        self.n = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, y_true: torch.Tensor) -> None:
        if pred.numel() == 0:
            return
        tau = self.thresholds[:, None]                                  # (T, 1)
        cls = 1 + (pred[None, :] > tau).long() - (pred[None, :] < -tau).long()   # (T, N)
        offs = torch.arange(len(self.thresholds), device=pred.device)[:, None] * (N_CLASSES * N_CLASSES)
        flat = (offs + y_true[None, :] * N_CLASSES + cls).reshape(-1)
        self.cms += torch.bincount(flat, minlength=self.cms.numel())
        self.n += int(pred.numel())

    def table(self) -> List[Dict]:
        cms = self.cms.view(len(self.thresholds), N_CLASSES, N_CLASSES).cpu().numpy()
        rows = []
        for tau, cm in zip(self.thresholds.cpu().tolist(), cms):
            m = cm_to_metrics(cm)
            rows.append({"tau": float(tau), "balanced_accuracy": m["balanced_accuracy"],
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "cm": m["cm"]})
        return rows

    def best(self, metric: str = "balanced_accuracy") -> Optional[Dict]:
        if self.n == 0:
            return None
        return max(self.table(), key=lambda r: r[metric])
