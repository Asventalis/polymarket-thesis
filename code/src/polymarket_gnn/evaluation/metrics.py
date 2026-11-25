"""
metrics.py

Shared 3-class classification metrics computed from a confusion matrix.
Class order everywhere in the project: 0 = DOWN, 1 = CONSTANT, 2 = UP.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

CLASS_NAMES = ["DOWN", "CONSTANT", "UP"]
N_CLASSES = len(CLASS_NAMES)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """cm[true][pred] for 3 classes, vectorised."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES)
    return cm.reshape(N_CLASSES, N_CLASSES)


def cm_to_metrics(cm) -> Dict:
    """
    Accuracy, balanced accuracy, macro-F1 and per-class precision/recall/F1
    from a 3x3 confusion matrix (rows = true class, cols = predicted class).
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    per_class = {}
    recalls = []
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
        recalls.append(recall)
        f1s.append(f1)
    return {
        "n": int(total),
        "accuracy": float(np.trace(cm) / total) if total > 0 else 0.0,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "class_distribution": {
            name: float(cm[i, :].sum() / total) if total > 0 else 0.0
            for i, name in enumerate(CLASS_NAMES)
        },
        "per_class": per_class,
        "cm": cm.astype(np.int64).tolist(),
    }


def format_metrics_row(label: str, m: Dict) -> str:
    pc = m["per_class"]
    return (
        f"  {label:<34} acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
        f"macroF1={m['macro_f1']:.4f}  DOWN_F1={pc['DOWN']['f1']:.4f}  "
        f"CONST_F1={pc['CONSTANT']['f1']:.4f}  UP_F1={pc['UP']['f1']:.4f}  n={m['n']}"
    )
