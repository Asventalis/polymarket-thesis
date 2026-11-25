import sys
import json
import numpy as np

def print_metrics(filename, title, last_n=None):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    # Aggregate CM and Loss over ALL blocks (or last_n if specified)
    subset = data[-last_n:] if last_n else data
    cm = np.zeros((3, 3))
    avg_loss = 0.0
    for d in subset:
        if "cm" in d:
            cm += np.array(d["cm"])
        if "loss" in d:
            avg_loss += d["loss"]
    
    if len(subset) > 0:
        avg_loss /= len(subset)

    scope = f"Last {last_n} blocks" if last_n else f"ALL {len(subset)} blocks"
    print(f"\n============= {title} =============")
    print(f"--- 3x3 Confusion Matrix ({scope}) ---")
    print("          Pred DOWN(0) | Pred CONST(1) | Pred UP(2)")
    print(f"True DOWN : {int(cm[0][0]):12d} | {int(cm[0][1]):13d} | {int(cm[0][2]):10d}")
    print(f"True CONST: {int(cm[1][0]):12d} | {int(cm[1][1]):13d} | {int(cm[1][2]):10d}")
    print(f"True UP   : {int(cm[2][0]):12d} | {int(cm[2][1]):13d} | {int(cm[2][2]):10d}")

    print("\n--- Per-Class Metrics ---")
    classes = ["DOWN", "CONSTANT", "UP"]
    for i in range(3):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Class {classes[i]}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    if cm.sum() > 0:
        accuracy = np.trace(cm) / cm.sum()
        print(f"\nTotal Samples: {int(cm.sum())}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")

# Print metrics over ALL blocks for fair baseline comparison
print_metrics("val_history.json", "VALIDATION (Out-Of-Sample) - FULL DAY")
