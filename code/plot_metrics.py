import json
import matplotlib.pyplot as plt
import numpy as np

with open("loss_history.json", "r") as f:
    data = json.load(f)

blocks = [d["block"] for d in data if "accuracy" in d]
accuracies = [d["accuracy"] for d in data if "accuracy" in d]
f1_scores = [d["f1"] for d in data if "f1" in d]

plt.figure(figsize=(10, 6))

window = 50
if len(accuracies) >= window:
    smoothed_acc = np.convolve(accuracies, np.ones(window)/window, mode='valid')
    smoothed_f1 = np.convolve(f1_scores, np.ones(window)/window, mode='valid')
    
    plt.plot(blocks[window-1:], smoothed_acc, color='orange', linewidth=2, label=f"Accuracy (MA={window})")
    plt.plot(blocks[window-1:], smoothed_f1, color='green', linewidth=2, label=f"F1 Score (MA={window})")

plt.ylim(0.0, 1.0)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guessing (50%)')
plt.xlabel("Block Step")
plt.ylabel("Score")
plt.title("Model Metrics (Hour 13:00)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("metrics_plot.png", dpi=150)
print("Metrics plot saved.")

# Confusion Matrix of last 100 blocks
last_100 = data[-100:]
total_tp = sum(d["tp"] for d in last_100)
total_tn = sum(d["tn"] for d in last_100)
total_fp = sum(d["fp"] for d in last_100)
total_fn = sum(d["fn"] for d in last_100)

print("\n--- Confusion Matrix (Last 100 blocks) ---")
print(f"True Positives (Predicted UP, Actually UP): {total_tp}")
print(f"True Negatives (Predicted DOWN, Actually DOWN): {total_tn}")
print(f"False Positives (Predicted UP, Actually DOWN): {total_fp}")
print(f"False Negatives (Predicted DOWN, Actually UP): {total_fn}")

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

print("\n--- Aggregate Metrics (Last 100 blocks) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

