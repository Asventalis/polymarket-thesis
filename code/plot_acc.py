import json
import matplotlib.pyplot as plt
import numpy as np

with open("loss_history.json", "r") as f:
    data = json.load(f)

blocks = [d["block"] for d in data if "accuracy" in d]
accuracies = [d["accuracy"] for d in data if "accuracy" in d]

plt.figure(figsize=(10, 6))
plt.plot(blocks, accuracies, alpha=0.3, label="Raw Batch Accuracy", color='blue')

# Add a rolling average (e.g. 200 blocks) to show the trend clearly
window = 200
if len(accuracies) >= window:
    smoothed = np.convolve(accuracies, np.ones(window)/window, mode='valid')
    plt.plot(blocks[window-1:], smoothed, color='orange', linewidth=2, label=f"Rolling Mean (w={window})")

plt.ylim(0.0, 1.0)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guessing (50%)')
plt.xlabel("Block Step")
plt.ylabel("Prediction Accuracy")
plt.title("End-to-End Prediction Accuracy (5 Continuous Hours)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("acc_plot_5h.png", dpi=150)
print("Accuracy plot saved.")
