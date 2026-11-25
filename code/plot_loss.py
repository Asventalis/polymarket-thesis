import json
import matplotlib.pyplot as plt
import numpy as np

with open("loss_history.json", "r") as f:
    data = json.load(f)

blocks = [d["block"] for d in data]
losses = [d["loss"] for d in data]

# Apply log scale for the loss
plt.figure(figsize=(10, 6))
plt.plot(blocks, losses, alpha=0.6, label="Raw Loss (BCE)")

# Add a rolling average (e.g. 50 blocks) to show the trend clearly
window = 50
if len(losses) >= window:
    smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(blocks[window-1:], smoothed_losses, color='red', linewidth=2, label=f"Rolling Mean (w={window})")

plt.yscale('log')
plt.xlabel("Block Step")
plt.ylabel("Log Binary Cross-Entropy Loss")
plt.title("End-to-End Training Loss (Hour 13:00)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
print("Plot saved.")
