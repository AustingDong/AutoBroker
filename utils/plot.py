import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def plot_pnl_comparison(baseline: np.ndarray, model: np.ndarray):
    if baseline.shape[0] != model.shape[0]:
        raise ValueError("Baseline result and Model predictions have different shape.")
    
    days = np.arange(1, 253)

    plt.figure(figsize=(8, 5))
    plt.plot(days, baseline, label="Baseline", linewidth=2)
    plt.plot(days, model, label="Model", linewidth=2)
    plt.xlabel("Trading Day")
    plt.ylabel("Cumulative PnL")
    plt.title("Auto Stock Trading vs Baseline")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

