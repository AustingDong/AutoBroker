import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def plot_pnl_comparison(baseline: np.ndarray, model: np.ndarray):
    if baseline.shape[0] != 252 or model.shape[0] != 252:
        raise ValueError("Each input array must have exactly 252 elements.")
    
    days = np.arange(1, 253)

    plt.figure(figsize=(8, 5))
    plt.plot(days, baseline, label="Baseline", linewidth=2)
    plt.plot(days, model, label="Model", linewidth=2)
    plt.xlabel("Trading Day")
    plt.ylabel("Cumulative PnL")
    plt.title("AI Trading vs Baseline")
    plt.grid(True)
    plt.legend()
    
    return plt

demo = gr.Interface(
    fn=plot_pnl_comparison,
    inputs=[
        gr.NumpyInput(label="Baseline (252 values)"),
        gr.NumpyInput(label="Model Output (252 values)")
    ],
    outputs=gr.Plot(label="PnL Plot"),
    title="Trading Strategy PnL Comparison",
    description="Input two NumPy arrays of length 252 to compare baseline vs. AI model."
)

demo.launch()
