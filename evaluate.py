from agents.falcon_7B import Falcon7BAgent
from agents.qwen_7B import Qwen7BAgent
from agents.llama_8B import Llama8BAgent
from environment.market import Market
from utils.schemas import State
from utils.parser import truncate_ids
from utils.plot import plot_pnl_comparison
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import wandb
import os
import numpy as np
from tqdm import trange
import gradio as gr



if __name__ == "__main__":
    DTYPE = torch.bfloat16
    MODEL_DIR = "ckpt/7d-train-obs1/epoch-020"

    num_years = 1
    episode_length = 7
    days_per_year = 21 * 12
    bs = num_years * days_per_year // episode_length
    
    agent = Falcon7BAgent(batch_size=bs, n_steps=episode_length, dtype=DTYPE)
    _, model = agent.load_for_inference(MODEL_DIR, dtype=DTYPE)
    agent.model = model

    market = Market(watch_list=["AAPL", "GOOG", "MSFT", "TSLA"], period="5y")
    
    start_d_idx = max(market.get_index("20220103"), market.min_start_index())
    # start_d_idx = max(market.get_index("20230103"), market.min_start_index())
    print("start_d_idx:", start_d_idx)


    baseline_sap = market.get_baseline(start_d_idx, window=252, baseline_ticker="^GSPC")
    baseline_sap_vals = np.array(
        [baseline_sap[dt] for dt in sorted(baseline_sap.keys())],
        dtype=np.float32
    )

    G = [[] for _ in range(agent.batch_size)]
    s_batch = [market.init_state(start_d_idx=start_d_idx + (i * episode_length), start_cash=50000) for i in range(agent.batch_size)]

    inputs_lst, outputs_lst, rewards = [], [], []
    for t in trange(agent.n_steps):
        torch.cuda.empty_cache()
        # Agent makes a step in the market
        a_batch, inputs, outputs = agent.step(s_batch)

        s_batch_ = []
        r_batch = []
        for i in range(agent.batch_size):
            s_next, r = market.step(s_batch[i], a_batch[i])
            s_batch_.append(s_next)
            r_batch.append(r)

        
        # print(f"reward: ", r_batch)
        s_batch = s_batch_
        for g, r in zip(G, r_batch):
            g.append(r)
        
        input_ids = inputs["input_ids"].detach().cpu()
        attention_masks = inputs["attention_mask"].detach().cpu()
        response_ids = outputs[:, input_ids.shape[1]:].detach().cpu()
        reward_tensor = torch.tensor(r_batch, dtype=DTYPE).detach().cpu()


        for i in range(agent.batch_size):
            inputs_lst.append(input_ids[i][attention_masks[i].bool()])
            outputs_lst.append(truncate_ids(response_ids[i]))
            rewards.append(reward_tensor[i])

    
    
    torch.cuda.empty_cache()
    
    gain = np.array(G, dtype=np.float32)
    daily_gain = gain.flatten()[:252]
    accumulated_gain = np.cumsum(daily_gain)

    print("gain:", gain.shape)
    print("daily_gain:", daily_gain.shape)
    print("accumulated_gain:", accumulated_gain.shape)
    
    plot_pnl_comparison(baseline_sap_vals, accumulated_gain)
    # demo = gr.Interface(
    #     fn=plot_pnl_comparison,
    #     inputs=[
    #         # gr.NumpyInput(label="Baseline (252 values)"),
    #         # gr.NumpyInput(label="Model Output (252 values)")
    #         baseline_sap,
    #         daily_gain
    #     ],
    #     outputs=gr.Plot(label="PnL Plot"),
    #     title="Trading Strategy PnL Comparison",
    #     description="Input two NumPy arrays of length 252 to compare baseline vs. AI model."
    # )

    # demo.launch()
