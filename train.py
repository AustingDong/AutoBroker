from agents.falcon_7B import Falcon7BAgent
from agents.qwen_7B import Qwen7BAgent
from agents.llama_8B import Llama8BAgent
from environment.market import Market
from utils.schemas import State
from utils.parser import truncate_ids
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import wandb
import os
import numpy as np
from tqdm import trange



if __name__ == "__main__":
    # Initialize
    wandb.login()
    DTYPE = torch.bfloat16

    num_years = 1
    episode_length = 7
    days_per_year = 21 * 12
    bs = num_years * days_per_year // episode_length
    
    agent = Falcon7BAgent(batch_size=bs, n_steps=episode_length, dtype=DTYPE)
    

    market = Market(watch_list=["AAPL", "GOOG", "MSFT", "TSLA"], period="5y")
    
    start_d_idx = max(market.get_index("20220103"), market.min_start_index())
    # start_d_idx = max(market.get_index("20230103"), market.min_start_index())
    print("start_d_idx:", start_d_idx)

    trainer = PPOTrainer(config=agent.ppo_config, 
                         model=agent.model, 
                         tokenizer=agent.tokenizer, 
                         ref_model=agent.ref_model
                         )

    epochs = 20

    run = wandb.init(project="StockTrader", name="7d-train-obs1")
    ckpt_root = os.path.join("ckpt", run.name)
    os.makedirs(ckpt_root, exist_ok=True)
    for epoch in trange(epochs):
        
        G = [0 for _ in range(agent.batch_size)]
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
            G = [g + r for g, r in zip(G, r_batch)]
            
            input_ids = inputs["input_ids"].detach().cpu()
            attention_masks = inputs["attention_mask"].detach().cpu()
            response_ids = outputs[:, input_ids.shape[1]:].detach().cpu()
            reward_tensor = torch.tensor(r_batch, dtype=DTYPE).detach().cpu()

            # print(response_ids.shape)
            # print("decoded:", agent.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0])
            for i in range(agent.batch_size):
                inputs_lst.append(input_ids[i][attention_masks[i].bool()])
                outputs_lst.append(truncate_ids(response_ids[i]))
                rewards.append(reward_tensor[i])

        # print("gain", G)
        # print("inputs_lst", inputs_lst, "outputs_lst", outputs_lst, "rewards", rewards)
        # for i in range(len(inputs_lst)):
        #     print(inputs_lst[i].shape, outputs_lst[i].shape, rewards[i].shape)
        # break
        summary = trainer.step(inputs_lst, outputs_lst, rewards)
        # print(summary)
        with open("summary.txt", "w") as f:
            f.write(str(summary))
        trainer.optimizer.state.clear()
        torch.cuda.empty_cache()
        
        gain = np.array(G, dtype=np.float32)

        wandb.log({
            "gain/mean": float(gain.mean()),
            "gain/std":  float(gain.std()),
            "gain/min":  float(gain.min()),
            "gain/max":  float(gain.max()),
            "gain/hist": wandb.Histogram(gain),   # interactive histogram
        }, step=epoch)
    run.finish()
    epoch_ckpt = os.path.join(ckpt_root, f"epoch-{epoch+1:03d}")
    agent.save(epoch_ckpt, save_ref_value_head=False)

