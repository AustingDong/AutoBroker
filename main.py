from agents.falcon_7B import Falcon7BAgent
from agents.qwen_7B import Qwen7BAgent
from agents.llama_8B import Llama8BAgent
from environment.market import Market
from utils.schemas import State
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import wandb


if __name__ == "__main__":
    # Initialize
    wandb.login()
    # agent = Falcon7BAgent()
    agent = Llama8BAgent()
    

    market = Market(watch_list=["NVDA", "GLD", "TSLA"])
    
    t = 0
    # Example market input
    # s = {
    #     "cash": 10000.00,
    #     "holdings": [
    #         {"ticker": "AAPL", "quantity": 2, "avg_price": 172.50},
    #         {"ticker": "NVDA", "quantity": 1, "avg_price": 128.10},
    #         {"ticker": "GLD", "quantity": 10, "avg_price": 275.00}
    #     ],
    #     "market": [
    #         {"ticker": "AAPL", "price": 169.20, "change_pct": -1.92, "volume": 28000000},
    #         {"ticker": "NVDA", "price": 130.50, "change_pct": 1.85, "volume": 24000000},
    #         {"ticker": "GLD", "price": 305.50, "change_pct": 3.85, "volume": 24000000},
    #         {"ticker": "TSLA", "price": 315.50, "change_pct": 4.85, "volume": 29000000},
    #         {"ticker": "GOOG", "price": 215.50, "change_pct": -1.5, "volume": 21000000},
    #     ]
    # }
    start_ymd = {
        "year": 2025,
        "month": 5,
        "day": 1
    }
    s = market.init_state(start_ymd=start_ymd, start_cash=15000)
    s = State(**s)

    trainer = PPOTrainer(config=agent.ppo_config, model=agent.model, tokenizer=agent.tokenizer, ref_model=agent.ref_model)

    epochs = 50

    run = wandb.init(project="Stock", name="tr")
    for epoch in range(epochs):
        
        G = 0
        s = market.init_state(start_ymd=start_ymd, start_cash=15000)
        s = State(**s)
        inputs_lst, outputs_lst, rewards = [], [], []
        for t in range(agent.batch_size):
            # Agent makes a step in the market
            a, inputs, outputs = agent.step(s)
            s_, r = market.step(s, a)

            # Print the actions taken by the agent
            print("time: ", t)
            print("Actions taken by the agent:")
            for action in a:
                print(f"Ticker: {action.ticker}, Activity: {action.activity}, Quantity: {action.quantity}")
            # print("new state: ", s_)
            print(f"reward: ", r)
            s = s_
            G += r
            
            reward = torch.tensor(r, dtype=torch.float16).to("cuda")
            input_ids = inputs["input_ids"]
            response_ids = outputs[:, input_ids.shape[1]:]

            # print(response_ids.shape)
            print("decoded:", agent.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0])
            inputs_lst.append(input_ids[0])
            outputs_lst.append(response_ids[0])
            rewards.append(reward)

        print("gain", G)
        mean_r = torch.mean(torch.tensor(rewards))
        std_r = torch.std(torch.tensor(rewards))
        normalized_rewards = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        summary = trainer.step(inputs_lst, outputs_lst, normalized_rewards)
        print(summary)
        wandb.log({
            "gain": G
        })
    run.finish()

