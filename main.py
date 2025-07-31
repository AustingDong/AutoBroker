from agents.falcon_7B import Falcon7BAgent
from agents.qwen_7B import Qwen7BAgent
from agents.llama_8B import Llama8BAgent
from environment.market import Market
from utils.schemas import State
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import wandb
import gc



if __name__ == "__main__":
    # Initialize
    wandb.login()
    DTYPE = torch.bfloat16
    agent = Falcon7BAgent(batch_size=1, n_steps=30, dtype=DTYPE)
    

    market = Market(watch_list=["NVDA", "GLD", "TSLA", "AAPL", "GOOG", "MSFT"])
    
    t = 0
    
    start_d_idx = 10

    trainer = PPOTrainer(config=agent.ppo_config, 
                         model=agent.model, 
                         tokenizer=agent.tokenizer, 
                         ref_model=agent.ref_model
                         )
    # trainer.accelerator_config = {"mixed_precision": "bf16"}
    epochs = 30

    run = wandb.init(project="StockTrader", name="30d-obs1")
    for epoch in range(epochs):
        
        G = 0
        s_batch = [market.init_state(start_d_idx=start_d_idx+i, start_cash=25000) for i in range(agent.batch_size)]

        inputs_lst, outputs_lst, rewards = [], [], []
        for t in range(agent.n_steps):
            # Agent makes a step in the market
            a_batch, inputs, outputs = agent.step(s_batch)

            s_batch_ = []
            r_batch = []
            for i in range(agent.batch_size):
                s_next, r = market.step(s_batch[i], a_batch[i])
                s_batch_.append(s_next)
                r_batch.append(r)

            # Print the actions taken by the agent
            # print("time: ", t)
            # print("Actions taken by the agent:")
            # for action in a:
            #     print(f"Ticker: {action.ticker}, Activity: {action.activity}, Quantity: {action.quantity}")
            # print("new state: ", s_)
            print(f"reward: ", r_batch)
            s_batch = s_batch_
            G += sum(r_batch) / agent.batch_size
            
            input_ids = inputs["input_ids"]
            response_ids = outputs[:, input_ids.shape[1]:]
            reward_tensor = torch.tensor(r_batch, dtype=DTYPE)

            # print(response_ids.shape)
            # print("decoded:", agent.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0])
            for i in range(agent.batch_size):
                inputs_lst.append(input_ids[i].detach().cpu())
                outputs_lst.append(response_ids[i].detach().cpu())
                rewards.append(reward_tensor[i].detach().cpu())

        print("gain", G)
        # rewards_tensor = torch.stack(rewards)
        # mean_r = rewards_tensor.mean()
        # std_r = rewards_tensor.std()
        # normalized_rewards = torch.stack([(r - mean_r) / (std_r + 1e-8) for r in rewards_tensor])
        # clipped_rewards = torch.clamp(normalized_rewards, -10, 10)
        # print("clipped_rewards:", clipped_rewards)
        # normalized_rewards = [r.detach().clone() for r in clipped_rewards]
        
        summary = trainer.step(inputs_lst, outputs_lst, rewards)
        # print(summary)
        with open("summary.txt", "w") as f:
            f.write(str(summary))
        trainer.optimizer.state.clear()
        torch.cuda.empty_cache()

        # print(summary["ppo/policy/ratio"].shape)

        # for ratio in summary["ppo/policy/ratio"]:
        #     print(ratio)
        # break
        # del inputs_lst, outputs_lst, rewards, rewards_tensor, normalized_rewards
        # del summary
        # gc.collect()
        
        wandb.log({
            "gain": G
        })
    run.finish()

