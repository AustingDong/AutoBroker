from agents.falcon_7B import Falcon7BAgent
from agents.qwen_7B import Qwen7BAgent
from environment.market import Market
from utils.schemas import State

if __name__ == "__main__":
    # Initialize
    agent = Falcon7BAgent()
    market = Market(watch_list=["NVDA", "GLD"])
    # agent = Qwen7BAgent()

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
    G = 0
    while t <= 10:
        # Agent makes a step in the market
        a = agent.step(s)
        s_, r = market.step(s, a)

        # Print the actions taken by the agent
        print("time: ", t)
        print("Actions taken by the agent:")
        for action in a:
            print(f"Ticker: {action.ticker}, Activity: {action.activity}, Quantity: {action.quantity}")
        print("new state: ", s_)
        print(f"reward: ", r)
        t += 1
        s = s_
        G += r
    print("gain", G)

