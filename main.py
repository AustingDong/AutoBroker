from agents.falcon_7B import Falcon7BAgent

if __name__ == "__main__":
    # Initialize the Falcon-7B agent
    agent = Falcon7BAgent()

    # Example market input
    market_state = {
        "cash": 10.00,
        "holdings": [
            {"ticker": "AAPL", "quantity": 15, "avg_price": 172.50},
            {"ticker": "NVDA", "quantity": 50, "avg_price": 128.10},
            {"ticker": "GLD", "quantity": 50, "avg_price": 275.00}
        ],
        "market": [
            {"ticker": "AAPL", "price": 169.20, "change_pct": -1.92, "volume": 28000000},
            {"ticker": "NVDA", "price": 130.50, "change_pct": 1.85, "volume": 24000000},
            {"ticker": "GLD", "price": 305.50, "change_pct": 3.85, "volume": 24000000},
            {"ticker": "TSLA", "price": 315.50, "change_pct": 4.85, "volume": 29000000},
            {"ticker": "GOOG", "price": 215.50, "change_pct": -1.5, "volume": 21000000},
        ]
    }

    # Agent makes a step in the market
    actions = agent.step(market_state)

    # Print the actions taken by the agent
    print("Actions taken by the agent:")
    for action in actions:
        print(f"Ticker: {action['ticker']}, Activity: {action['activity']}, Quantity: {action['quantity']}")