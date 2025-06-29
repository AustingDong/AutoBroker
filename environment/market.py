from utils.schemas import State, Action, HoldingData, MarketData
from copy import deepcopy


class Market:
    def __init__(self):
        pass

    def _total_asset(self, s: State, ticker_to_price: dict[str, float]) -> float:
        total = s.cash
        for h in s.holdings:
            price = ticker_to_price.get(h.ticker, 0)
            total += h.quantity * price
        return total

    def step(self, s: State, a: list[Action]):
        s_ = deepcopy(s)
        penalty = 0

        ticker_to_price = {m.ticker: m.price for m in s.market}

        # Sell
        for act in a:
            ticker = act.ticker
            price = ticker_to_price.get(ticker)
            if price is None or act.quantity <= 0:
                continue

            current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)

            if act.activity == "Sell":
                if current_holding and current_holding.quantity >= act.quantity:
                    s_.cash += price * act.quantity
                    current_holding.quantity -= act.quantity
                    if current_holding.quantity == 0:
                        s_.holdings.remove(current_holding)
                else:
                    penalty += 100 * act.quantity

        # Buy & Hold
        for act in a:
            ticker = act.ticker
            price = ticker_to_price.get(ticker)
            if price is None or act.quantity <= 0:
                continue

            current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)

            if act.activity == "Buy":
                total_cost = round(price * act.quantity, 2)
                if s_.cash >= total_cost:
                    s_.cash -= total_cost
                    if current_holding:
                        total_qty = current_holding.quantity + act.quantity
                        new_avg_price = (
                            current_holding.avg_price * current_holding.quantity + price * act.quantity
                        ) / total_qty
                        current_holding.quantity = total_qty
                        current_holding.avg_price = round(new_avg_price, 2)
                    else:
                        s_.holdings.append(HoldingData(ticker=ticker, quantity=act.quantity, avg_price=price))
                else:
                    penalty += 100 * act.quantity

        # Calculate reward
        total_prev = self._total_asset(s, ticker_to_price)
        total_next = self._total_asset(s_, ticker_to_price)
        reward = total_next - total_prev - penalty

        return s_, reward
