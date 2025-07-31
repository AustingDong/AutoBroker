from utils.schemas import State, Action, HoldingData, MarketData
from copy import deepcopy
import yfinance as yf
from datetime import date

class Market:
    def __init__(self, watch_list=["AAPL", "NVDA", "GLD", "TSLA", "GOOG"]):
        self.period = "1y"
        self.watch_list = watch_list
        self.prices = yf.download(self.watch_list, period=self.period)

    def init_state(self, start_d_idx, start_cash):

        start_d = self.prices.index[start_d_idx].strftime("%Y%m%d")
        last_d = self.prices.index[start_d_idx - 1]
        
        market_info = self.get_market_info(start_d, last_d)

        s = {
            "date": start_d,
            "cash": start_cash,
            "holdings": [],
            "market": market_info
        }

        return State(**s)

    def get_market_info(self, start_d, last_d):
        market_info = []
        for ticker in self.watch_list:
            start_price = self.prices.Open[ticker][start_d].round(2)
            last_price = self.prices.Open[ticker][last_d].round(2)
            change_pct = (100 * (start_price - last_price) / last_price).round(2)
            volume = self.prices.Volume[ticker][start_d]
            info = {
                "ticker": ticker,
                "price": start_price,
                "change_pct": change_pct,
                "volume": volume
            }
            market_info.append(info)
        return market_info

    def _total_asset(self, s: State, ticker_to_price: dict[str, float]) -> float:
        total = s.cash
        for h in s.holdings:
            price = ticker_to_price.get(h.ticker, 0)
            total += h.quantity * price
        return total

    def step_old(self, s: State, a: list[Action]):
        s_ = deepcopy(s)
        penalty = 0

        penalty_strength = 10

        last_d = s.date
        last_d_idx = self.prices.index.get_loc(last_d)
        start_d = self.prices.index[last_d_idx + 1]

        s_.date = start_d
        new_market = self.get_market_info(start_d, last_d)
        s_.market = [MarketData(**new_m) for new_m in new_market]

        ticker_to_price = {m.ticker: m.price for m in s.market}

        # Sell
        for act in a:
            ticker = act.ticker
            price = ticker_to_price.get(ticker)
            if price is None or act.score <= 0:
                continue

            current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)

            if act.activity == "Sell":
                if current_holding and current_holding.quantity >= act.score:
                    s_.cash += price * act.score
                    current_holding.quantity -= act.score
                    if current_holding.quantity == 0:
                        s_.holdings.remove(current_holding)
                else:
                    penalty += penalty_strength * act.score

        # Buy & Hold
        for act in a:
            ticker = act.ticker
            price = ticker_to_price.get(ticker)
            if price is None or act.score <= 0:
                continue

            current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)

            if act.activity == "Buy":
                total_cost = round(price * act.score, 2)
                if s_.cash >= total_cost:
                    s_.cash -= total_cost
                    if current_holding:
                        total_qty = current_holding.quantity + act.score
                        new_avg_purchase_price = (
                            current_holding.avg_purchase_price * current_holding.quantity + price * act.score
                        ) / total_qty
                        current_holding.quantity = total_qty
                        current_holding.avg_purchase_price = round(new_avg_purchase_price, 2)
                    else:
                        s_.holdings.append(HoldingData(ticker=ticker, quantity=act.score, avg_purchase_price=price))
                else:
                    penalty += penalty_strength * act.score

        # Calculate reward
        ticker_to_price_next = {m.ticker: m.price for m in s_.market}
        total_prev = self._total_asset(s, ticker_to_price)
        total_next = self._total_asset(s_, ticker_to_price_next)
        reward = total_next - total_prev - penalty

        return s_, reward
    
    def step(self, s: State, a: list[Action], k: float = 0.5):
        s_ = deepcopy(s)
        penalty = 0

        penalty_strength = 10

        last_d = s.date
        last_d_idx = self.prices.index.get_loc(last_d)
        start_d = self.prices.index[last_d_idx + 1]

        s_.date = start_d
        new_market = self.get_market_info(start_d, last_d)
        s_.market = [MarketData(**new_m) for new_m in new_market]

        ticker_to_price = {m.ticker: m.price for m in s.market}

        # Extract Actions
        buy_actions = [act for act in a if act.activity == "Buy" and act.score > 0 and act.ticker in ticker_to_price]
        sell_actions = [act for act in a if act.activity == "Sell" and act.score > 0 and act.ticker in ticker_to_price]

        # Sell
        total_sell_score = sum(act.score for act in sell_actions)
        if total_sell_score > 0:
            for act in sell_actions:
                ticker = act.ticker
                price = ticker_to_price[ticker]
                current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)
                if current_holding:
                    proportion = act.score / total_sell_score
                    sell_qty = int(proportion * current_holding.quantity)
                    if sell_qty > 0:
                        s_.cash += price * sell_qty
                        current_holding.quantity -= sell_qty
                        if current_holding.quantity == 0:
                            s_.holdings.remove(current_holding)
                    else:
                        # nothing to sell
                        pass
                # else:
                #     penalty += penalty_strength * act.score

        # Buy & Hold
        total_buy_score = sum(act.score for act in buy_actions)
        available_cash = s_.cash
        buy_budget = k * available_cash

        if total_buy_score > 0:
            for act in buy_actions:
                ticker = act.ticker
                price = ticker_to_price[ticker]
                proportion = act.score / total_buy_score
                allocated_cash = proportion * buy_budget
                buy_qty = int(allocated_cash // price)

                if buy_qty <= 0:
                    continue  # not enough to buy even 1 unit

                total_cost = price * buy_qty
                if total_cost <= s_.cash:
                    s_.cash -= total_cost
                    current_holding = next((h for h in s_.holdings if h.ticker == ticker), None)
                    if current_holding:
                        total_qty = current_holding.quantity + buy_qty
                        new_avg_price = (
                            current_holding.avg_purchase_price * current_holding.quantity + price * buy_qty
                        ) / total_qty
                        current_holding.quantity = total_qty
                        current_holding.avg_purchase_price = round(new_avg_price, 2)
                    else:
                        s_.holdings.append(HoldingData(ticker=ticker, quantity=buy_qty, avg_purchase_price=price))
                # else:
                #     penalty += penalty_strength * act.score

        # Calculate reward
        ticker_to_price_next = {m.ticker: m.price for m in s_.market}
        total_prev = self._total_asset(s, ticker_to_price)
        total_next = self._total_asset(s_, ticker_to_price_next)
        # reward = total_next - total_prev - penalty
        reward = total_next - total_prev

        return s_, reward
