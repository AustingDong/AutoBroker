# market.py
from utils.schemas import State, Action, HoldingData, MarketData
from copy import deepcopy
import yfinance as yf
import pandas as pd

class Market:
    def __init__(self, watch_list=["AAPL", "NVDA", "GLD", "TSLA", "GOOG"],
                 period="5y",
                 ma_windows=(7, 30),
                 mom_windows=(7, 14)):
        self.period = period
        self.watch_list = watch_list
        self.ma_windows = tuple(sorted(ma_windows))
        self.mom_windows = tuple(sorted(mom_windows))
        self.max_lookback = max((self.ma_windows + self.mom_windows) or (0,))  # e.g., 30

        # MultiIndex columns: ('Open'|'Volume'|...), ticker
        self.prices = yf.download(self.watch_list, period=self.period)

        # Build a tidy feature frame per ticker to speed up lookups
        # We'll compute on Open to stay consistent with your current logic.
        self.features = {}
        for ticker in self.watch_list:
            df = pd.DataFrame({
                "open": self.prices["Open"][ticker],
                "volume": self.prices["Volume"][ticker]
            }).copy()

            # Moving averages
            for w in self.ma_windows:
                df[f"ma{w}"] = df["open"].rolling(w).mean()

            # Momentum as percent change over n days
            for w in self.mom_windows:
                df[f"mom{w}"] = df["open"].pct_change(w)

            self.features[ticker] = df

    def min_start_index(self):
        """Minimum index you should start at to have all features non-NaN."""
        return self.max_lookback
    
    def get_index(self, date_str):
        """Get the index of the given date string in the prices DataFrame."""
        return self.prices.index.get_loc(pd.Timestamp(date_str))

    def init_state(self, start_d_idx, start_cash):
        # Ensure we don’t start before features exist
        start_d_idx = max(start_d_idx, self.min_start_index())

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
            # current & last
            start_price = float(self.prices["Open"][ticker][start_d].round(2))
            last_price = float(self.prices["Open"][ticker][last_d].round(2))
            change_pct = round(100 * (start_price - last_price) / last_price, 2)
            volume = int(self.prices["Volume"][ticker][start_d])

            # Features (pull by timestamp)
            fdf = self.features[ticker]
            # start_d is string; align to Timestamp index
            ts = fdf.index.get_loc(pd.Timestamp(start_d))
            row = fdf.iloc[ts]

            info = {
                "ticker": ticker,
                "price": start_price,
                "change_pct": change_pct,
                "volume": volume,
                # new fields (may be NaN at early indices; cast to float or None)
                "ma7":  float(row.get("ma7"))  if "ma7"  in row else None,
                "ma30": float(row.get("ma30")) if "ma30" in row else None,
                "mom7": float(row.get("mom7")) if "mom7" in row else None,
                "mom14": float(row.get("mom14")) if "mom14" in row else None,
            }
            market_info.append(info)
        return market_info

    def _total_asset(self, s: State, ticker_to_price: dict[str, float]) -> float:
        total = s.cash
        for h in s.holdings:
            price = ticker_to_price.get(h.ticker, 0.0)
            total += h.quantity * price
        return total

    def step(self, s: State, a: list[Action], k: float = 0.5):
        s_ = deepcopy(s)
        last_d = s.date
        last_d_idx = self.prices.index.get_loc(last_d)
        next_idx = last_d_idx + 1
        if next_idx >= len(self.prices.index):
            # Terminal: no more days
            return s, 0.0

        start_d = self.prices.index[next_idx]
        s_.date = start_d

        new_market = self.get_market_info(start_d, last_d)
        s_.market = [MarketData(**new_m) for new_m in new_market]

        ticker_to_price = {m.ticker: m.price for m in s.market}

        # Weighted sells
        sell_actions = [act for act in a if act.activity == "Sell" and act.score > 0 and act.ticker in ticker_to_price]
        total_sell_score = sum(act.score for act in sell_actions)
        if total_sell_score > 0:
            for act in sell_actions:
                price = ticker_to_price[act.ticker]
                current = next((h for h in s_.holdings if h.ticker == act.ticker), None)
                if not current or current.quantity <= 0:
                    continue
                proportion = act.score / total_sell_score
                sell_qty = int(proportion * current.quantity)
                if sell_qty > 0:
                    s_.cash += price * sell_qty
                    current.quantity -= sell_qty
                    if current.quantity == 0:
                        s_.holdings.remove(current)

        # Weighted buys (budget = k * cash)
        buy_actions = [act for act in a if act.activity == "Buy" and act.score > 0 and act.ticker in ticker_to_price]
        total_buy_score = sum(act.score for act in buy_actions)
        buy_budget = k * s_.cash
        if total_buy_score > 0 and buy_budget > 0:
            for act in buy_actions:
                price = ticker_to_price[act.ticker]
                proportion = act.score / total_buy_score
                alloc = proportion * buy_budget
                buy_qty = int(alloc // price)
                if buy_qty <= 0:
                    continue
                cost = price * buy_qty
                if cost <= s_.cash:
                    s_.cash -= cost
                    current = next((h for h in s_.holdings if h.ticker == act.ticker), None)
                    if current:
                        total_qty = current.quantity + buy_qty
                        new_avg = (current.avg_purchase_price * current.quantity + price * buy_qty) / total_qty
                        current.quantity = total_qty
                        current.avg_purchase_price = round(new_avg, 2)
                    else:
                        s_.holdings.append(HoldingData(ticker=act.ticker, quantity=buy_qty, avg_purchase_price=price))

        # Reward = proportional change in total asset
        ticker_to_price_next = {m.ticker: m.price for m in s_.market}
        total_prev = self._total_asset(s, ticker_to_price)
        total_next = self._total_asset(s_, ticker_to_price_next)
        reward = (total_next - total_prev) / total_prev if total_prev != 0 else 0.0

        return s_, reward
