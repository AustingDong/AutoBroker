from utils.schemas import State, Action, HoldingData, MarketData
from copy import deepcopy
import yfinance as yf
from datetime import date

class Market:
    def __init__(self, watch_list=["AAPL", "NVDA", "GLD", "TSLA", "GOOG"]):
        self.period = "1y"
        self.watch_list = watch_list
        self.prices = yf.download(self.watch_list, period=self.period)

    def init_state(self, start_ymd, start_cash):
        start_date = date(**start_ymd)
        start_d = start_date.strftime("%Y%m%d")
        start_d_idx = self.prices.index.get_loc(start_d)
        last_d = self.prices.index[start_d_idx - 1]
        
        market_info = self.get_market_info(start_d, last_d)

        s = {
            "date": start_d,
            "cash": start_cash,
            "holdings": [],
            "market": market_info
        }

        return s

    def get_market_info(self, start_d, last_d):
        market_info = []
        for ticker in self.watch_list:
            start_price = self.prices.Close[ticker][start_d].round(2)
            last_price = self.prices.Close[ticker][last_d].round(2)
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

    def step(self, s: State, a: list[Action]):
        s_ = deepcopy(s)
        penalty = 0

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
