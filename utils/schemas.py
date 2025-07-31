from pydantic import BaseModel

class HoldingData(BaseModel):
    ticker: str
    quantity: int
    avg_purchase_price: float

class MarketData(BaseModel):
    ticker: str
    price: float
    change_pct: float
    volume: int
    ma7: float | None = None
    ma30: float | None = None
    mom7: float | None = None
    mom14: float | None = None

class State(BaseModel):
    date: str
    cash: float
    holdings: list[HoldingData]
    market: list[MarketData]

class Action(BaseModel):
    ticker: str
    activity: str  # "Buy", "Sell", or "Hold"
    score: int
    reason: str