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