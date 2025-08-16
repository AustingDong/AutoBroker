import torch
import json
import re
from utils.schemas import State, Action
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

long_template = """You are a trading assistant AI. Based on market data, generate a list of trade actions in the JSON format described below.

    Each action must include:
    - ticker: stock symbol
    - activity: "Buy", "Sell", or "Hold"
    - quantity: number of changing (bought, sold) shares (integer)
    - reason: the reason why you do this action

    You shouldn't sell your holdings that exceeds the number of holdings you have.
    When you are buying, the total cost of purchase price cannot exceed your cash after sold.

    Only output a JSON list of such actions.

    {format_instructions}

    Market data:
    {market_state}
    """

short_template = """
You're a trading bot. Output a JSON list of trade actions.
Each action must include:
- ticker: stock symbol
- activity: "Buy", "Sell", or "Hold"
- score: integer 0~10
- reason: short explanation

Only output a JSON list of such actions.

{format_instructions}

Market data:
{market_state}
"""
def parse_input(market_state: list[State]) -> str:
    # Define the response schema
    response_schemas = [
        ResponseSchema(name="ticker", description="The stock symbol (e.g., AAPL)"),
        ResponseSchema(name="activity", description='The trade activity: "Buy", "Sell", or "Hold"'),
        ResponseSchema(name="score", description='''
                       A number within 0 to 10 to evaluate your action. The higher the stronger'''
                    ),
        ResponseSchema(name="reason", description="The reason why you do this action.")
    ]

    # Use LangChain's output parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template=short_template, # modify template here
        input_variables=["market_state"],
        partial_variables={"format_instructions": format_instructions}
    )
    formatted_prompt = prompt.format(market_state=market_state)
    return formatted_prompt


def parse_json_array(text: str) -> list[Action]:
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("❌ No JSON array found.")
        json_text = match.group(0)
        actions = json.loads(json_text)
        if not isinstance(actions, list):
            raise ValueError("❌ Not a list.")
        actions_obj = []
        for a in actions:
            actions_obj.append(Action(**a))
        return actions_obj
    except Exception as e:
        print("❌ Parse failed:", e)
        return []
    
def truncate_ids(input_ids: torch.Tensor, eos_id: int = 11) -> torch.Tensor:
    """
    Truncate the input_ids tensor to remove everything after the first occurrence of eos_id.
    """
    eos_indices = (input_ids == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        first_eos_index = eos_indices[0].item()
        return input_ids[:first_eos_index + 1]
    return input_ids