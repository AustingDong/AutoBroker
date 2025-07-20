import json
import re
from utils.schemas import State, Action
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

def parse_input(market_state: list[State]) -> str:
    # Define the response schema
    response_schemas = [
        ResponseSchema(name="ticker", description="The stock symbol (e.g., AAPL)"),
        ResponseSchema(name="activity", description='The trade activity: "Buy", "Sell", or "Hold"'),
        ResponseSchema(name="quantity", description='''
                       A number within 0 to 10 of shares to trade (integer), 0 for Hold.
                       You shouldn't sell your holdings that exceeds the number of holdings you have. 
                       When you are buying, the total cost of purchase price cannot exceed your cash after sold.'''
                    ),
        ResponseSchema(name="reason", description="The reason why you do this action.")
    ]

    # Use LangChain's output parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""You are a trading assistant AI. Based on market data, generate a list of trade actions in the JSON format described below.

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
    """,
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