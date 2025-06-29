from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline
from utils.parser import parse_json_array, parse_input

class Qwen7BAgent:
    def __init__(self, name="Qwen-7B"):
        self.name = name
        self.pipe = pipeline("text-generation", model="Xiaojian9992024/Qwen2.5-Dyanka-7B-Preview")

    def step(self, market_state):
        formatted_prompt = parse_input(market_state)
        output = self.pipe(formatted_prompt)[0]
        print("🔹 Raw output:\n", output)

        parsed_actions = parse_json_array(output)
        if not parsed_actions:
            print("❌ No valid actions found in the output.")
        else:
            print("🔹 Parsed actions:")
            for action in parsed_actions:
                print(f"Ticker: {action['ticker']}, Activity: {action['activity']}, Quantity: {action['quantity']}")
        return parsed_actions



