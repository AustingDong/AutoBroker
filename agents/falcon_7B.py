from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline
from utils.parser import parse_json_array, parse_input

class Falcon7BAgent:
    def __init__(self, name="Falcon-7B"):
        self.name = name
        self.pipe = pipeline("text-generation", model="tiiuae/Falcon3-7B-Instruct", max_new_tokens=512)

    def step(self, market_state):
        formatted_prompt = parse_input(market_state)
        output = self.pipe(formatted_prompt)[0]["generated_text"].split("<|assistant|>")[-1]
        print("🔹 Raw output:\n", output)

        parsed_actions = parse_json_array(output)
        return parsed_actions



