from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils.parser import parse_json_array, parse_input
import torch

class Falcon7BAgent:
    def __init__(self, name="Falcon-7B"):
        self.name = name

        model_name = "tiiuae/Falcon-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"             
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256
        )

    def step(self, market_state):
        formatted_prompt = parse_input(market_state)
        output = self.pipe(formatted_prompt)[0]["generated_text"].split("<|assistant|>")[-1]
        print("🔹 Raw output:\n", output)

        parsed_actions = parse_json_array(output)
        return parsed_actions



