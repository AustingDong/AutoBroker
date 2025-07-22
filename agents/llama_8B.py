from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from utils.parser import parse_json_array, parse_input
import torch

class Llama8BAgent:
    def __init__(self, name="llama-7B"):
        self.name = name

        model_name = "meta-llama/Llama-3.1-8B"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.batch_size = 10

        self.ppo_config = PPOConfig(
            learning_rate=5e-6,
            batch_size=self.batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"             
        )

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"             
        )
        self.ref_model.eval()
        

    def step(self, market_state):
        formatted_prompt = parse_input(market_state)
        # messages = [
        #     {"role": "user", "content": formatted_prompt}
        # ]
        # inputs = self.tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        #     return_tensors="pt",
        # ).to(self.model.device)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("Output:")[-1]
        print("🔹 Raw output:\n", response)

        parsed_actions = parse_json_array(response)
        return parsed_actions, inputs, outputs



