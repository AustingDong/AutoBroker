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
        self.batch_size = 20

        self.ppo_config = PPOConfig(
            learning_rate=5e-6,
            batch_size=self.batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        # response = self.pipe(formatted_prompt)[0]["generated_text"].split("<|assistant|>")[-1]
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = self.model.generate(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        max_new_tokens=512
                                        )
        response = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].split("<|assistant|>")[-1]
        print("🔹 Raw output:\n", response)

        parsed_actions = parse_json_array(response)
        return parsed_actions, inputs, outputs



