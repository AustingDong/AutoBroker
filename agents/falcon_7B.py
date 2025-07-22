from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from utils.parser import parse_json_array, parse_input
import torch
import copy
from peft import LoraConfig, get_peft_model

class Falcon7BAgent:
    def __init__(self, name="Falcon-7B", batch_size=64, n_steps=64, dtype=torch.float32):
        self.name = name
        self.dtype = dtype
        
        model_name = "tiiuae/Falcon3-7B-Instruct"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=self.dtype)
        self.batch_size = batch_size
        self.n_steps = n_steps

        self.ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=self.batch_size * self.n_steps,
            mini_batch_size=2,
            gradient_accumulation_steps=1,
            cliprange_value=0.2,
            cliprange=0.2,
            vf_coef=0.5,
            target_kl=1,
            ppo_epochs=2,
            optimize_cuda_cache=True,
            log_with=None,
            early_stopping=False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        
        
        # print(self.model)
        peft_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

        self.model = self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",            
            peft_config=peft_config)

        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"             
        ).eval()
        with torch.no_grad():
            for param in self.ref_model.parameters():
                param.requires_grad = False
                
        

        
        

    def step(self, market_state_lst):
        formatted_prompts = [parse_input(state) for state in market_state_lst]
        # response = self.pipe(formatted_prompt)[0]["generated_text"].split("<|assistant|>")[-1]
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            max_new_tokens=256,
                                            do_sample=True,
                                            temperature=1.0, 
                                            top_p=0.9,
                                            return_dict_in_generate=False,
                                            )
            
        
        decoded_outputs = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        parsed_actions = []
        for output in decoded_outputs:
            response = output.split("<|assistant|>")[-1].strip()
            actions = parse_json_array(response)
            parsed_actions.append(actions)

        return parsed_actions, inputs, outputs



