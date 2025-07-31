from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel
from utils.parser import parse_json_array, parse_input
import torch
import copy
from peft import LoraConfig, get_peft_model
import json
import os

class Falcon7BAgent:
    def __init__(self, name="Falcon-7B", batch_size=1, n_steps=1, dtype=torch.float32):
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
            max_grad_norm=1.0,
            gradient_accumulation_steps=6,
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
                                            max_new_tokens=384,
                                            do_sample=True,
                                            temperature=0.95, 
                                            top_p=0.9,
                                            return_dict_in_generate=False,
                                            )
            
        
        decoded_outputs = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # print(decoded_outputs)
        parsed_actions = []
        for output in decoded_outputs:
            response = output.split("<|assistant|>")[-1].strip()
            actions = parse_json_array(response)
            parsed_actions.append(actions)

        # print(parsed_actions)
        return parsed_actions, inputs, outputs


    def save(self, save_dir: str, save_ref_value_head: bool = False) -> None:
        """
        Save everything needed to resume training or run inference:
        - tokenizer
        - LoRA policy adapter (the trainable part)
        - policy value head
        - optional: ref model's value head (if you want bit‑for‑bit PPO resumption)
        - minimal metadata for reproducible reload
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # 2) policy LoRA adapter (inside the TRL wrapper, the PEFT-wrapped LM lives at .pretrained_model)
        self.model.pretrained_model.save_pretrained(f"{save_dir}/policy_adapter")

        # 3) policy value head
        torch.save(self.model.v_head.state_dict(), f"{save_dir}/policy_value_head.pt")

        # 4) (optional) reference value head to continue PPO exactly
        if save_ref_value_head:
            torch.save(self.ref_model.v_head.state_dict(), f"{save_dir}/ref_value_head.pt")

        # 5) minimal metadata (adjust if you change LoRA/quant settings)
        meta = {
            "base": "tiiuae/Falcon3-7B-Instruct",
            "quant": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                # We record desired compute dtype; you may run in fp32 for safety.
                "bnb_4bit_compute_dtype": str(self.dtype).split(".")[-1]
            },
            "lora": {
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "task_type": "CAUSAL_LM"
            }
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_for_inference(save_dir: str, device_map: str = "auto"):
        """
        Reload tokenizer + policy (adapter + value head) for inference.
        If you want to resume PPO, recreate the ref model separately and
        load 'ref_value_head.pt' as well.
        """
        

        meta = json.load(open(os.path.join(save_dir, "meta.json")))
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=meta["quant"]["load_in_4bit"],
            bnb_4bit_use_double_quant=meta["quant"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=meta["quant"]["quant_type"],
            # use fp32 compute by default to be safe; you can change to bf16 later
            bnb_4bit_compute_dtype=torch.float32,
        )

        peft_config = LoraConfig(
            r=meta["lora"]["r"],
            target_modules=meta["lora"]["target_modules"],
            task_type=meta["lora"]["task_type"],
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            meta["base"], quantization_config=bnb_config, device_map=device_map, peft_config=peft_config
        )

        # apply trained LoRA
        model.pretrained_model = PeftModel.from_pretrained(
            model.pretrained_model, os.path.join(save_dir, "policy_adapter")
        )

        # load the policy value head
        vh = torch.load(os.path.join(save_dir, "policy_value_head.pt"), map_location="cpu")
        model.v_head.load_state_dict(vh, strict=True)
        model.eval()
        return tokenizer, model