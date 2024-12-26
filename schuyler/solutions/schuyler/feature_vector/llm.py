from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer, util

class LLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
        self.model_name = model_name
        if "llama" in model_name:
            config = AutoConfig.from_pretrained(model_name)
            config._attn_implementation = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.float16, device_map="auto"
        )
    
    def predict(self, inputs, max_length=200, temperature=0.3, top_p=0.95, sample=True):
        print(f"Querying LLM model {self.model_name} with input: {inputs}")
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=sample,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class SentenceTransformerModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text):
        return self.model.encode(text, convert_to_tensor=True)