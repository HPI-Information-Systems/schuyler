from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import InputExample, losses, models, SentenceTransformer, SentenceTransformerTrainingArguments,SentenceTransformerTrainer
from torch.utils.data import DataLoader
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

import numpy as np
import random
from datasets import DatasetDict

import os
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
    
    def predict(self, inputs, max_length=2000, temperature=0.3, top_p=0.95, sample=True):
        print(f"Querying LLM model {self.model_name} with input: {inputs}")
        print(self.model.device)
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
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        print(model_name)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42)
        np.random.seed(42)

        self.model = SentenceTransformer(model_name, device='cuda:0')
        #self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        # self.model.eval()  # Disable dropout
    
    def encode(self, text):
        return self.model.encode(text, convert_to_tensor=True)
    
    def finetune(self, dataset, epochs=4, warmup_steps=100):
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 10% for eval

        # Create a DatasetDict with 'train' and 'eval' splits
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        })
        # train_loss = losses.GISTEmbedLoss(model=self.model, guide=self.model)
        train_loss = losses.TripletLoss(model=self.model)
        output_path = "/data/models/mpnet-base-all-nli-triplet"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        args = SentenceTransformerTrainingArguments(
            output_dir=output_path,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  
            bf16=False, 
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            evaluation_strategy="steps",  # Ensure evaluations are performed
            eval_steps=50,  # Reduced for more frequent evaluations
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,  # Reduced for more frequent logging
            run_name="mpnet-base-all-nli-triplet",
            report_to=["wandb"],  # Enable W&B reporting
        )
        dev_evaluator = TripletEvaluator(
            anchors=dataset["eval"]["anchor"],
            positives=dataset["eval"]["positive"],
            negatives=dataset["eval"]["negative"],
            name="all-nli-dev",
        )
        dev_evaluator(self.model)

        # Create the trainer with both train and eval datasets
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],  # Provide the eval dataset here
            loss=train_loss,
            evaluator=dev_evaluator,
        )
        trainer.train()
        test_evaluator = TripletEvaluator(
            anchors=dataset["eval"]["anchor"],  # Replace with your test data if different
            positives=dataset["eval"]["positive"],
            negatives=dataset["eval"]["negative"],
            name="all-nli-test",
        )
        test_evaluator(self.model)
        self.model.save('/data/fine-tuned-sentence-transformer')