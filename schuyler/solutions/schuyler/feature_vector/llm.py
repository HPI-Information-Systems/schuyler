from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import InputExample, losses, models, SentenceTransformer, SentenceTransformerTrainingArguments,SentenceTransformerTrainer
from torch.utils.data import DataLoader
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from schuyler.solutions.tuta.convert_df_to_jsonl import write_to_jsonl, dataframe_to_wikitables_json
from schuyler.solutions.tuta.dynamic_data import DynamicDataLoader#
from dataclasses import dataclass
import wandb
from pytorch_lightning.loggers import WandbLogger
from schuyler.solutions.tuta.prepare import WikiDataset
from schuyler.solutions.tuta.triplet_dataset import TripletDataset
from torch.utils.data import random_split, DataLoader
from schuyler.solutions.tuta.lightning import TUTALightningModule
from schuyler.solutions.tuta.model.pretrains import TUTAForTriplet
from schuyler.solutions.tuta.params import TutaParams, TutaPrepareParams
import numpy as np
import random
from datasets import DatasetDict
import pytorch_lightning as pl
from openai import OpenAI


import os
class LLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
        self.model_name = model_name
        self.__name__ = "LLM"
        if "llama" in model_name:
            config = AutoConfig.from_pretrained(model_name)
            config._attn_implementation = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.float16, device_map={"": 0}
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

class ChatGPT:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.__name__ = "ChatGPT"
        self.client = OpenAI()
    
    def predict(self, inputs, max_length=2000, temperature=0.3, top_p=0.95, sample=True):
        print(f"Querying LLM model {self.model_name} with input: {inputs}")
        inputs = inputs.replace("Description:", " ")
        inputs += f"Please only return the requested description and no additional text. Do not replicate the given information, but focus on writing a coherent and precise description of the table, incorporating all given information. Contextualize it."
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": inputs}
            ],
            temperature=temperature,
        )
        print("Response from LLM:", response)
        return response.choices[0].message.content

    
class SentenceTransformerModel:
    def __init__(self, database, model_name="sentence-transformers/all-mpnet-base-v2"):
        print(model_name)

        #! wieder einkommentieren
        # torch.manual_seed(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # random.seed(42)
        # np.random.seed(42)
        self.database = database
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name, device='cuda:0')
        #self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        # self.model.eval()  # Disable dropout
    
    def encode(self, table):
        return np.asarray(self.model.encode(table.llm_description, convert_to_tensor=True).cpu())
    
    def finetune(self, triplets, triplet_model, seed=42):
        dataset = triplet_model.enrich_triplets(triplets)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=seed)  # 10% for eval

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
            num_train_epochs=4,#5
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,#5e-5
            warmup_ratio=0.1,
            fp16=True,  
            bf16=False, 
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            evaluation_strategy="steps",
            eval_steps=50,
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

class TutaModel:
    def __init__(self, database):
        tp = TutaParams()
        ptp = TutaPrepareParams
        assert torch.cuda.is_available(), "No available GPUs." 
        # single GPU mode.
        tp.gpu_id = tp.gpu_ranks[0]
        assert tp.gpu_id <= torch.cuda.device_count(), "Invalid specified GPU device." 
        tp.dist_train = False
        tp.single_gpu = True
        ptp.gpu_id = 0
        assert ptp.gpu_id <= torch.cuda.device_count(), "Invalid specified GPU device." 
        ptp.dist_train = False
        ptp.single_gpu = True
        print("Using single GPU: {} for training.".format(ptp.gpu_id))
        tables = sorted(database.get_tables(), key=lambda x: x.table_name)
        output_path = write_to_jsonl(tables, 1,"/experiment/schuyler/solutions/tuta/table.jsonl", "table.table_name")
        ptp.input_path = output_path
        dataset_path = WikiDataset(ptp).build_and_save(TutaPrepareParams.processes_num)
        data = DynamicDataLoader(tp, 0, 1, False).load_data()

        self.table_dict = {}
        for i, table in enumerate(tables):
            #print(table)
            self.table_dict[table.table_name] = data[i]
        tuta_model = TUTAForTriplet(tp)
        self.model = TUTALightningModule(tuta_model, margin=0.5, lr=1e-5)
        # self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        # print(data[0])
        # with torch.no_grad():

        #     s = self.model(*data[0])
        # print(s)
        #raise ValueError("stop")
        

    def encode(self, table):
        input = self.table_dict[table.table_name]
        data = []
        for i in range(len(input)):
            #.unsqueeze(0)
            data.append(input[i].unsqueeze(0))
        # print(np.asarray(data).shape)
        #move data to cuda
        data = [d.cuda() for d in data]
        with torch.no_grad():
            s = torch.squeeze(self.model(data), 0)
        return s.cpu().numpy()
    
    def finetune(self, triplets, triplet_model):
        dataset = TripletDataset(triplets, self.table_dict)
        dataset_size = len(dataset)
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=12)
        val_dataloader = DataLoader(val_dataset, batch_size=12)

        
        wandb_logger = WandbLogger(
            project="schuyler",
            log_model=True
        )
        trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1,enable_progress_bar=True,  enable_checkpointing=True,callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],  logger=wandb_logger)

        trainer.fit(self.model, train_dataloader, val_dataloader)