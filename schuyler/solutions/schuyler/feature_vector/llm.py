import torch
import gc
import os
import sys
from sentence_transformers import losses, SentenceTransformer, SentenceTransformerTrainingArguments,SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import numpy as np
import random
from datasets import DatasetDict
import traceback
import multiprocessing as mp
from typing import List, Optional, Any, Dict

def _child_worker(
    prompts: List[str],
    model_name: str,
    max_model_len: int,
    gpu_mem_util: float,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    chat_template_kwargs: Optional[Dict[str, Any]],
    result_queue: mp.SimpleQueue,  # <- SimpleQueue-Typhinweis (optional)
):
    """
    Läuft im Child-Prozess: erstellt LLM, führt chat() aus, gibt Resultate über SimpleQueue zurück und fährt sauber herunter.
    """
    try:
        # Frühe Crash-Dumps bei harten Fehlern (Segfault, Thread deadlocks)
        import faulthandler
        faulthandler.enable(file=sys.stderr, all_threads=True)

        print("Child: starting, importing vLLM...")
        from vllm import LLM, SamplingParams

        print("Child: constructing LLM...")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_seq_len_to_capture=2**15, max_model_len=2**15,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=True,  # falls unterstützt
        )

        print(f"Child: building conversations for {len(prompts)} prompt(s)")
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=32768,
        )

        print("Child: running llm.chat()...")
        outputs = llm.chat(
            conversations,
            sp,
            chat_template_kwargs=(chat_template_kwargs or {}),
        )
        print("Child: llm.chat() done.")

        # Nur Strings zurückgeben
        texts = [
            (o.outputs[0].text.strip() if (o.outputs and len(o.outputs) > 0) else "")
            for o in outputs
        ]
        print(f"Child: produced {len(texts)} texts.")

        # vLLM sauber beenden
        print("Child: shutting down vLLM...")
        try:
            llm.shutdown()
        except Exception as e:
            print(f"Child: error during vLLM shutdown: {e!r}")

        # Python/CUDA-Aufräumen
        del outputs
        del llm
        gc.collect()
        print("Child: collecting CUDA/GC memory...")

        # Ergebnis in die SimpleQueue legen (nicht blockierend ist hier nicht nötig,
        # da der Parent vorher `get()` macht – siehe predict()).
        result_queue.put({"ok": True, "data": texts})
        print("Child: result put into queue. Exiting now.")
        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        # Fehler zurückreichen
        try:
            result_queue.put({"ok": False, "error": f"{e}", "traceback": tb})
        except Exception:
            # Wenn sogar das Put fehlschlägt, bleibt wenigstens der Exitcode > 0
            pass
        # Explizit mit Fehlercode beenden
        sys.exit(1)


class vLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        start_method: str = "spawn",
        default_top_k: int = 20,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.__name__ = "vLLM"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.default_top_k = default_top_k
        self.chat_template_kwargs = chat_template_kwargs or {"enable_thinking": False}

        try:
            self.mp_ctx = mp.get_context(start_method)
        except ValueError:
            self.mp_ctx = mp.get_context()

    def predict(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        sample: bool = True,                 # bleibt für API-Kompatibilität
        timeout_seconds: Optional[int] = None,
    ) -> List[str]:
        print(f"Parent: Querying {self.model_name} with {len(prompts)} prompt(s) via subprocess")
        result_queue = self.mp_ctx.SimpleQueue()

        p = self.mp_ctx.Process(
            target=_child_worker,
            args=(
                prompts,
                self.model_name,
                self.max_model_len,
                self.gpu_memory_utilization,
                temperature,
                top_p,
                self.default_top_k,
                max_tokens,
                self.chat_template_kwargs,
                result_queue,
            ),
            daemon=False,
        )
        p.start()
        result = None
        try:
            if timeout_seconds is not None:
                result = result_queue.get(timeout=timeout_seconds)
            else:
                result = result_queue.get()
        except Exception as e:
            print(f"Parent: get() failed or timed out: {e!r}. Terminating child.")
            if p.is_alive():
                p.terminate()
            p.join()
            raise TimeoutError(
                f"Inference subprocess did not return a result within timeout ({timeout_seconds}s) or failed."
            )

        print("Parent: got result from queue, now joining child...")
        p.join()

        print(f"Parent: child exitcode = {p.exitcode}")
        if p.exitcode is None:
            print("Parent: child still alive unexpectedly; terminating.")
            p.terminate()
            p.join()
            raise RuntimeError("Child process did not terminate as expected.")

        if p.exitcode < 0:
            raise RuntimeError(f"Child was terminated by signal {p.exitcode} (likely OOM if -9).")

        if result is None:
            raise RuntimeError("Inference subprocess ended without returning a result (result=None).")

        if not result.get("ok", False):
            tb = result.get("traceback", "")
            err = result.get("error", "Unknown error in child process.")
            raise RuntimeError(f"Child process error: {err}\n{tb}")

        data = result.get("data", [])
        print("Parent: returning data.")
        return data

class SentenceTransformerModel:
    def __init__(self, database, model_name="sentence-transformers/all-mpnet-base-v2"):
        print(model_name)
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42)
        np.random.seed(42)
        self.database = database
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name, device='cuda:0')
    
    def encode(self, table):
        return np.asarray(self.model.encode(table.llm_description, convert_to_tensor=True).cpu())
    
    def finetune(self, triplets, triplet_model, seed=42):
        dataset = triplet_model.enrich_triplets(triplets)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=seed)  # 10% for eval

        dataset = DatasetDict({
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        })
        train_loss = losses.TripletLoss(model=self.model)
        output_path = "/data/models/mpnet-base-all-nli-triplet"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        args = SentenceTransformerTrainingArguments(
            output_dir=output_path,
            num_train_epochs=1,#5
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            warmup_ratio=0.1,
            fp16=True,  
            bf16=False, 
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
            run_name="mpnet-base-all-nli-triplet",
            report_to=["wandb"],
        )
        anchors   = list(dataset["eval"]["anchor"])
        positives = list(dataset["eval"]["positive"])
        negatives = list(dataset["eval"]["negative"])
        dev_evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="all-nli-dev",
        )
        print(dataset["eval"]["anchor"])
        dev_evaluator(self.model)
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            loss=train_loss,
            evaluator=dev_evaluator,
        )
        trainer.train()
        anchors   = list(dataset["eval"]["anchor"])
        positives = list(dataset["eval"]["positive"])
        negatives = list(dataset["eval"]["negative"])
        test_evaluator = TripletEvaluator(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            name="all-nli-test",
        )
        test_evaluator(self.model)
        self.model.save('/data/fine-tuned-sentence-transformer')
        del trainer, train_loss, dev_evaluator, test_evaluator
        torch.cuda.empty_cache()
        gc.collect()