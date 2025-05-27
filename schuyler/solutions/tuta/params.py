from schuyler.solutions.tuta import tokenizer as tknr

args = {
    "vocab_path": "/experiment/schuyler/solutions/tuta/vocab/bert_vocab.txt",
    "context_repo_path": "/experiment/schuyler/solutions/tuta/vocab/context_repo_init.txt",
    "cellstr_repo_path": "/experiment/schuyler/solutions/tuta/vocab/cellstr_repo_init.txt",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "magnitude_size": 10,
    "precision_size": 10,
    "top_digit_size": 10,
    "low_digit_size": 10,
    "row_size": 256,
    "column_size": 256,
    "tree_depth": 4,
    "node_degree": [32,32,64,256],
    "num_format_feature": 11,
    "attention_distance": 8,
    "attention_step": 0,
    "num_attention_heads": 12,
    "num_encoder_layers": 12,
    "num_tcr_type": 2,
    "hidden_dropout_prob": 0.1,
    "attention_dropout_prob": 0.1,
    "layer_norm_eps": 1e-06,
    "hidden_act": "gelu",
    "target": "tuta",
    "attn_method": "add",
    "max_seq_len": 2048,
    "max_cell_num": 1024,
    "max_cell_length": 256,
    "max_disturb_num": 20,
    "disturb_prob": 0.15,
    "add_separate": True,
    "text_threshold": 0.5,
    "value_threshold": 0.1,
    "clc_rate": 0.3,
    "hier_or_flat": "both",
    "wcm_rate": 0.3,
    "clc_weight": 1.0,
    "batch_size": 2,
    "total_steps": 1000000,
    "report_steps": 100,
    "save_checkpoint_steps": 100000,
    "buffer_size": 500000,
    "chunk_size": 50000,
    "dataset_paths": "/experiment/schuyler/solutions/tuta/dataset.pt",
    "pretrained_model_path": None,
    "load_type": "tuta",
    "output_model_path": "tuta.bin",
    "warmup": 0.1,
    "learning_rate": 2e-05,
    "head_dropout_prob": 0.1,
    "world_size": 1,
    "gpu_ranks": [0],
    "master_ip": "tcp://localhost:12345",
    "backend": "nccl",
    }

class TutaParams:
    vocab_path: str = args["vocab_path"]
    context_repo_path: str = args["context_repo_path"]
    cellstr_repo_path: str = args["cellstr_repo_path"]
    hidden_size: int = args["hidden_size"]
    intermediate_size: int = args["intermediate_size"]
    magnitude_size: int = args["magnitude_size"]
    precision_size: int = args["precision_size"]
    top_digit_size: int = args["top_digit_size"]
    low_digit_size: int = args["low_digit_size"]
    row_size: int = args["row_size"]
    column_size: int = args["column_size"]
    tree_depth: int = args["tree_depth"]
    node_degree: str = args["node_degree"]
    num_format_feature: int = args["num_format_feature"]
    attention_distance: int = args["attention_distance"]
    attention_step: int = args["attention_step"]
    num_attention_heads: int = args["num_attention_heads"]
    num_encoder_layers: int = args["num_encoder_layers"]
    num_tcr_type: int = args["num_tcr_type"]
    hidden_dropout_prob: float = args["hidden_dropout_prob"]
    attention_dropout_prob: float = args["attention_dropout_prob"]
    layer_norm_eps: float = args["layer_norm_eps"]
    hidden_act: str = args["hidden_act"]
    target: str = args["target"]
    attn_method: str = args["attn_method"]
    max_seq_len: int = args["max_seq_len"]
    max_cell_num: int = args["max_cell_num"]
    max_cell_length: int = args["max_cell_length"]
    max_disturb_num: int = args["max_disturb_num"]
    disturb_prob: float = args["disturb_prob"]
    add_separate: bool = args["add_separate"]
    text_threshold: float = args["text_threshold"]
    value_threshold: float = args["value_threshold"]
    clc_rate: float = args["clc_rate"]
    hier_or_flat: str = args["hier_or_flat"]
    wcm_rate: float = args["wcm_rate"]
    clc_weight: float = args["clc_weight"]
    batch_size: int = args["batch_size"]
    total_steps: int = args["total_steps"]
    report_steps: int = args["report_steps"]
    save_checkpoint_steps: int = args["save_checkpoint_steps"]
    buffer_size: int = args["buffer_size"]
    chunk_size: int = args["chunk_size"]
    dataset_paths: str = args["dataset_paths"]
    pretrained_model_path: str = args["pretrained_model_path"]
    load_type: str = args["load_type"]
    output_model_path: str = args["output_model_path"]
    warmup: float = args["warmup"]
    learning_rate: float = args["learning_rate"]
    head_dropout_prob: float = args["head_dropout_prob"]
    world_size: int = args["world_size"]
    gpu_ranks: list = args["gpu_ranks"]
    master_ip: str = args["master_ip"]

    def __init__(self):
        self.tokenizer = tknr.TutaTokenizer(self)
        self.vocab_size = len(self.tokenizer.vocab)
        self.dataset_paths = self.dataset_paths.split('+')

prepare_params = {
    "input_dir": "/experiment/schuyler/solutions/tuta/data/pretrain/spreadsheet",
    "input_path": "/experiment/schuyler/solutions/tuta/data/pretrain/wiki-table-samples.json",
    "source_type": "wiki",
    "cache_dir": "/experiment/schuyler/solutions/tuta/",

    "output_path": "/experiment/schuyler/solutions/tuta/dataset.pt",
    "vocab_path": "/experiment/schuyler/solutions/tuta/vocab/bert_vocab.txt",
    "context_repo_path": "/experiment/schuyler/solutions/tuta/vocab/context_repo_init.txt",
    "cellstr_repo_path": "/experiment/schuyler/solutions/tuta/vocab/cellstr_repo_init.txt",
    "magnitude_size": 10,
    "precision_size": 10,
    "top_digit_size": 10,
    "low_digit_size": 10,
    "row_size": 256,
    "column_size": 256,
    "tree_depth": 4,
    "node_degree": [32,32,64,256],
    "num_format_feature": 11,
    "attention_distance": 8,
    "attention_step": 0,
    "num_attention_heads": 12,
    "num_encoder_layers": 12,
    "num_tcr_type": 2,
    "hidden_dropout_prob": 0.1,
    
    "attention_dropout_prob": 0.1,
    "layer_norm_eps": 1e-6,

    "hidden_act": "gelu",
    "max_seq_len": 2048,
    "max_cell_num": 1024,
    "max_cell_length": 256,
    "max_disturb_num": 20,
    "disturb_prob": 0.15,
    "add_separate": True,
    "text_threshold": 0.5,
    "value_threshold": 0.1,
    "clc_rate": 0.3,
    "hier_or_flat": "both",
    "wcm_rate": 0.3,
    "buffer_size": 500000,
    "valid_num": 100000,
    "processes_num": 1
}

class TutaPrepareParams:
    input_dir: str = prepare_params["input_dir"]
    input_path: str = prepare_params["input_path"]
    source_type: str = prepare_params["source_type"]
    cache_dir: str = prepare_params["cache_dir"]
    output_path: str = prepare_params["output_path"]
    vocab_path: str = prepare_params["vocab_path"]
    context_repo_path: str = prepare_params["context_repo_path"]
    cellstr_repo_path: str = prepare_params["cellstr_repo_path"]
    magnitude_size: int = prepare_params["magnitude_size"]
    precision_size: int = prepare_params["precision_size"]
    top_digit_size: int = prepare_params["top_digit_size"]
    low_digit_size: int = prepare_params["low_digit_size"]
    row_size: int = prepare_params["row_size"]
    column_size: int = prepare_params["column_size"]
    tree_depth: int = prepare_params["tree_depth"]
    node_degree: str = prepare_params["node_degree"]
    num_format_feature: int = prepare_params["num_format_feature"]
    attention_distance: int = prepare_params["attention_distance"]
    attention_step: int = prepare_params["attention_step"]
    num_attention_heads: int = prepare_params["num_attention_heads"]
    num_encoder_layers: int = prepare_params["num_encoder_layers"]
    num_tcr_type: int = prepare_params["num_tcr_type"]
    hidden_dropout_prob: float = prepare_params["hidden_dropout_prob"]
    attention_dropout_prob: float = prepare_params["attention_dropout_prob"]
    layer_norm_eps: float = prepare_params["layer_norm_eps"]
    hidden_act: str = prepare_params["hidden_act"]
    max_seq_len: int = prepare_params["max_seq_len"]
    max_cell_num: int = prepare_params["max_cell_num"]
    max_cell_length: int = prepare_params["max_cell_length"]
    max_disturb_num: int = prepare_params["max_disturb_num"]
    disturb_prob: float = prepare_params["disturb_prob"]
    add_separate: bool = prepare_params["add_separate"]
    text_threshold: float = prepare_params["text_threshold"]
    value_threshold: float = prepare_params["value_threshold"]
    clc_rate: float = prepare_params["clc_rate"]
    hier_or_flat: str = prepare_params["hier_or_flat"]
    wcm_rate: float = prepare_params["wcm_rate"]
    buffer_size: int = prepare_params["buffer_size"]
    valid_num: int = prepare_params["valid_num"]
    processes_num: int = prepare_params["processes_num"]
