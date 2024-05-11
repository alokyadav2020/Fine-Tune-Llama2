from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    local_file_path:Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path   
    local_file_validation: Path 


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path    
    local_data_file: Path



@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: str
    traning_data_file: Path   
    model_check_point: str 
    model_name: str
    tokenizer_name: str


@dataclass(frozen=True)
class TraningArgumentConfig:
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    gradient_checkpointing: bool
    optim: str
    save_steps: int
    logging_steps: int
    learning_rate: float
    weight_decay: float
    fp16: bool
    bf16: bool
    max_grad_norm: float
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    lr_scheduler_type: str
    report_to: str
    max_seq_length: None
    packing: bool


@dataclass(frozen=True)
class LoraCongif:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool
    lora_alpha: int
    lora_dropout: float
    lora_r: int
    bias: str
    task_type: str

@dataclass(frozen=True)
class ModelPredictionConfig:
    model_name: Path
    tokenizer_name: Path    
