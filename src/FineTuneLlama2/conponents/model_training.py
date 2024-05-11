import os
from src.FineTuneLlama2.config.configuration import ModelTrainingConfig,TraningArgumentConfig,LoraCongif
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch
from datasets import load_from_disk
from transformers import (  AutoModelForCausalLM,
                            AutoTokenizer,
                            BitsAndBytesConfig,
                            HfArgumentParser,
                            TrainingArguments,
                            pipeline,
                            logging,) 




class ModelTraining:
    def __init__(self,config:ModelTrainingConfig, param: TraningArgumentConfig, lora_config: LoraCongif) -> None:
        self.config = config
        self.param = param
        self.lora_config = lora_config
        compute_dtype = getattr(torch, self.lora_config.bnb_4bit_compute_dtype)
        if compute_dtype == torch.float16 and self.lora_config.load_in_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

    # Load Base Model
    def Base_model(self):

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=self.lora_config.load_in_4bit,
        bnb_4bit_quant_type=self.lora_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=self.lora_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=self.lora_config.bnb_4bit_use_double_quant,
        )


        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_check_point,
            quantization_config = bnb_config,
            device_map = {"": 0}
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        return model
    
    # Load Llama2 tokenizer

    def Llama_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_check_point, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return tokenizer
    

    # Load LoRA Configuration

    def peft_Lora_configuration(self):
        peft_config = LoraConfig(
        lora_alpha=self.lora_config.lora_alpha,
        lora_dropout=self.lora_config.lora_dropout,
        r=self.lora_config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
       )
        
        return peft_config
        
    def Load_dataset(self):

        dataset = load_from_disk(self.config.traning_data_file)
        dataset = dataset.select(range(2))

        return dataset
    
    def Training_Argument(self):
        training_arguments = TrainingArguments(
        output_dir=self.config.model_name,
        num_train_epochs=self.param.num_train_epochs,
        per_device_train_batch_size=self.param.per_device_train_batch_size,
        gradient_accumulation_steps=self.param.gradient_accumulation_steps,
        optim=self.param.optim,
        save_steps=self.param.save_steps,
        logging_steps=self.param.logging_steps,
        learning_rate=self.param.learning_rate,
        weight_decay=self.param.weight_decay,
        fp16=self.param.fp16,
        bf16=self.param.bf16,
        max_grad_norm=self.param.max_grad_norm,
        max_steps=self.param.max_steps,
        warmup_ratio=self.param.warmup_ratio,
        group_by_length=self.param.group_by_length,
        lr_scheduler_type=self.param.lr_scheduler_type,
        report_to="tensorboard"
        )

        return training_arguments
    
    def Start_Traning_Model(self):
        torch.cuda.empty_cache()
        
        dataset = self.Load_dataset()
        model = self.Base_model()
        tokenizer = self.Llama_tokenizer()
        peft_config= self.lora_config()
        training_arguments = self.Training_Argument()

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.param.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.param.packing,

           )

        trainer.train()

        trainer.model.save_pretrained(self.config.model_name)

        del model
        del trainer
        import gc
        gc.collect()
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(
        self.config.model_check_point,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
        )
        model = PeftModel.from_pretrained(base_model, self.config.model_name)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_check_point, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.save_pretrained(self.config.tokenizer_name)









