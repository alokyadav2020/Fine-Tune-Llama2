from src.FineTuneLlama2.constants import *
from src.FineTuneLlama2.logging import logger
from src.FineTuneLlama2.utils.common import read_yaml,create_directories
from src.FineTuneLlama2.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig,
                                       TraningArgumentConfig,
                                       LoraCongif,
                                       ModelTrainingConfig,
                                       ModelPredictionConfig
                                       
                                       )


class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,param_filepath= PARAMS_FILE_PATH):
        self.config= read_yaml(config_filepath)
        self.param = read_yaml(param_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:    
        config = self.config.Data_Ingestion

        

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(

            root_dir= config.root_dir,
            local_file_path=config.local_file_path
        )

        return data_ingestion_config
    


    def get_data_valdation_config(self)-> DataValidationConfig:
        config = self.config.Data_Validaton

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(

            root_dir=config.root_dir,
            status_file=config.status_file,
            local_file_validation = config.local_file_validation
        )

        return data_validation_config
    

    def get_data_transfomation_config(self)->DataTransformationConfig:
        config= self.config.Data_Transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(

            root_dir=config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_transformation_config
    

    def get_model_training_config(self)-> ModelTrainingConfig:
        config = self.config.Model_training

        create_directories([config.root_dir])

        model_training_config= ModelTrainingConfig(

            root_dir= config.root_dir,
            traning_data_file=  config.traning_data_file,
            model_check_point= config.model_check_point,
            model_name= config.model_name,
            tokenizer_name= config.tokenizer_name


        )

        return model_training_config
    

    def get_trainingargumentconfig(self)->TraningArgumentConfig:
        param = self.param.TrainingArguments

        train_argument = TraningArgumentConfig(
                num_train_epochs= param.num_train_epochs,
                per_device_train_batch_size= param.per_device_train_batch_size,
                gradient_accumulation_steps= param.gradient_accumulation_steps,
                per_device_eval_batch_size= param.per_device_eval_batch_size,
                gradient_checkpointing= param.gradient_checkpointing,
                optim= param.optim,
                save_steps= param.save_steps,
                logging_steps= param.logging_steps,
                learning_rate= param.learning_rate,
                weight_decay= param.weight_decay,
                fp16= param.fp16,
                bf16= param.bf16,
                max_grad_norm= param.max_grad_norm,
                max_steps= param.max_steps,
                warmup_ratio= param.warmup_ratio,
                group_by_length= param.group_by_length,
                lr_scheduler_type= param.lr_scheduler_type,
                report_to= param.report_to,
                max_seq_length= param.max_seq_length,
                packing= param.packing

        )

        return train_argument
    

    def get_loraconfiguration(self)-> LoraCongif:
        param = self.param.LoraConfiguration

        lora_config = LoraCongif(
            load_in_4bit= param.load_in_4bit,
            bnb_4bit_quant_type= param.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype= param.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant= param.bnb_4bit_use_double_quant,
            lora_alpha= param.lora_alpha,
            lora_dropout= param.lora_dropout,
            lora_r= param.lora_r,
            bias= param.bias,
            task_type= param.task_type

        )

        return lora_config
    


    def get_prediction_config(self)-> ModelPredictionConfig:
        config= self.config.Model_Prediction

        prediction_config = ModelPredictionConfig(

            model_name= config.model_name,
            tokenizer_name= config.tokenizer_name
        )

        return prediction_config