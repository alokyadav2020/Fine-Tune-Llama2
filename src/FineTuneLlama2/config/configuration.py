from src.FineTuneLlama2.constants import *
from src.FineTuneLlama2.logging import logger
from src.FineTuneLlama2.utils.common import read_yaml,create_directories
from src.FineTuneLlama2.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataTransformationConfig
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