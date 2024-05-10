import os
from pathlib import Path
from src.FineTuneLlama2.logging import logger
from src.FineTuneLlama2.entity import DataValidationConfig
from src.FineTuneLlama2.utils.common import WriteFile





class DataValidation:
    def __init__(self, Vconfig: DataValidationConfig) -> None:
        self.Validation_config = Vconfig


    def Validation_Data(self):


        if os.path.exists(self.Validation_config.local_file_validation):
            WriteFile(self.Validation_config.status_file,"True")
                
            logger.info(f"Data file is available for data processing in {self.Validation_config.local_file_validation}")

        else:
            WriteFile(self.Validation_config.status_file,"False")    
            logger.info(f"Data file is not available for data processing in {self.Validation_config.local_file_validation}")