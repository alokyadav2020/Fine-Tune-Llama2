from src.FineTuneLlama2.config.configuration import ConfigurationManager
from src.FineTuneLlama2.conponents.model_training import ModelTraining


class ModelTrainingPieline:
    def __init__(self) -> None:
        pass

    def main(self):
        config_file= ConfigurationManager()
        model_config= config_file.get_model_training_config()
        Arg_config = config_file.get_trainingargumentconfig()
        Lora_config = config_file.get_loraconfiguration()

        Training= ModelTraining(model_config, Arg_config,Lora_config)
        Training.Start_Traning_Model()
        