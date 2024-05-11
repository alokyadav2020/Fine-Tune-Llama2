from src.FineTuneLlama2.config.configuration import ConfigurationManager
from transformers import AutoTokenizer,pipeline



class GenerationPipeline:
    def __init__(self) -> None:
        self.config = ConfigurationManager().get_prediction_config()



    def predict(self,text):

        pipe = pipeline('text-generation',model=self.config.model_name, tokenizer=self.config.tokenizer_name, max_length=200)
        print("Text :")
        print(text)
        result = pipe(f"<s>[INST] {text} [/INST]")
        print(result[0]['generated_text'])
        return result
