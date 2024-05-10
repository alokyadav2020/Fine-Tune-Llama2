import os
from datasets import Dataset
from src.FineTuneLlama2.entity import DataTransformationConfig
from src.FineTuneLlama2.logging import logger
from pathlib import Path



class Datatransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config


    def transform_conversation(self,example):
        conversation_text = example['text']
        segments = conversation_text.split('###')

        reformatted_segments = []

        # Iterate over pairs of segments
        for i in range(1, len(segments) - 1, 2):
            human_text = segments[i].strip().replace('Human:', '').strip()

            # Check if there is a corresponding assistant segment before processing
            if i + 1 < len(segments):
                assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

                # Apply the new template
                reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
            else:
                # Handle the case where there is no corresponding assistant segment
                reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

        return {'text': ''.join(reformatted_segments)}   



    def save_transformed_data(self):
         
         logger.info(f"Load Dataset from {self.config.local_data_file}")
         dataset = Dataset.from_csv(self.config.local_data_file)
         logger.info('transform dataset as Llama2 model required')
         transformed_datset = dataset.map(self.transform_conversation)
         
         transformed_datset.save_to_disk(self.config.root_dir)
         logger.info(f"saved transformed dataset to {self.config.root_dir}")



