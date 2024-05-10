import pandas as pd
import os
from pathlib import Path
from pymongo import MongoClient
from src.FineTuneLlama2.logging import logger
from src.FineTuneLlama2.entity import DataIngestionConfig
from src.FineTuneLlama2.config.mongoDbConfiguration import ServerConfigurationManager
from src.FineTuneLlama2.utils.common import get_size


class DataIngestion:
    def __init__(self,Dconfig : DataIngestionConfig) -> None:
        self.data_config = Dconfig
       



    def Download_Data(self):

        server_obj = ServerConfigurationManager()
        S_Variables=server_obj.return_mongoconfiguration()

        if not os.path.exists(self.data_config.local_file_path):
            mongo_client = MongoClient(S_Variables.mongodb_url)
            collection = mongo_client[S_Variables.mongodb_name][S_Variables.mongodb_collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            logger.info(f"Data Download successful from mongo db server{df.head()}")
            print()
            print("===== Data Collected From Mongo DB Server =====")
            print(df.head())
            print("===== ================ ================== =====")
            print()

            df.to_csv(self.data_config.local_file_path,index=False)
            print(df.head())
            logger.info("Data saved successful in form of CSV file")

        else: 
            logger.info(f"File already exists of size: {get_size(Path(self.data_config.local_file_path))}")   