from src.FineTuneLlama2.constants import *
from src.FineTuneLlama2.logging import logger
from src.FineTuneLlama2.utils.common import read_yaml,check_mongo_server_connection
from src.FineTuneLlama2.entity.server_entity import Mongo_DB_Server
from dotenv import load_dotenv
import os

load_dotenv()
class ServerConfigurationManager:
    def __init__(self, SrvrCnfg = SERVER_CONFIG_FILE_PATH):
        self.Sconfig = read_yaml(SrvrCnfg)
        self.Musername= os.getenv("mongo_username")
        self.Mpassword = os.getenv("mongo_password")
        


    def return_mongoconfiguration(self)-> Mongo_DB_Server:

        URI = f"mongodb+srv://{self.Musername}:{self.Mpassword}@cluster0.u3qomry.mongodb.net/?retryWrites=true&w=majority"

        # mongo_username=os.getenv("mongo_username")
        # mongo_password=os.getenv("mongo_password")
        if check_mongo_server_connection(URI):
            mongo_db_sever_varirables = Mongo_DB_Server(

                mongodb_url=URI,
                mongodb_name=self.Sconfig.MONGODB.mongodb_name,
                mongodb_collection_name=self.Sconfig.MONGODB.mongodb_collection_name

            )
            return mongo_db_sever_varirables
        else :
            print("Connection not atablished +++++====  Return From ServerConfigManger ++++++======")    

        

