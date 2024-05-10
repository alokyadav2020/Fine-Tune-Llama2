import os
from box.exceptions import BoxValueError
import yaml
from src.FineTuneLlama2.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from pymongo.mongo_client import MongoClient



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def check_mongo_server_connection(uri: str)->bool:
    """
    To check mongo db server connection status

    Args:
         uri: str

    Return:
        bool: True or False     
    
    
    """
    Connection_Status = None
    client = MongoClient(uri)

    try:
        client.admin.command('ping')
        Connection_Status = True
        
    except BoxValueError:
        raise ValueError("Mongo DB Connection NOT Stablised")
        Connection_Status = False
    except Exception as e:
        Connection_Status = False
        raise e
    
    return Connection_Status


def WriteFile(file_path: Path, txt: str):
       
    try: 
        with open(file_path, 'w') as fwrite:
                fwrite.write(txt)
    except BoxValueError:
            raise ValueError("file not exist")
    except Exception as e:
            raise e

    
