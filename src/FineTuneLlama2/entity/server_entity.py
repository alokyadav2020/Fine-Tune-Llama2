from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mongo_DB_Server:
    mongodb_url:str
    mongodb_name:str
    mongodb_collection_name:str



@dataclass(frozen=True)
class AWS_Server:
        aws_s3_bucket_name:str