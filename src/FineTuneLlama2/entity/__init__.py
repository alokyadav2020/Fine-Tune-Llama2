from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    local_file_path:Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path   
    local_file_validation: Path 


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path    
    local_data_file: Path



