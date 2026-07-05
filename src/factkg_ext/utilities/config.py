from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    """Application configuration."""
    llm_model: str = "gpt-4o"
    embedding_model: str = "all-MiniLM-L6-v2"
    data_path: str = "data/"
    gap1_config: Optional[Dict[str, Any]] = None
    gap2_config: Optional[Dict[str, Any]] = None
    gap3_config: Optional[Dict[str, Any]] = None

def load_config(path: str) -> AppConfig:
    """Load configuration from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return AppConfig(**(data or {}))
