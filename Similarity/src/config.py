# Configuration with defaults and environment override support.
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM: int = 768

    # Retrieval
    K_NEIGHBORS: int = 20

    # Hybrid similarity weights (should sum to ~1.0)
    ALPHA_TEXT: float = 0.6
    BETA_BUSINESS_MODEL: float = 0.2
    GAMMA_SECTOR: float = 0.2

    # Paths
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "output"

    # LLM
    LLM_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_SEC: int = 10

# Singleton config instance
config = AppConfig()