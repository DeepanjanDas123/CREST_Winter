# src/embedder.py
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .models import StartupCanonical
from .config import config
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        logger.debug(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def embed_startups(self, startups: List[StartupCanonical]) -> np.ndarray:
        texts = [s.canonical_description for s in startups]
        return self.embed(texts)