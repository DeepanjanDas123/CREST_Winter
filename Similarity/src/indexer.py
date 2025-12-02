# Unified vector index (FAISS or sklearn fallback).
import numpy as np
import logging
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from sklearn.neighbors import NearestNeighbors
from .config import config

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.index = None
        self.is_faiss = False
        self.num_vectors = 0

    def build(self, embeddings: np.ndarray):
        self.num_vectors = embeddings.shape[0]
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}")

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(self.dim)
            index.add(embeddings)
            self.index = index
            self.is_faiss = True
            logger.info(f"Built FAISS index with {self.num_vectors} vectors")
        else:
            k = min(config.K_NEIGHBORS, self.num_vectors)
            nn = NearestNeighbors(n_neighbors=k, metric="cosine")
            nn.fit(embeddings)
            self.index = nn
            self.is_faiss = False
            logger.info(f"Built sklearn NN index (FAISS not available)")

    def search(self, query_embeddings: np.ndarray, k: int = None) -> tuple[np.ndarray, np.ndarray]:
        if k is None:
            k = config.K_NEIGHBORS
        k = min(k, self.num_vectors)
        if self.is_faiss:
            return self.index.search(query_embeddings, k)
        else:
            distances, indices = self.index.kneighbors(query_embeddings, n_neighbors=k)
            return 1.0 - distances, indices  # convert distance → similarity