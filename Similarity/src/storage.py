# src/storage.py
import pickle
import numpy as np
from typing import List
from .models import StartupCanonical
from .indexer import VectorIndex
from pathlib import Path

VERSION = "v1.0"

def save_artifacts(
    canonicals: List[StartupCanonical],
    embeddings: np.ndarray,
    index: VectorIndex,
    output_dir: str = "output"
):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    meta = {"version": VERSION, "count": len(canonicals)}
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(out / "canonicals.pkl", "wb") as f:
        pickle.dump(canonicals, f)

    np.save(out / "embeddings.npy", embeddings)

    with open(out / "index.pkl", "wb") as f:
        pickle.dump(index, f)

def load_artifacts(output_dir: str = "output"):
    out = Path(output_dir)
    with open(out / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
        assert meta["version"] == VERSION, "Artifact version mismatch!"

    with open(out / "canonicals.pkl", "rb") as f:
        canonicals = pickle.load(f)

    embeddings = np.load(out / "embeddings.npy")
    with open(out / "index.pkl", "rb") as f:
        index = pickle.load(f)

    return canonicals, embeddings, index