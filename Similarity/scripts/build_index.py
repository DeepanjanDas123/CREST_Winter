# End-to-end ingestion pipeline.
import pandas as pd
import logging
import sys
from pathlib import Path
from src.config import config
from src.featurizer import canonicalize_dataset
from src.embedder import TextEmbedder
from src.indexer import VectorIndex
from src.storage import save_artifacts

logging.basicConfig(level=logging.INFO)

def main():
    data_path = Path(config.DATA_DIR) / "startups_raw.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Please add your data.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logging.info(f"Loaded {len(df)} startups from {data_path}")

    canonicals = canonicalize_dataset(df)
    logging.info("Canonicalization complete")

    embedder = TextEmbedder()
    embeddings = embedder.embed_startups(canonicals)

    index = VectorIndex(dim=embeddings.shape[1])
    index.build(embeddings)

    save_artifacts(canonicals, embeddings, index, config.OUTPUT_DIR)
    logging.info(f"✅ Artifacts saved to {config.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()