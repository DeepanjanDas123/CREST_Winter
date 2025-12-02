# scripts/query.py
import logging
import sys
from src.embedder import TextEmbedder
from src.featurizer import build_canonical_from_raw
from src.similarity import hybrid_similarity
from src.storage import load_artifacts
from src.config import config

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py 'your startup description'")
        sys.exit(1)

    description = sys.argv[1]
    canonicals, embeddings, index = load_artifacts(config.OUTPUT_DIR)
    embedder = TextEmbedder()

    query = build_canonical_from_raw("query_temp", description)
    query_emb = embedder.embed([query.canonical_description])

    sims, idxs = index.search(query_emb, k=min(15, len(canonicals)))
    candidates = [canonicals[i] for i in idxs[0]]

    ranked = hybrid_similarity(query, candidates, sims[0])

    for r in ranked[:10]:
        s = r["startup"]
        print(f"{s['id']} | {s['canonical_description']}")
        print(f"  → hybrid={r['hybrid_score']:.3f} (text={r['text_sim']:.3f}, bm={r['bm_sim']:.2f}, sector={r['sector_sim']:.2f})")
        print()

if __name__ == "__main__":
    main()