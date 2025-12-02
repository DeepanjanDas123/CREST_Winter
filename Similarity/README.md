# Startup Similarity Engine

A fully offline hybrid startup similarity engine using **local LLMs (Ollama + Llama 3)** and **sentence embeddings** to find relevant matches — no API keys, no internet required.

## ✨ Features
- Runs 100% offline with Ollama
- Extracts structured attributes (sector, business model, B2B/B2C, etc.)
- Hybrid scoring: semantic + categorical similarity
- FAISS-powered fast retrieval
- CLI interface for ingestion and querying

## 🚀 Quick Start

1. **Prerequisites**  
   ```bash
   ollama pull llama3
   ```

2. **Install**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data (```data/startups_raw.csv```)**
   ```csv
   id,description
   1,"AI for hospital records"
   2,"EV charging for fleets"
   ```

4. **Build index**
    ```bash
    python -m scripts.build_index
    ```

5. **Query**
    ```bash
    python -m scripts.query "AI tools for doctors"
    ```

## 📁 Structure
- ```src/``` – Core logic (LLM, embeddings, indexing)
- ```scripts/``` – build_index.py, query.py
- ```data/``` – Input CSV
- ```output/``` – Cached artifacts (not committed)

## Privacy

All processing happens locally, as of now. You can change that and use open source API keys as you wish.