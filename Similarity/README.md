# Startup Similarity Engine

Finds similar startups using hybrid (text + structured) similarity.

## Setup

1. `cp .env.example .env` and fill in your OpenAI API key.
2. Place your startup data in `data/startups_raw.csv` with columns: `id`, `description`.
3. `pip install -r requirements.txt`
4. Build index: `python scripts/build_index.py`
5. Query: `python scripts/query.py "We build EV chargers for fleets in India"`

## Data Format

`data/startups_raw.csv` must have:
- `id`: unique string/integer
- `description`: free-text startup description