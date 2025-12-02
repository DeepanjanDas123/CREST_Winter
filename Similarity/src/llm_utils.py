# src/llm_utils.py
import json
import re
import logging
from typing import Dict, Any
import ollama  # official Ollama Python client

logger = logging.getLogger(__name__)

# Use your local model — change if you use 'llama3:8b' or 'llama3.1'
OLLAMA_MODEL = "llama3"

CANONICAL_PROMPT_TEMPLATE = """\
You are an expert startup analyst. Given a startup description, extract structured fields in JSON.

Output ONLY a valid JSON object with these keys:
- canonical_description: one concise sentence in the format: "This startup [does X] for [Y] using [Z] in [sector]."
- sector: e.g., "Mobility", "FinTech", "HealthTech"
- subsector: more specific, e.g., "EV charging"
- business_model: one of ["manufacturing", "marketplace", "SaaS", "aggregator", "service", "other"]
- b2x: one of ["B2B", "B2C", "B2B2C", "B2G", "unknown"]
- target_customer: e.g., "logistics companies"
- geography: e.g., "India", "global", "unknown"
- stage: one of ["idea", "MVP", "pre-seed", "seed", "Series A+", "unknown"]

Description:
\"\"\"{description}\"\"\"

Respond with ONLY the JSON object. No other text.
"""

def _clean_json_response(text: str) -> str:
    """Remove markdown fences and extra text before/after JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Extract first valid JSON object if extra text exists
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    return text.strip()

def call_llm(description: str) -> Dict[str, Any]:
    """Call local Llama 3 via Ollama to canonicalize startup description."""
    prompt = CANONICAL_PROMPT_TEMPLATE.format(description=description)
    
    try:
        # Call Ollama (runs locally!)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 256}  # deterministic, limit tokens
        )
        content = response["message"]["content"]
        clean_content = _clean_json_response(content)
        return json.loads(clean_content)
    
    except (json.JSONDecodeError, KeyError, ollama.ResponseError) as e:
        logger.error(f"Ollama parsing error: {e} | Raw output: {content if 'content' in locals() else 'N/A'}")
        raise RuntimeError(f"Failed to parse Ollama response: {e}")