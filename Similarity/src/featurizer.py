# Convert raw startup descriptions into structured canonical forms.
import pandas as pd
import logging
from typing import List
from .models import StartupCanonical
from .llm_utils import call_llm

logger = logging.getLogger(__name__)

def build_canonical_from_raw(startup_id: str, raw_description: str) -> StartupCanonical:
    """Canonicalize a single startup using LLM."""
    desc = raw_description.strip() if raw_description else ""
    if not desc:
        desc = "No description provided."
        logger.warning(f"Empty description for startup {startup_id}")

    try:
        data = call_llm(desc)
    except Exception as e:
        logger.error(f"LLM failed for {startup_id}: {e}. Using fallback.")
        data = {
            "canonical_description": desc,
            "sector": None,
            "subsector": None,
            "business_model": None,
            "b2x": None,
            "target_customer": None,
            "geography": None,
            "stage": None,
        }

    # Separate known fields from extras
    known_fields = set(StartupCanonical.model_fields.keys())
    extra = {k: v for k, v in data.items() if k not in known_fields}
    filtered = {k: v for k, v in data.items() if k in known_fields}

    return StartupCanonical(
        id=str(startup_id),
        raw_description=raw_description,
        extra=extra,
        **filtered
    )

def canonicalize_dataset(df: pd.DataFrame) -> List[StartupCanonical]:
    """Process entire DataFrame (must have 'id' and 'description')."""
    if not {"id", "description"}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'id' and 'description' columns")
    return [build_canonical_from_raw(row["id"], str(row["description"])) for _, row in df.iterrows()]