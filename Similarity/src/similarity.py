# src/similarity.py
import numpy as np
from typing import List, Dict, Any
from .models import StartupCanonical
from .config import config

def simple_cat_sim(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 0.0
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return 1.0
    if ("marketplace" in a and "marketplace" in b):
        return 0.8
    if ("manufactur" in a and "manufactur" in b):
        return 0.8
    if ("saas" in a and "saas" in b):
        return 0.9
    return 0.0

def hybrid_similarity(
    query: StartupCanonical,
    candidates: List[StartupCanonical],
    text_sims: np.ndarray
) -> List[Dict[str, Any]]:
    if len(candidates) != len(text_sims):
        raise ValueError("Mismatch between candidates and similarities")

    results = []
    for idx, cand in enumerate(candidates):
        bm_sim = simple_cat_sim(query.business_model, cand.business_model)
        sector_sim = simple_cat_sim(query.sector, cand.sector)

        score = (
            config.ALPHA_TEXT * float(text_sims[idx]) +
            config.BETA_BUSINESS_MODEL * bm_sim +
            config.GAMMA_SECTOR * sector_sim
        )

        results.append({
            "startup": cand.model_dump(),  # serializable dict
            "text_sim": float(text_sims[idx]),
            "bm_sim": bm_sim,
            "sector_sim": sector_sim,
            "hybrid_score": score,
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results