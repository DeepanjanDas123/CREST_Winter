# Structured startup representation with validation.
from pydantic import BaseModel
from typing import Optional, Dict, Any

class StartupCanonical(BaseModel):
    id: str
    raw_description: str
    canonical_description: str
    sector: Optional[str] = None
    subsector: Optional[str] = None
    business_model: Optional[str] = None
    b2x: Optional[str] = None
    target_customer: Optional[str] = None
    geography: Optional[str] = None
    stage: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True  # Immutable for safe indexing