"""
task_a/main.py  —  Review Simulator API
POST /simulate-review
POST /simulate-review/batch
GET  /health
GET  /metadata        ← NEW: dropdown data for the UI
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from task_a.simulator import ReviewSimulator

logging.basicConfig(level="INFO", format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

_simulator: Optional[ReviewSimulator] = None
_metadata : Optional[dict] = None


def _load_metadata() -> dict:
    path = PROCESSED_DIR / "ui_metadata.json"
    if not path.exists():
        log.warning(f"ui_metadata.json not found at {path}")
        return {"top_users": [], "top_cities": [], "top_states": [],
                "top_categories": [], "sample_businesses": []}
    with open(path) as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _simulator, _metadata
    log.info("Starting Task A - loading data ...")
    _simulator = ReviewSimulator()
    _metadata  = _load_metadata()
    log.info("Task A ready")
    yield


app = FastAPI(
    title="BCT Hackathon — Task A: Review Simulator",
    description="LLM agent that simulates user reviews.",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ProductDetails(BaseModel):
    name        : str   = Field("",  description="Business name")
    categories  : str   = Field("",  description="Comma-separated categories")
    city        : str   = Field("",  description="City")
    state       : str   = Field("",  description="State/region")
    stars       : float = Field(3.5, description="Business average stars")
    review_count: int   = Field(0,   description="Number of existing reviews")


class SimulateRequest(BaseModel):
    user_id        : str            = Field(..., description="Yelp user ID")
    business_id    : str            = Field(..., description="Yelp business ID")
    product_details: ProductDetails = Field(default_factory=ProductDetails)


class SimulateResponse(BaseModel):
    predicted_stars   : float
    review_text       : str
    persona_summary   : dict
    rating_confidence : float


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "task": "A - Review Simulator", "ready": _simulator is not None}


@app.get("/metadata", tags=["System"], summary="Dropdown options for the UI")
def metadata():
    if _metadata is None:
        raise HTTPException(503, "Metadata not loaded yet")
    return _metadata


@app.post("/simulate-review", response_model=SimulateResponse, tags=["Task A"])
def simulate_review(req: SimulateRequest):
    if _simulator is None:
        raise HTTPException(503, "Simulator not initialised yet")
    try:
        result = _simulator.simulate(
            user_id        = req.user_id,
            business_id    = req.business_id,
            product_details= req.product_details.model_dump(),
        )
        return SimulateResponse(**result)
    except Exception as e:
        log.error(f"Simulation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/simulate-review/batch", tags=["Task A"])
def simulate_batch(requests: list[SimulateRequest]):
    if _simulator is None:
        raise HTTPException(503, "Simulator not initialised yet")
    results = []
    for req in requests:
        try:
            r = _simulator.simulate(
                user_id=req.user_id, business_id=req.business_id,
                product_details=req.product_details.model_dump(),
            )
            results.append({"status": "ok", **r})
        except Exception as e:
            results.append({"status": "error", "detail": str(e)})
    return results
