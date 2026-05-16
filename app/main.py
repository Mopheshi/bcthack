"""
app/main.py  —  Consolidated PersonaRAG API (Task A + Task B in one service)
-----------------------------------------------------------------------------
WHY ONE SERVICE
  Running Task A and Task B as two separate containers each loading their
  own copy of the data doubled the memory footprint and made them
  impossible to run together on a 16 GB laptop. They share the SAME
  PersonaBuilder and (Task B) the vector store, so one process serving
  both is strictly better:
    - one persona store in memory, not two
    - one cold start, not two
    - one URL to deploy, one health check
    - the UI talks to a single origin (no cross-port confusion)

ENDPOINTS
  GET  /health             readiness probe (used by Cloud Run startup probe)
  GET  /metadata           dropdown data for the UI
  POST /api/simulate       Task A — review simulation
  POST /api/recommend      Task B — recommendation
  POST /api/simulate/batch Task A — batch simulation

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from task_a.simulator import ReviewSimulator
from task_b.recommender import RecommendationAgent

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

# Singletons populated by the background init thread after uvicorn starts.
_simulator: Optional[ReviewSimulator]    = None
_agent:     Optional[RecommendationAgent] = None
_metadata:  Optional[dict]                = None
_init_error: Optional[str]               = None


def _load_metadata() -> dict:
    path = PROCESSED_DIR / "ui_metadata.json"
    if not path.exists():
        log.warning(f"ui_metadata.json not found at {path}")
        return {"top_users": [], "top_cities": [], "top_states": [],
                "top_categories": [], "sample_businesses": []}
    with open(path) as f:
        return json.load(f)


def _background_init():
    """Heavy startup work runs in a thread so uvicorn binds port 8080 immediately.
    Cloud Run's HTTP startup probe can then reach /healthz/startup right away."""
    global _simulator, _agent, _metadata, _init_error
    try:
        log.info("=== PersonaRAG init starting (background) ===")
        sim   = ReviewSimulator()
        agent = RecommendationAgent(persona_builder=sim.persona_builder)
        meta  = _load_metadata()
        # Assign atomically so /health never sees a partial state
        _simulator = sim
        _agent     = agent
        _metadata  = meta
        log.info("=== PersonaRAG ready ===")
    except Exception as exc:
        _init_error = str(exc)
        log.exception("Background init failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off heavy init in a daemon thread and yield immediately so uvicorn
    # starts serving HTTP before data is loaded.  /healthz/startup returns 503
    # until _simulator and _agent are set, then switches to 200.
    threading.Thread(target=_background_init, daemon=True).start()
    yield
    log.info("=== PersonaRAG shutting down ===")


app = FastAPI(
    title="PersonaRAG — Agentic LLM Framework",
    description="Task A (review simulation) + Task B (recommendation) in one service.",
    version="6.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ──────────────────────────────────────────────────────────────────

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


class ConversationTurn(BaseModel):
    role   : str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class RecommendRequest(BaseModel):
    user_id             : str                    = Field(..., description="Yelp user ID")
    context             : str                    = Field("",  description="Current request or mood")
    conversation_history: list[ConversationTurn]  = Field(default_factory=list)
    top_k               : int                    = Field(10, ge=1, le=10)


class RecommendItem(BaseModel):
    business_id: str
    name       : str
    categories : str
    city       : str
    stars      : str
    score      : float
    reason     : str


class RecommendResponse(BaseModel):
    recommendations : list[RecommendItem]
    cold_start      : bool
    persona_summary : dict
    search_intent   : str


# ── System endpoints ─────────────────────────────────────────────────────────

@app.get("/healthz/startup", tags=["System"], include_in_schema=False)
def startup_probe():
    """Cloud Run startup probe target. Returns 503 while background init runs,
    200 once both singletons are set. Keeps the HTTP probe from timing out."""
    if _init_error:
        raise HTTPException(500, f"Init failed: {_init_error}")
    if _simulator is None or _agent is None:
        raise HTTPException(503, "Initialising — not ready yet")
    return {"status": "ready"}


@app.get("/health", tags=["System"])
def health():
    ready = _simulator is not None and _agent is not None
    return {
        "status": "ok" if ready else "starting",
        "ready" : ready,
        "warm_users": _simulator.persona_builder.warm_user_count if _simulator else 0,
    }


@app.get("/metadata", tags=["System"], summary="Dropdown data for the UI")
def metadata():
    if _metadata is None:
        raise HTTPException(503, "Metadata not loaded yet")
    return _metadata


# ── Task A ───────────────────────────────────────────────────────────────────

@app.post("/api/simulate", response_model=SimulateResponse, tags=["Task A"])
def simulate(req: SimulateRequest):
    if _simulator is None:
        raise HTTPException(503, "Simulator not initialised yet")
    try:
        result = _simulator.simulate(
            user_id         = req.user_id,
            business_id     = req.business_id,
            product_details = req.product_details.model_dump(),
        )
        return SimulateResponse(**result)
    except Exception as e:
        log.error(f"Simulation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/simulate/batch", tags=["Task A"])
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


# ── Task B ───────────────────────────────────────────────────────────────────

@app.post("/api/recommend", response_model=RecommendResponse, tags=["Task B"])
async def recommend(req: RecommendRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not initialised yet")
    try:
        result = await _agent.recommend(
            user_id              = req.user_id,
            context              = req.context,
            conversation_history = [t.model_dump() for t in req.conversation_history],
            top_k                = req.top_k,
        )
        return RecommendResponse(**result)
    except Exception as e:
        log.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
