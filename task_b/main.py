"""
task_b/main.py  (v3 — async)
-----------------------------
Recommendation Agent API. Now uses the async recommender for parallel
intent reasoning + retrieval, giving ~30% speed-up on warm-user requests.

POST /recommend
POST /recommend/multiturn
GET  /health
GET  /metadata
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

from task_b.recommender import RecommendationAgent

logging.basicConfig(level="INFO", format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

_agent   : Optional[RecommendationAgent] = None
_metadata: Optional[dict] = None


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
    global _agent, _metadata
    log.info("Starting Task B - loading data ...")
    _agent    = RecommendationAgent()
    _metadata = _load_metadata()
    log.info("Task B ready")
    yield


app = FastAPI(
    title="BCT Hackathon — Task B: Recommendation Agent",
    description="Async agentic LLM recommender. Handles cold-start, cross-domain, multi-turn.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ConversationTurn(BaseModel):
    role   : str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class RecommendRequest(BaseModel):
    user_id             : str                    = Field(..., description="Yelp user ID")
    context             : str                    = Field("",  description="Current request or mood")
    conversation_history: list[ConversationTurn] = Field([],  description="Prior conversation turns")
    top_k               : int                    = Field(10,  description="Number of recs", ge=1, le=10)


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


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "task": "B - Recommendation Agent", "ready": _agent is not None}


@app.get("/metadata", tags=["System"], summary="Dropdown options for the UI")
def metadata():
    if _metadata is None:
        raise HTTPException(503, "Metadata not loaded yet")
    return _metadata


@app.post("/recommend", response_model=RecommendResponse, tags=["Task B"])
async def recommend(req: RecommendRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not initialised yet")
    try:
        # The recommender is now async-native — directly await it
        result = await _agent.recommend(
            user_id             = req.user_id,
            context             = req.context,
            conversation_history= [t.model_dump() for t in req.conversation_history],
            top_k               = req.top_k,
        )
        return RecommendResponse(
            recommendations = result["recommendations"],
            cold_start      = result["cold_start"],
            persona_summary = result["persona_summary"],
            search_intent   = result["search_intent"],
        )
    except Exception as e:
        log.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/recommend/multiturn", response_model=RecommendResponse, tags=["Task B"])
async def recommend_multiturn(req: RecommendRequest):
    return await recommend(req)
