"""
task_b/main.py  —  Recommendation Agent API
POST /recommend
POST /recommend/multiturn
GET  /health
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from task_b.recommender import RecommendationAgent

logging.basicConfig(level="INFO", format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_agent: Optional[RecommendationAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    log.info("Starting Task B — loading data ...")
    _agent = RecommendationAgent()
    log.info("Task B ready")
    yield


app = FastAPI(
    title="BCT Hackathon — Task B: Recommendation Agent",
    description="Agentic LLM recommender that reasons before recommending. Handles cold-start, cross-domain, and multi-turn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role   : str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class RecommendRequest(BaseModel):
    user_id             : str                    = Field(...,  description="Yelp user ID")
    context             : str                    = Field("",   description="Current request or mood e.g. 'Looking for a good suya spot tonight'")
    conversation_history: list[ConversationTurn] = Field([], description="Prior conversation turns for multi-turn support")
    top_k               : int                    = Field(10,  description="Number of recommendations to return (max 10)", ge=1, le=10)

    model_config = {"json_schema_extra": {"examples": [{
        "user_id": "yelp_user_abc123",
        "context": "I want somewhere good for a Friday night out with friends",
        "conversation_history": [],
        "top_k": 5
    }]}}


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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "task": "B — Recommendation Agent", "ready": _agent is not None}


@app.post("/recommend", response_model=RecommendResponse, tags=["Task B"],
          summary="Get personalised recommendations for a user")
async def recommend(req: RecommendRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not initialised yet")
    try:
        result = _agent.recommend(
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


@app.post("/recommend/multiturn", response_model=RecommendResponse, tags=["Task B"],
          summary="Multi-turn recommendation — continues an existing conversation")
async def recommend_multiturn(req: RecommendRequest):
    """
    Same as /recommend but explicitly for multi-turn flows.
    Pass the full conversation_history to maintain context across turns.
    """
    return await recommend(req)