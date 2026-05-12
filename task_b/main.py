"""
task_b/main.py  —  Recommendation Agent (Task B)
Day 3+ will fill in the recommender logic.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="BCT Hackathon — Task B: Recommendation Agent", version="0.1.0")


class RecommendRequest(BaseModel):
    user_id: str
    context: str = ""          # Optional: current mood, location, etc.
    conversation_history: List[dict] = []   # Multi-turn support
    top_k: int = 10


class RecommendItem(BaseModel):
    business_id: str
    name: str
    score: float
    reason: str


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
    cold_start: bool = False
    persona_summary: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "task": "B"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    # TODO (Day 3): wire in PersonaBuilder + RecommendationAgent
    return RecommendResponse(
        recommendations=[],
        cold_start=False,
    )
