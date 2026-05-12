"""
task_a/main.py  —  Review Simulator (Task A)
Day 2+ will fill in the simulator logic.
This placeholder lets you confirm the container starts correctly.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BCT Hackathon — Task A: Review Simulator", version="0.1.0")


class SimulateRequest(BaseModel):
    user_id: str
    business_id: str
    # Optional context the judge may pass
    product_details: dict = {}


class SimulateResponse(BaseModel):
    predicted_stars: float
    review_text: str
    persona_summary: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "task": "A"}


@app.post("/simulate-review", response_model=SimulateResponse)
async def simulate_review(req: SimulateRequest):
    # TODO (Day 2): wire in PersonaBuilder + ReviewSimulator
    return SimulateResponse(
        predicted_stars=4.0,
        review_text="[Simulator not yet implemented — scaffold only]",
    )
