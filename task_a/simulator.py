"""
LATENCY FIXES over v3:

1. RAG context cap: 3 sample reviews × 150 chars max (was 5 × 300 chars).
   Cuts ~900 input tokens per warm-user call → ~3-5s faster Gemini response.

2. Trimmed system prompt verbosity. Same information, fewer tokens.

3. LLM_MAX_TOKENS lowered to 600 (was 8000). Reviews are 80-200 words,
   600 tokens is plenty and prevents Gemini from over-generating.
"""

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv

from shared.persona.builder import PersonaBuilder
from shared.llm.client import llm
from shared.nigerian.adapter import apply as nigerian_apply, build_system_prompt

load_dotenv()
log = logging.getLogger(__name__)

# Tightened for latency
MAX_RAG_REVIEWS    = int(os.getenv("MAX_RAG_REVIEWS",     "3"))
RAG_SNIPPET_LEN    = int(os.getenv("RAG_SNIPPET_LEN",     "150"))
LLM_MAX_TOKENS     = int(os.getenv("LLM_REVIEW_TOKENS",   "600"))


# ── Rating Predictor (unchanged from v3) ──────────────────────────────────────

class RatingPredictor:
    """
    Calibrated star rating predictor.

    Strategy (no ML model needed — pure statistics):
      predicted = clip(user_mean + business_deviation_from_global, 1, 5)
    """
    GLOBAL_MEAN = 3.63

    def predict(self, persona: dict, biz_avg_stars: float, biz_review_count: int = 0) -> float:
        user_mean = persona.get("mean_stars", self.GLOBAL_MEAN)
        bias      = persona.get("rating_bias", "neutral")
        biz_offset = biz_avg_stars - self.GLOBAL_MEAN
        bias_adj = {"generous": 0.3, "neutral": 0.0, "critical": -0.3}.get(bias, 0.0)

        confidence = min(1.0, biz_review_count / 50.0) if biz_review_count else 0.5
        adjusted_offset = biz_offset * confidence
        raw = user_mean + adjusted_offset + bias_adj
        clipped = max(1.0, min(5.0, raw))
        return round(clipped * 2) / 2


# ── Review Simulator ──────────────────────────────────────────────────────────

class ReviewSimulator:
    """
    Main Task A class. Instantiate once at app startup.

    Usage:
        sim = ReviewSimulator()
        result = sim.simulate(
            user_id="abc123",
            business_id="xyz456",
            product_details={"name": "Joe's Pizza", "categories": "Pizza, Italian", ...}
        )
    """

    def __init__(self):
        log.info("Initialising ReviewSimulator ...")
        self.persona_builder  = PersonaBuilder()
        self.rating_predictor = RatingPredictor()
        self._llm             = None
        log.info("ReviewSimulator ready")

    def simulate(
        self,
        user_id: str,
        business_id: str,
        product_details: Optional[dict] = None,
    ) -> dict:
        """
        Returns:
          {
            predicted_stars : float,
            review_text     : str,
            persona_summary : dict,
            rating_confidence: float,
          }
        """
        product_details = product_details or {}

        persona = self.persona_builder.build(user_id)

        # Apply Nigerian adapter (uses business metadata as additional signal)
        context_text = " ".join(filter(None, [
            product_details.get("name",       ""),
            product_details.get("categories", ""),
            product_details.get("city",       ""),
            product_details.get("state",      ""),
        ]))
        persona = nigerian_apply(persona, user_context=context_text)

        # Predict rating
        biz_avg    = float(product_details.get("stars", RatingPredictor.GLOBAL_MEAN))
        biz_cnt    = int(product_details.get("review_count", 0))
        pred_stars = self.rating_predictor.predict(persona, biz_avg, biz_cnt)

        # RAG context — capped at 3 short snippets
        rag_context = self._build_rag_context(persona, product_details)

        # Build prompts
        system_prompt = self._build_system_prompt(persona, product_details, pred_stars)
        system_prompt = build_system_prompt(system_prompt, persona, task="review")
        user_prompt   = self._build_user_prompt(product_details, pred_stars, rag_context)

        # Generate
        review_text = self._get_llm().complete(
            system=system_prompt, user=user_prompt,
            max_tokens=LLM_MAX_TOKENS,
        )

        # Clean any stray artefacts
        review_text = _clean_review(review_text)

        return {
            "predicted_stars"   : pred_stars,
            "review_text"       : review_text,
            "persona_summary"   : _persona_summary(persona),
            "rating_confidence" : _rating_confidence(persona),
        }

    # ── Prompt construction (tightened) ──────────────────────

    def _build_system_prompt(self, persona: dict, biz: dict, stars: float) -> str:
        style   = persona.get("style_label",  "medium")
        bias    = persona.get("rating_bias",  "neutral")
        is_cold = persona.get("is_cold", False)
        avg     = persona.get("mean_stars", 3.5)

        word_range = {
            "concise": "50-80 words",
            "medium" : "80-130 words",
            "verbose": "130-200 words",
        }.get(style, "80-130 words")

        history_note = (
            "First-time reviewer." if is_cold
            else f"Established {bias} {avg:.1f}-star rater."
        )

        return (
            f"Simulate a {stars}-star Yelp review.\n"
            f"USER: {history_note} Writes in {style} style ({word_range}).\n"
            f"BUSINESS: {biz.get('name','this business')} "
            f"({biz.get('categories','')}) in {biz.get('city','')}, {biz.get('state','')}. "
            f"Average rating {biz.get('stars','?')} stars.\n"
            f"RULES: Output ONLY the review. Match {word_range}. "
            f"Reflect {stars}-star sentiment. Sound human, not corporate."
        )

    def _build_rag_context(self, persona: dict, biz: dict) -> str:
        """Select up to MAX_RAG_REVIEWS sample reviews most relevant to this business."""
        samples = persona.get("sample_reviews", [])
        if not samples:
            return ""

        # Score sample reviews by category overlap with target business
        biz_cats = {
            c.strip().lower()
            for c in biz.get("categories", "").split(",")
            if c.strip()
        }

        scored = []
        for text in samples:
            t_lower = text.lower()
            overlap = sum(1 for cat in biz_cats if cat in t_lower)
            scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[:MAX_RAG_REVIEWS]]
        if not top:
            return ""

        formatted = " | ".join(f'"{t[:RAG_SNIPPET_LEN]}"' for t in top)
        return f"\nUSER'S STYLE EXAMPLES: {formatted}"

    def _build_user_prompt(self, biz: dict, stars: float, rag_context: str) -> str:
        return (
            f"Write a {stars}-star review for {biz.get('name','this business')}.{rag_context}"
        )

    def _get_llm(self):
        if self._llm is None:
            self._llm = llm()
        return self._llm


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_review(text: str) -> str:
    """Remove common LLM artefacts from generated review text."""
    # Remove leading labels like "Review:", "User review:", etc.
    text = re.sub(r"^(review|user review|yelp review)\s*:\s*", "", text, flags=re.IGNORECASE)
    # Remove star prefix like "★★★★☆" or "4/5"
    text = re.sub(r"^[★☆✩\d/\s]+\s*[-–—]?\s*", "", text)
    return text.strip()


def _persona_summary(persona: dict) -> dict:
    return {
        "is_cold"     : persona.get("is_cold",         False),
        "review_count": persona.get("review_count",    0),
        "rating_bias" : persona.get("rating_bias",     "neutral"),
        "style"       : persona.get("style_label",     "medium"),
        "nigerian"    : persona.get("nigerian_mode",   False),
        "top_cats"    : persona.get("top_categories",  []),
    }


def _rating_confidence(persona: dict) -> float:
    """
    Returns a 0–1 confidence score for the rating prediction.
    Higher review count = more confidence.
    """
    rc = persona.get("review_count", 0)
    return round(min(1.0, rc / 50.0), 2)
