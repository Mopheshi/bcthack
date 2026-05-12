"""
task_a/simulator.py
--------------------
ReviewSimulator — core logic for Task A.

Pipeline per request:
  1. PersonaBuilder  → extract user behavioural fingerprint
  2. RatingPredictor → calibrated star prediction (separate from LLM)
  3. RAG context     → select most relevant sample reviews to guide tone
  4. NigerianAdapter → inject cultural layer if applicable
  5. LLM             → generate review text conditioned on all of the above

Rating prediction is deliberately decoupled from text generation.
This ensures RMSE is optimised independently of text quality (BERTScore/ROUGE).
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

MAX_SAMPLE_REVIEWS = int(os.getenv("TOP_K_REVIEWS", "20"))
MAX_RAG_REVIEWS    = 5     # how many sample reviews to include in LLM context
# Fix for Gemini API truncating responses when max_output_tokens is set too low:
LLM_MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "8000"))


# ── Rating Predictor ──────────────────────────────────────────────────────────

class RatingPredictor:
    """
    Calibrated star rating predictor.

    Strategy (no ML model needed — pure statistics):
      predicted = clip(user_mean + business_deviation_from_global, 1, 5)

    Where business_deviation = biz_avg_stars - global_avg_stars.

    This beats naive LLM star prediction on RMSE because:
      - It anchors to the user's known rating behaviour
      - It adjusts for business quality signal
      - It never outputs 3.7 stars — rounds to nearest 0.5 (Yelp-style)
    """

    GLOBAL_MEAN = 3.63   # from EDA

    def predict(
        self,
        persona: dict,
        biz_avg_stars: float,
        biz_review_count: int = 0,
    ) -> float:
        """
        Returns predicted star rating as float, rounded to nearest 0.5.
        """
        user_mean = persona.get("mean_stars", self.GLOBAL_MEAN)
        user_std  = persona.get("std_stars",  1.2)
        bias      = persona.get("rating_bias", "neutral")

        # Business quality offset relative to global mean
        biz_offset = biz_avg_stars - self.GLOBAL_MEAN

        # Bias adjustment — generous users rate higher, critical lower
        bias_adj = {"generous": 0.3, "neutral": 0.0, "critical": -0.3}.get(bias, 0.0)

        # Low review count = uncertain business quality, regress to mean
        confidence = min(1.0, biz_review_count / 50.0) if biz_review_count else 0.5
        adjusted_offset = biz_offset * confidence

        raw = user_mean + adjusted_offset + bias_adj

        # Clip to [1, 5] and round to nearest 0.5 (Yelp uses 0.5 increments)
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
        self._llm             = None   # lazy init
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

        # ── 1. Build persona ──────────────────────────────────
        persona = self.persona_builder.build(user_id)

        # ── 2. Apply Nigerian adapter ─────────────────────────
        context_text = " ".join(filter(None, [
            product_details.get("name",       ""),
            product_details.get("categories", ""),
            product_details.get("city",       ""),
            product_details.get("state",      ""),
        ]))
        persona = nigerian_apply(persona, user_context=context_text)

        # ── 3. Predict rating ─────────────────────────────────
        biz_avg    = float(product_details.get("stars", RatingPredictor.GLOBAL_MEAN))
        biz_cnt    = int(product_details.get("review_count", 0))
        pred_stars = self.rating_predictor.predict(persona, biz_avg, biz_cnt)

        # ── 4. Build RAG context from sample reviews ──────────
        rag_context = self._build_rag_context(persona, product_details)

        # ── 5. Generate review text ───────────────────────────
        system_prompt = self._build_system_prompt(persona, product_details, pred_stars)
        system_prompt = build_system_prompt(system_prompt, persona, task="review")

        user_prompt   = self._build_user_prompt(persona, product_details, pred_stars, rag_context)

        review_text   = self._get_llm().complete(
            system    = system_prompt,
            user      = user_prompt,
            max_tokens= LLM_MAX_TOKENS,
        )

        # Clean any stray artefacts
        review_text = _clean_review(review_text)

        return {
            "predicted_stars"   : pred_stars,
            "review_text"       : review_text,
            "persona_summary"   : _persona_summary(persona),
            "rating_confidence" : _rating_confidence(persona),
        }

    # ── Prompt construction ───────────────────────────────────

    def _build_system_prompt(self, persona: dict, biz: dict, stars: float) -> str:
        style   = persona.get("style_label",  "medium")
        bias    = persona.get("rating_bias",  "neutral")
        avg_wc  = int(persona.get("mean_word_count", 104))
        is_cold = persona.get("is_cold", False)

        word_guide = {
            "concise": "50–80 words",
            "medium" : "80–130 words",
            "verbose": "130–200 words",
        }.get(style, "80–130 words")

        return f"""You are simulating a realistic Yelp review written by a specific user.

USER PROFILE:
- Rating tendency  : {bias} (they typically give {persona.get('mean_stars', 3.5):.1f} stars)
- Writing style    : {style} ({word_guide})
- Has review history: {"No — write as a typical first-time reviewer" if is_cold else "Yes — match their established style"}

BUSINESS BEING REVIEWED:
- Name       : {biz.get('name', 'this business')}
- Categories : {biz.get('categories', 'Restaurant')}
- Location   : {biz.get('city', '')}, {biz.get('state', '')}
- Avg rating : {biz.get('stars', 'unknown')} stars

THE USER IS GIVING THIS BUSINESS: {stars} stars

RULES:
- Write ONLY the review text. No star rating prefix, no "Review:" label.
- Match the word count range: {word_guide}.
- Reflect the {stars}-star sentiment authentically throughout.
- Sound like a real person, not a marketing copy.
- Vary sentence structure. Avoid lists."""

    def _build_rag_context(self, persona: dict, biz: dict) -> str:
        """Select up to MAX_RAG_REVIEWS sample reviews most relevant to this business."""
        samples = persona.get("sample_reviews", [])
        if not samples:
            return ""

        biz_cats = set(
            c.strip().lower()
            for c in biz.get("categories", "").split(",")
            if c.strip()
        )

        # Score each sample review by category overlap
        scored = []
        for text in samples[:MAX_SAMPLE_REVIEWS]:
            text_lower = text.lower()
            overlap    = sum(1 for cat in biz_cats if cat in text_lower)
            scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[:MAX_RAG_REVIEWS]]

        if not top:
            return ""

        formatted = "\n---\n".join(f'"{t[:300]}"' for t in top)
        return f"\nPAST REVIEWS BY THIS USER (for style reference only):\n{formatted}"

    def _build_user_prompt(
        self,
        persona: dict,
        biz: dict,
        stars: float,
        rag_context: str,
    ) -> str:
        prompt = f"Write a {stars}-star Yelp review for {biz.get('name', 'this business')}."

        if biz.get("categories"):
            prompt += f" It is a {biz.get('categories')} establishment."

        if rag_context:
            prompt += f"\n{rag_context}"

        prompt += "\n\nWrite the review now:"
        return prompt

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
