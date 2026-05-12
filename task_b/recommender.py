"""
task_b/recommender.py  (v2)
----------------------------
Fixes in this version:
  - complete_json() for native Gemini JSON enforcement (no regex fallback needed)
  - Intent prompt tightened + max_tokens raised to 80 (was 50, causing truncation)
  - Reranker uses few-shot JSON example in prompt for extra enforcement
"""

import logging
import os
import json
from typing import Optional

from dotenv import load_dotenv

from shared.persona.builder import PersonaBuilder
from shared.vectorstore.store import VectorStore
from shared.llm.client import llm
from shared.nigerian.adapter import apply as nigerian_apply, build_system_prompt

load_dotenv()
log = logging.getLogger(__name__)

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
TOP_K_RETRIEVE = 20
TOP_K_RETURN   = 10


class RecommendationAgent:

    def __init__(self):
        log.info("Initialising RecommendationAgent ...")
        self.persona_builder = PersonaBuilder()
        self.vector_store    = VectorStore()
        self._llm            = None
        log.info("RecommendationAgent ready")

    def recommend(
        self,
        user_id             : str,
        context             : str = "",
        conversation_history: list = None,
        top_k               : int  = TOP_K_RETURN,
    ) -> dict:
        conversation_history = conversation_history or []
        top_k = min(top_k, TOP_K_RETURN)

        persona = self.persona_builder.build(user_id)
        persona = nigerian_apply(persona, user_context=context)

        search_intent = self._reason_intent(persona, context, conversation_history)
        log.info(f"Search intent [{user_id}]: {search_intent}")

        candidates = self._retrieve(persona, search_intent, context)
        ranked     = self._rerank(persona, candidates, context, search_intent, top_k)

        return {
            "recommendations" : ranked,
            "cold_start"      : persona.get("is_cold", False),
            "persona_summary" : _persona_summary(persona),
            "search_intent"   : search_intent,
        }

    # ── Intent Reasoning ─────────────────────────────────────

    def _reason_intent(self, persona, context, history) -> str:
        top_cats   = persona.get("top_categories", [])
        mean_stars = persona.get("mean_stars", 3.5)
        is_cold    = persona.get("is_cold", False)

        history_str = ""
        if history:
            turns = history[-4:]
            history_str = "\n".join(
                f"{t.get('role','user').upper()}: {t.get('content','')}"
                for t in turns
            )

        system = (
            "You are the intent reasoning module of a recommendation agent.\n"
            "Output a single search query string of 5-12 words that captures what the user wants.\n"
            "Output ONLY the query. No quotes, no explanation, no punctuation at the end."
        )

        user = (
            f"USER REQUEST: {context or 'General recommendation'}\n"
            f"USER FAVOURITE CATEGORIES: {', '.join(top_cats) if top_cats else 'unknown'}\n"
            f"USER TYPICAL RATING: {mean_stars:.1f} stars\n"
            f"NEW USER (no history): {is_cold}\n"
            + (f"\nCONVERSATION:\n{history_str}\n" if history_str else "")
            + "\nSearch query:"
        )

        try:
            intent = self._get_llm().complete(system=system, user=user, max_tokens=80)
            # Clean any stray quotes/punctuation the model might add
            intent = intent.strip().strip('"').strip("'").rstrip(".")
            # Safety: if somehow still too long, truncate at word boundary
            words = intent.split()
            return " ".join(words[:15]) if len(words) > 15 else intent
        except Exception as e:
            log.warning(f"Intent reasoning failed: {e}")
            return context or ", ".join(top_cats[:3]) or "restaurants"

    # ── Retrieval ─────────────────────────────────────────────

    def _retrieve(self, persona, search_intent, context) -> list:
        is_cold  = persona.get("is_cold", False)
        top_cats = persona.get("top_categories", [])

        query = search_intent if (is_cold or not top_cats) \
                else f"{search_intent} {' '.join(top_cats[:2])}"

        candidates = self.vector_store.query_businesses(query=query, n=TOP_K_RETRIEVE)

        if len(candidates) < 5 and top_cats:
            extra = self.vector_store.query_by_category(top_cats, n=TOP_K_RETRIEVE)
            seen  = {c["business_id"] for c in candidates}
            for biz in extra:
                if biz["business_id"] not in seen:
                    candidates.append(biz)

        return candidates[:TOP_K_RETRIEVE]

    # ── LLM Reranker ──────────────────────────────────────────

    def _rerank(self, persona, candidates, context, search_intent, top_k) -> list:
        if not candidates:
            return []

        candidate_list = "\n".join(
            f"{i+1}. id={c['business_id']} | {c.get('name','?')} | "
            f"{c.get('categories','')[:60]} | {c.get('city','')} | "
            f"{c.get('stars','?')}★"
            for i, c in enumerate(candidates)
        )

        system = build_system_prompt(
            _RERANK_SYSTEM.format(
                top_k        = top_k,
                mean_stars   = persona.get("mean_stars", 3.5),
                bias         = persona.get("rating_bias", "neutral"),
                top_cats     = ", ".join(persona.get("top_categories", ["general"])),
                is_cold      = persona.get("is_cold", False),
                search_intent= search_intent,
                context      = context or "None",
            ),
            persona,
            task="recommend",
        )

        # Few-shot JSON example baked into the user prompt for extra enforcement
        user = f"""Select and rank the best {top_k} from these {len(candidates)} businesses.

CANDIDATES:
{candidate_list}

Return ONLY this JSON structure (example with 2 items shown):
{{
  "ranked": [
    {{"business_id": "abc123xyz", "score": 0.95, "reason": "Great seafood spot matching user taste for upscale dining"}},
    {{"business_id": "def456uvw", "score": 0.82, "reason": "Popular brunch place in the right category"}}
  ]
}}

Now return the real JSON for all {top_k} picks:"""

        try:
            # We use max_tokens=2000 to prevent long JSON responses from being truncated
            raw    = self._get_llm().complete_json(system=system, user=user, max_tokens=max(LLM_MAX_TOKENS, 2000))
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt to recover truncated JSON by appending closing brackets
                raw = raw.strip()
                if not raw.endswith("}"):
                    if not raw.endswith("]"):
                        raw += '"}]}'
                    else:
                        raw += '}'
                parsed = json.loads(raw)

            ranked_map = {item["business_id"]: item for item in parsed.get("ranked", [])}
        except Exception as e:
            log.warning(f"LLM reranking failed ({e}) — falling back to distance ranking")
            return [
                {
                    "business_id": c["business_id"],
                    "name"       : c.get("name", ""),
                    "categories" : c.get("categories", ""),
                    "city"       : c.get("city", ""),
                    "stars"      : c.get("stars", ""),
                    "score"      : round(1 - float(c.get("distance", 0.5)), 3),
                    "reason"     : "Matched based on semantic similarity.",
                }
                for c in candidates[:top_k]
            ]

        output = []
        for c in candidates:
            bid = c["business_id"]
            if bid in ranked_map:
                item = ranked_map[bid]
                output.append({
                    "business_id": bid,
                    "name"       : c.get("name", ""),
                    "categories" : c.get("categories", ""),
                    "city"       : c.get("city", ""),
                    "stars"      : c.get("stars", ""),
                    "score"      : float(item.get("score", 0.5)),
                    "reason"     : item.get("reason", ""),
                })

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    def _get_llm(self):
        if self._llm is None:
            self._llm = llm()
        return self._llm


# ── Prompt template ───────────────────────────────────────────────────────────

_RERANK_SYSTEM = """You are a personalised recommendation agent.

USER PROFILE:
- Typical rating   : {mean_stars:.1f} stars ({bias} rater)
- Favourite types  : {top_cats}
- New user         : {is_cold}

WHAT THEY WANT: {search_intent}
CONTEXT       : {context}

Pick the best {top_k} businesses. Score each 0.0–1.0.
Write one specific reason per item explaining why it fits THIS user's profile."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _persona_summary(persona: dict) -> dict:
    return {
        "is_cold"     : persona.get("is_cold",        False),
        "review_count": persona.get("review_count",   0),
        "rating_bias" : persona.get("rating_bias",    "neutral"),
        "style"       : persona.get("style_label",    "medium"),
        "nigerian"    : persona.get("nigerian_mode",  False),
        "top_cats"    : persona.get("top_categories", []),
    }
