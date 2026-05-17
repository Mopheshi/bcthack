"""
Task B — recommendation agent.

Accepts an optional persona_builder so the consolidated app can share the
same PersonaBuilder instance as Task A, keeping the persona store in memory
once. Omitting it builds its own (backward-compatible for standalone use).

Persona lookups are O(1) dict reads against the precomputed store; the only
real latency is the two LLM calls (intent + rerank), which overlap with retrieval.
"""

import asyncio
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from shared.llm.client import llm
from shared.nigerian.adapter import apply as nigerian_apply, build_system_prompt
from shared.persona.builder import PersonaBuilder
from shared.vectorstore.store import VectorStore

load_dotenv()
log = logging.getLogger(__name__)

LLM_RERANK_TOKENS = int(os.getenv("LLM_RERANK_TOKENS", "1500"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "15"))
TOP_K_RETURN = int(os.getenv("TOP_K_RETURN", "10"))


class RecommendationAgent:

    def __init__(self, persona_builder: Optional[PersonaBuilder] = None):
        log.info("Initialising RecommendationAgent ...")
        # Reuse a shared PersonaBuilder if provided (consolidated app),
        # otherwise build our own (standalone use).
        self.persona_builder = persona_builder or PersonaBuilder()
        self.vector_store = VectorStore()
        self._llm = None
        log.info("RecommendationAgent ready")

    async def recommend(
            self,
            user_id: str,
            context: str = "",
            conversation_history: list = None,
            top_k: int = TOP_K_RETURN,
    ) -> dict:
        conversation_history = conversation_history or []
        top_k = min(top_k, TOP_K_RETURN)

        # Stage 0: persona — O(1) dict lookup
        persona = self.persona_builder.build(user_id)
        persona = nigerian_apply(persona, user_context=context)

        bootstrap_query = (
                context
                or " ".join(persona.get("top_categories", [])[:2])
                or "restaurants"
        )

        # Stages 1 + 2 overlapped: intent reasoning (LLM) || retrieval (ChromaDB)
        intent_task = asyncio.to_thread(
            self._reason_intent_sync, persona, context, conversation_history
        )
        retrieve_task = asyncio.to_thread(
            self._retrieve_sync, persona, bootstrap_query
        )
        search_intent, candidates = await asyncio.gather(
            intent_task, retrieve_task, return_exceptions=True
        )

        if isinstance(search_intent, Exception):
            log.warning(f"Intent reasoning failed: {search_intent}")
            search_intent = bootstrap_query
        if isinstance(candidates, Exception):
            log.warning(f"Bootstrap retrieval failed: {candidates}")
            candidates = []

        log.info(f"Search intent [{user_id}]: {search_intent}")

        # One targeted retrieval only if the first came back sparse
        if len(candidates) < 5 and search_intent != bootstrap_query:
            extra = await asyncio.to_thread(self._retrieve_sync, persona, search_intent)
            seen = {c["business_id"] for c in candidates}
            for c in extra:
                if c["business_id"] not in seen:
                    candidates.append(c)
            candidates = candidates[:TOP_K_RETRIEVE]

        # Stage 3: rerank
        ranked = await asyncio.to_thread(
            self._rerank_sync, persona, candidates, search_intent, top_k
        )

        return {
            "recommendations": ranked,
            "cold_start": persona.get("is_cold", False),
            "persona_summary": _persona_summary(persona),
            "search_intent": search_intent,
        }

    def rerank_candidates(
            self,
            user_id: str,
            context: str,
            candidates: list,
            top_k: int = TOP_K_RETURN,
    ) -> dict:
        """Rerank a fixed candidate pool; returns the same shape as recommend()."""
        top_k = min(top_k, len(candidates)) if candidates else 0
        persona = self.persona_builder.build(user_id)
        persona = nigerian_apply(persona, user_context=context)

        search_intent = self._reason_intent_sync(persona, context, [])

        ranked = self._rerank_sync(persona, candidates, search_intent, top_k)

        return {
            "recommendations": ranked,
            "cold_start": persona.get("is_cold", False),
            "persona_summary": _persona_summary(persona),
            "search_intent": search_intent,
        }

    def _reason_intent_sync(self, persona, context, history) -> str:
        top_cats = persona.get("top_categories", [])[:3]

        history_str = ""
        if history:
            turns = history[-2:]
            history_str = " | ".join(
                f"{t.get('role', 'user')[:1].upper()}: {t.get('content', '')[:80]}"
                for t in turns
            )

        system = (
            "You produce a short search query of 5-10 words for a recommendation "
            "system. Output ONLY the query. No quotes, no trailing punctuation."
        )
        user = (
                f"Request: {context or 'general recommendation'}\n"
                f"User prefers: {', '.join(top_cats) if top_cats else 'unknown'}\n"
                + (f"Recent: {history_str}\n" if history_str else "")
                + "Query:"
        )

        try:
            intent = self._get_llm().complete(system=system, user=user, max_tokens=40)
            intent = intent.strip().strip('"').strip("'").rstrip(".")
            words = intent.split()
            return " ".join(words[:12]) if len(words) > 12 else intent
        except Exception as e:
            log.warning(f"Intent reasoning failed: {e}")
            return context or ", ".join(top_cats[:3]) or "restaurants"

    def _retrieve_sync(self, persona, query) -> list:
        candidates = self.vector_store.query_businesses(query=query, n=TOP_K_RETRIEVE)
        top_cats = persona.get("top_categories", [])
        if len(candidates) < 5 and top_cats:
            extra = self.vector_store.query_by_category(top_cats, n=TOP_K_RETRIEVE)
            seen = {c["business_id"] for c in candidates}
            for biz in extra:
                if biz["business_id"] not in seen:
                    candidates.append(biz)
        return candidates[:TOP_K_RETRIEVE]

    def _rerank_sync(self, persona, candidates, search_intent, top_k) -> list:
        if not candidates:
            return []

        candidate_list = "\n".join(
            f"{i + 1}. id={c['business_id']} | {c.get('name', '?')} | "
            f"{(c.get('categories', '') or '')[:50]}"
            for i, c in enumerate(candidates)
        )

        system = build_system_prompt(
            _RERANK_SYSTEM.format(
                top_k=top_k,
                mean_stars=persona.get("mean_stars", 3.5),
                bias=persona.get("rating_bias", "neutral"),
                top_cats=", ".join(persona.get("top_categories", ["any"])[:3]),
                is_cold=persona.get("is_cold", False),
                search_intent=search_intent,
            ),
            persona,
            task="recommend",
        )
        user = (
            f"Pick the best {top_k} from these {len(candidates)} businesses.\n\n"
            f"{candidate_list}\n\n"
            f'Return JSON: {{"ranked":[{{"business_id":"...","score":0.95,"reason":"..."}}]}}'
        )

        try:
            raw = self._get_llm().complete_json(
                system=system, user=user, max_tokens=LLM_RERANK_TOKENS,
            )
            parsed = json.loads(raw)
            ranked_map = {item["business_id"]: item for item in parsed.get("ranked", [])}
        except Exception as e:
            log.warning(f"LLM reranking failed ({e}) — distance fallback")
            return _distance_fallback(candidates, top_k)

        output = []
        for c in candidates:
            bid = c["business_id"]
            if bid in ranked_map:
                item = ranked_map[bid]
                output.append({
                    "business_id": bid,
                    "name": c.get("name", ""),
                    "categories": c.get("categories", ""),
                    "city": c.get("city", ""),
                    "stars": str(c.get("stars", "")),
                    "score": float(item.get("score", 0.5)),
                    "reason": item.get("reason", ""),
                })

        if len(output) < top_k:
            seen = {o["business_id"] for o in output}
            for fb in _distance_fallback(candidates, top_k):
                if fb["business_id"] not in seen:
                    output.append(fb)
                    if len(output) >= top_k:
                        break

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    def _get_llm(self):
        if self._llm is None:
            self._llm = llm()
        return self._llm


_RERANK_SYSTEM = """You are a recommendation reranker.

USER: typical {mean_stars:.1f}-star rater ({bias}), prefers {top_cats}. Cold-start: {is_cold}.
WANTS: {search_intent}

For each of the top {top_k} businesses, output a score in [0,1] and a one-sentence
reason matching this specific user's taste."""


def _persona_summary(persona: dict) -> dict:
    return {
        "is_cold": persona.get("is_cold", False),
        "review_count": persona.get("review_count", 0),
        "rating_bias": persona.get("rating_bias", "neutral"),
        "style": persona.get("style_label", "medium"),
        "nigerian": persona.get("nigerian_mode", False),
        "top_cats": persona.get("top_categories", []),
    }


def _distance_fallback(candidates: list, top_k: int) -> list:
    return [
        {
            "business_id": c["business_id"],
            "name": c.get("name", ""),
            "categories": c.get("categories", ""),
            "city": c.get("city", ""),
            "stars": str(c.get("stars", "")),
            "score": round(1 - float(c.get("distance", 0.5)), 3),
            "reason": "Matched on semantic similarity.",
        }
        for c in candidates[:top_k]
    ]
