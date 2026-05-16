"""
shared/llm/client.py  (v3 — Gemini 3.x thinking control + retries)
-------------------------------------------------------------------
Unified LLM client. Set LLM_PROVIDER in .env to switch.

WHAT'S NEW IN v3
  - thinking_level support for Gemini 3.x models. Gemini 3 replaced the
    old thinking_budget integer with a thinking_level string
    (minimal|low|medium|high). For low-latency structured tasks we want
    'minimal'. This is set via .env: LLM_THINKING_LEVEL=minimal
  - Backward compatible: if the model is a 2.5 model, thinking_budget=0
    is used instead (the old mechanism).
  - Bounded retry with backoff on transient API errors (503/timeout).
  - max_output_tokens is actually passed through now.

Supported:
  LLM_PROVIDER=gemini      -> google-genai
  LLM_PROVIDER=anthropic   -> Anthropic Claude
  LLM_PROVIDER=openai      -> OpenAI

Methods on every client:
  complete(system, user, max_tokens)        -> str
  complete_json(system, user, max_tokens)   -> str  (JSON-enforced)
"""

import os
import re
import time
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PROVIDER       = os.getenv("LLM_PROVIDER", "gemini").lower()
MODEL          = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
MAX_TOKENS     = int(os.getenv("LLM_MAX_TOKENS", "1000"))
THINKING_LEVEL = os.getenv("LLM_THINKING_LEVEL", "minimal").lower()
MAX_RETRIES    = int(os.getenv("LLM_MAX_RETRIES", "2"))


# ── Gemini (google-genai SDK) ────────────────────────────────────────────────

class GeminiClient:
    def __init__(self):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")

        self._types  = genai_types
        self._client = genai.Client(api_key=api_key)
        self._model  = MODEL
        self._is_gemini3 = self._model.startswith("gemini-3")
        log.info(f"GeminiClient initialised — model: {self._model}, "
                 f"thinking: {THINKING_LEVEL if self._is_gemini3 else 'budget=0'}")

    def _thinking_config(self):
        """
        Gemini 3.x uses thinking_level (minimal|low|medium|high).
        Gemini 2.5 uses thinking_budget (integer; 0 disables).
        Returns the right ThinkingConfig for the active model, or None.
        """
        try:
            if self._is_gemini3:
                return self._types.ThinkingConfig(thinking_level=THINKING_LEVEL)
            else:
                return self._types.ThinkingConfig(thinking_budget=0)
        except Exception as e:
            # If the SDK version doesn't support the field, skip it gracefully.
            log.debug(f"ThinkingConfig unavailable: {e}")
            return None

    def _safety(self):
        return [
            self._types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ]

    def _generate(self, system, user, config):
        """Single call with bounded retry on transient errors."""
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=user, config=config,
                )
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                transient = any(s in msg for s in
                                ("503", "500", "unavailable", "timeout", "deadline", "overloaded"))
                if transient and attempt < MAX_RETRIES:
                    wait = 1.5 * (attempt + 1)
                    log.warning(f"Gemini transient error (attempt {attempt+1}): {e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
        raise last_err

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        cfg_kwargs = dict(
            system_instruction=system or None,
            temperature=0.7,
            max_output_tokens=max_tokens,
            safety_settings=self._safety(),
        )
        tc = self._thinking_config()
        if tc is not None:
            cfg_kwargs["thinking_config"] = tc

        config   = self._types.GenerateContentConfig(**cfg_kwargs)
        response = self._generate(system, user, config)

        if not getattr(response, "candidates", None):
            log.warning("Gemini returned no candidates")
            return ""
        return (response.text or "").strip()

    def complete_json(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        cfg_kwargs = dict(
            system_instruction=system or None,
            temperature=0.2,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            safety_settings=self._safety(),
        )
        tc = self._thinking_config()
        if tc is not None:
            cfg_kwargs["thinking_config"] = tc

        config   = self._types.GenerateContentConfig(**cfg_kwargs)
        response = self._generate(system, user, config)

        if not getattr(response, "candidates", None):
            return "{}"
        return _extract_json((response.text or "{}").strip())


# ── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        log.info(f"AnthropicClient initialised — model: {self._model}")

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        r = self._client.messages.create(
            model=self._model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text.strip()

    def complete_json(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        json_system = system + "\n\nRespond with valid JSON only. No markdown, no preamble."
        return _extract_json(self.complete(json_system, user, max_tokens))


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self._client = OpenAI(api_key=api_key)
        self._model  = os.getenv("LLM_MODEL", "gpt-4o-mini")
        log.info(f"OpenAIClient initialised — model: {self._model}")

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        r = self._client.chat.completions.create(
            model=self._model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        return r.choices[0].message.content.strip()

    def complete_json(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        r = self._client.chat.completions.create(
            model=self._model, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
        )
        return _extract_json(r.choices[0].message.content.strip())


# ── Factory + singleton ──────────────────────────────────────────────────────

_PROVIDERS = {"gemini": GeminiClient, "anthropic": AnthropicClient, "openai": OpenAIClient}
_client = None


def llm():
    global _client
    if _client is None:
        if PROVIDER not in _PROVIDERS:
            raise ValueError(f"Unknown LLM_PROVIDER='{PROVIDER}'. Choose: {list(_PROVIDERS)}")
        _client = _PROVIDERS[PROVIDER]()
    return _client


def _extract_json(text: str) -> str:
    """Strip markdown fences and locate the first JSON object/array."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    start     = text.find('{')
    start_arr = text.find('[')
    if start == -1 and start_arr == -1:
        return text
    if start != -1 and (start_arr == -1 or start < start_arr):
        return text[start:]
    return text[start_arr:]
