"""
shared/llm/client.py  (v2 — google-genai SDK)
----------------------------------------------
Unified LLM client. Set LLM_PROVIDER in .env to switch.

Supported:
  LLM_PROVIDER=gemini      → google-genai (new SDK, replaces google-generativeai)
  LLM_PROVIDER=anthropic   → Anthropic Claude
  LLM_PROVIDER=openai      → OpenAI

Two methods on every client:
  complete(system, user, max_tokens)          → str
  complete_json(system, user, max_tokens)     → str  (enforced JSON output)
"""

import os
import logging
import json
import re
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

PROVIDER   = os.getenv("LLM_PROVIDER",  "gemini").lower()
MODEL      = os.getenv("LLM_MODEL",     "gemini-2.5-flash")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))


# ── Gemini (google-genai SDK) ─────────────────────────────────────────────────

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

        self._genai       = genai
        self._types       = genai_types
        self._client      = genai.Client(api_key=api_key)
        self._model       = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        log.info(f"GeminiClient initialised — model: {self._model}")

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        """Standard text completion."""
        # Gemini takes system via system_instruction, user via contents
        config = self._types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=0.7,
            # Add safety settings to avoid it returning None on slightly sensitive text
            safety_settings=[
                self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )
        log.info(f"Gemini complete() max_tokens: {max_tokens}")
        response = self._client.models.generate_content(
            model    = self._model,
            contents = user,
            config   = config,
        )
        # If response is blocked, return empty string instead of None
        if getattr(response, "candidates", None) and response.candidates[0].finish_reason:
             log.info(f"Finish reason: {response.candidates[0].finish_reason}")
             log.info(f"Raw text length: {len(response.text) if response.text else 0}")
        if not response.candidates:
             return ""
        text = response.text or ""
        return text.strip()

    def complete_json(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        """
        JSON-enforced completion using Gemini's native structured output.
        Guarantees the response is valid JSON — no regex parsing needed.
        """
        config = self._types.GenerateContentConfig(
            system_instruction  = system if system else None,
            temperature         = 0.2,          # lower temp for deterministic JSON
            response_mime_type  = "application/json",
            safety_settings=[
                self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )
        response = self._client.models.generate_content(
            model    = self._model,
            contents = user,
            config   = config,
        )
        if not response.candidates:
             return "{}"
        text = response.text or "{}"
        return _extract_json(text.strip())


# ── Anthropic ─────────────────────────────────────────────────────────────────

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
        # Anthropic: append JSON instruction to system prompt
        json_system = system + "\n\nYou MUST respond with valid JSON only. No markdown, no preamble."
        return _extract_json(self.complete(json_system, user, max_tokens))


# ── OpenAI ────────────────────────────────────────────────────────────────────

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


# ── Factory + singleton ────────────────────────────────────────────────────────

_PROVIDERS = {"gemini": GeminiClient, "anthropic": AnthropicClient, "openai": OpenAIClient}

_client = None

def llm() -> GeminiClient:
    global _client
    if _client is None:
        if PROVIDER not in _PROVIDERS:
            raise ValueError(f"Unknown LLM_PROVIDER='{PROVIDER}'. Choose: {list(_PROVIDERS)}")
        _client = _PROVIDERS[PROVIDER]()
    return _client

def _extract_json(text: str) -> str:
    """Helper to strip Markdown formatting if any model returns ```json ... ```"""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Also find first { or [ just in case
    start = text.find('{')
    start_arr = text.find('[')
    if start == -1 and start_arr == -1:
        return text
    if start != -1 and (start_arr == -1 or start < start_arr):
        return text[start:]
    return text[start_arr:]
