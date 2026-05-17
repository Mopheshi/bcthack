"""
Nigerian Style Adapter — the bonus marks differentiator.

Two responsibilities:
  1. detect_nigerian_context(text)  → bool
     Detects Nigerian linguistic signals in any input text.

  2. build_nigerian_system_prompt(persona, task)  → str
     Injects Nigerian cultural + linguistic context into the LLM system prompt.

  3. apply(persona)  → persona (with nigerian_mode=True if detected)
     Convenience wrapper used by both Task A and Task B.
"""

import re
import logging

log = logging.getLogger(__name__)

# ── Nigerian signal vocabulary ────────────────────────────────────────────────
# These are linguistic markers common in Nigerian English and Naija Pidgin.
# Not exhaustive — enough to detect intent and guide tone.

NIGERIAN_LEXICAL_SIGNALS = {
    # Pidgin function words
    "na", "dem", "dey", "abi", "sha", "oya", "wahala", "chop", "wey",
    "sabi", "gist", "jare", "abeg", "ehen", "chai", "nawa", "shey", "omo",
    # Nigerian food & culture
    "suya", "egusi", "jollof", "puff puff", "akara", "buka", "mama put",
    "eba", "amala", "tuwo", "pepper soup", "nkwobi", "ofe", "banga",
    # Common Nigerian expressions in English
    "how far", "e be like", "no dulling", "correct", "ogbonge",
    "area boys", "danfo", "okada", "mallam", "oga", "madam",
    "nigeria", "nigerian", "naija",
}

NIGERIAN_CITIES = {
    "lagos", "abuja", "kano", "ibadan", "benin city", "port harcourt",
    "kaduna", "enugu", "onitsha", "aba", "jos", "ilorin", "maiduguri",
    "zaria", "abeokuta", "warri", "calabar", "uyo", "owerri", "akure",
}

# Regex for common Naija Pidgin patterns
PIDGIN_PATTERN = re.compile(
    r"\b(na|dem|dey|abi|sha|oya|abeg|shey|jare|wahala|sabi|nawa|chai)\b",
    re.IGNORECASE,
)


def detect_nigerian_context(text: str) -> bool:
    """
    Returns True if the text contains Nigerian linguistic signals.
    Used to auto-enable nigerian_mode when input text carries those signals.
    """
    if not text:
        return False

    lower = text.lower()

    # Check Pidgin regex
    if PIDGIN_PATTERN.search(lower):
        return True

    # Check lexical signals (word-boundary safe)
    words = set(re.findall(r"\b\w[\w\s]*\w\b|\b\w\b", lower))
    if words & NIGERIAN_LEXICAL_SIGNALS:
        return True

    # Check city names
    for city in NIGERIAN_CITIES:
        if city in lower:
            return True

    return False


def apply(persona: dict, user_context: str = "") -> dict:
    """
    Checks if nigerian_mode should be enabled based on user context
    or existing persona signals, then sets persona['nigerian_mode'].
    Mutates and returns the persona dict.
    """
    if persona.get("nigerian_mode"):
        return persona  # already set upstream

    # Auto-detect from categories or context text
    categories_str = " ".join(persona.get("top_categories", []))
    if detect_nigerian_context(user_context) or detect_nigerian_context(categories_str):
        persona["nigerian_mode"] = True
        log.debug(f"Nigerian mode enabled for user {persona.get('user_id')}")

    return persona


def build_system_prompt(base_prompt: str, persona: dict, task: str = "review") -> str:
    """
    Wraps a base system prompt with Nigerian cultural + linguistic instructions
    when persona['nigerian_mode'] is True.

    task: "review" (Task A) or "recommend" (Task B)
    """
    if not persona.get("nigerian_mode"):
        return base_prompt

    nigerian_layer = _nigerian_instructions(task)
    return f"{base_prompt}\n\n{nigerian_layer}"


def _nigerian_instructions(task: str) -> str:
    base = """
NIGERIAN CULTURAL CONTEXT:
You are generating output for a Nigerian user. Apply the following naturally — 
do NOT force every sentence, but let these traits emerge authentically:

Language & Tone:
- Nigerian English is direct, expressive, and warm. Avoid overly formal American phrasing.
- Occasional Naija Pidgin words are natural: "abeg", "sha", "abi", "wahala", "correct", "oya".
- Exclamations like "Chai!", "Nawa o!", "Hmmm" signal strong emotion authentically.
- Nigerians often intensify: "very very good", "too sweet", "I no go lie".
- Humour and sarcasm are common even in negative reviews.

Food & Context References (where relevant):
- Local equivalents resonate: compare to jollof rice, suya, buka experience.
- Service speed and "Nigerian time" are known cultural reference points.
- Family, community, and "correct business" framing resonate strongly.

What to AVOID:
- Do not use American slang ("awesome sauce", "y'all", "super duper").
- Do not force Pidgin if the context is clearly formal or professional.
- Do not stereotype — write as an educated, urban Nigerian would naturally write.
"""

    if task == "recommend":
        base += """
Recommendation Framing:
- Frame recommendations as you would advise a friend: "I go tell you say..."
- Reference local equivalents when making analogies.
- Trust and community signals matter: "people dey go there since forever" carries weight.
"""
    elif task == "review":
        base += """
Review Writing Style:
- Nigerian reviewers are honest and direct — they will call out bad service bluntly.
- Positive reviews often include food comparisons or family/group context.
- Star ratings from positive-biased Nigerian users skew high when satisfied.
"""

    return base.strip()
