from __future__ import annotations

from dataclasses import dataclass, field

from .patterns import injection_score, redact_pii

FLAG_THRESHOLD = 0.6


@dataclass
class InputVerdict:
    text: str                        # text that goes to the model (PII redacted)
    flagged: bool = False
    score: float = 0.0
    labels: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)


def check_user_input(text: str, classifier=None) -> InputVerdict:
    """Layer 1 heuristics (+ optional LLM classifier hook returning 0..1). Flagging does NOT refuse the user:
    it taints the session (privilege restriction) and escalates the model tier - false positives stay cheap."""
    score, labels = injection_score(text)
    if classifier is not None:
        score = max(score, float(classifier(text)))
    clean, pii = redact_pii(text, kinds=("card", "ssn"))
    return InputVerdict(clean, score >= FLAG_THRESHOLD, score, labels, pii)
