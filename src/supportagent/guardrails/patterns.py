"""Shared injection heuristics and PII patterns. Heuristics are layer 1; see docs/adr/0004 for the layered design
(+ optional LLM classifier hook, + structural defenses that do not depend on detection at all)."""
from __future__ import annotations

import math
import re

# (regex, weight, label)
INJECTION_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?|messages?)", re.I), 0.9, "ignore_previous"),
    (re.compile(r"disregard\s+(all\s+|any\s+|your\s+)?(previous|prior|above|earlier)?\s*(instructions?|rules?|polic(y|ies)|guidelines?)", re.I), 0.9, "disregard"),
    (re.compile(r"(you\s+are\s+now|from\s+now\s+on\s+you)\b.{0,40}(admin|developer|jailbreak|dan|unrestricted|root)", re.I), 0.8, "role_override"),
    (re.compile(r"\b(admin|developer|debug|god)\s+mode\b", re.I), 0.7, "mode_switch"),
    (re.compile(r"(^|\n|\.\s+)\s*(system|assistant|developer)\s*(override)?\s*:", re.I), 0.7, "fake_role_prefix"),
    (re.compile(r"\bsystem\s+override\b", re.I), 0.8, "system_override"),
    (re.compile(r"(policy|manager|supervisor)\s+(override|exception)\s+(is\s+)?(approved|authori[sz]ed|granted)", re.I), 0.8, "claimed_authorization"),
    (re.compile(r"(approved|authori[sz]ed)\s+by\s+(the\s+)?(manager|supervisor|admin|ceo)", re.I), 0.6, "claimed_authorization"),
    (re.compile(r"without\s+(any\s+)?(verification|checking|approval)", re.I), 0.5, "skip_checks"),
    (re.compile(r"do\s+not\s+(mention|tell|reveal|inform)\b", re.I), 0.6, "concealment"),
    (re.compile(r"(print|reveal|show|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)", re.I), 0.8, "prompt_extraction"),
    (re.compile(r"\byou\s+must\s+(now\s+)?(immediately\s+)?(refund|issue|transfer|call|send)\b", re.I), 0.7, "imperative_to_agent"),
    (re.compile(r"(refund|issue)\s+(a\s+)?(full\s+)?(refund\s+)?(of\s+)?\$?\d[\d,]*.{0,30}(immediately|now|right away)", re.I), 0.6, "urgent_money_instruction"),
    (re.compile(r"<\|?(im_start|system|endoftext)\|?>|\[/?INST\]", re.I), 0.9, "chat_template_token"),
]

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SECRET_RE = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|CANARY-[0-9a-f]{6})\b")


def luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(ds) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def injection_score(text: str) -> tuple[float, list[str]]:
    hits = [(w, label) for rx, w, label in INJECTION_PATTERNS if rx.search(text)]
    if not hits:
        return 0.0, []
    score = 1 - math.prod(1 - w for w, _ in hits)      # noisy-or
    return round(score, 3), sorted({label for _, label in hits})


def redact_pii(text: str, kinds: tuple[str, ...] = ("card", "ssn", "phone")) -> tuple[str, list[str]]:
    found: list[str] = []
    if "ssn" in kinds and SSN_RE.search(text):
        text, found = SSN_RE.sub("[REDACTED-SSN]", text), found + ["ssn"]
    if "card" in kinds:
        def _card(m: re.Match) -> str:
            if luhn_ok(m.group(0)):
                found.append("card")
                return "[REDACTED-CARD]"
            return m.group(0)
        text = CARD_RE.sub(_card, text)
    if "phone" in kinds and PHONE_RE.search(text):
        text, found = PHONE_RE.sub("[REDACTED-PHONE]", text), found + ["phone"]
    if "email" in kinds and EMAIL_RE.search(text):
        text, found = EMAIL_RE.sub("[REDACTED-EMAIL]", text), found + ["email"]
    return text, found
