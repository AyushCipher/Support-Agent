"""Query expansion / rewriting hooks for the lexical side of retrieval.

An expander is any object with `expand(query) -> str`. The returned text is used ONLY for BM25 (the dense side sees the
raw question), and expanded terms are appended so the original wording is never lost.

  VocabCorrector     spelling repair against the corpus vocabulary (difflib), for typos like "refnd" or "delivry"
  ThesaurusExpander  small hand-written support-domain synonym groups and phrase rules (authored once, before any tuning;
                     see docs/adr/0012). It is deliberately generic, not a list of test questions.
  CallableRewriter   hook for an LLM/other rewriter: `CallableRewriter(lambda q: my_model(q))`. Never used by default
                     (a live rewrite costs money and would make evals non-deterministic).
  ChainExpander      apply several in order
"""
from __future__ import annotations

import difflib
import re
from typing import Callable, Protocol

from .text import STOP, legacy_stem, words


class Expander(Protocol):
    def expand(self, query: str) -> str: ...


# Each group: words a customer or an agent might use interchangeably. Lookup uses a light plural/-ed/-ing stripper (not Porter: Porter conflates customer and customs).
SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("package", "parcel", "shipment", "consignment"),
    ("refund", "reimburse", "reimbursement", "repay", "payback"),
    ("cancel", "terminate", "discontinue"),
    ("broken", "defective", "faulty", "damaged", "smashed", "cracked", "malfunction"),
    ("delete", "erase", "remove", "wipe", "erasure"),
    ("fraud", "fraudulent", "scam", "hacked", "hijacked", "unauthorized", "unrecognized"),
    ("charge", "bill", "debit"),
    ("guarantee", "warranty"),
    ("fix", "repair", "mend"),
    ("replace", "swap", "exchange", "replacement"),
    ("email", "inbox", "newsletter"),
    ("promotional", "marketing", "promo", "advertising", "offers"),
    ("lawyer", "solicitor", "attorney", "lawsuit", "sue", "court", "legal"),
    ("press", "journalist", "reporter", "media", "newspaper"),
    ("weekend", "saturday", "sunday"),
    ("abroad", "overseas", "international", "foreign", "border"),
    ("customs", "duty", "import", "tariff", "levy"),
    ("receipt", "invoice", "proof"),
    ("membership", "subscription", "plan"),
    ("yearly", "annual", "annually"),
    ("discount", "coupon", "voucher", "code", "promo"),
    ("fast", "fastest", "quick", "express", "rush"),
    ("deadline", "window", "period"),
    ("bounce", "declined", "rejected", "failed"),
    ("rude", "abusive", "offensive", "insult"),
    ("accessible", "braille", "disability"),
    ("germany", "france", "spain", "italy", "berlin", "paris", "europe", "european"),
    ("uk", "britain", "england", "london", "kingdom"),
    ("stolen", "theft", "steal"),
    ("late", "delayed", "overdue", "slow"),
]

# phrase -> extra terms appended when the phrase is present in the query
PHRASE_RULES: dict[str, str] = {
    "money back": "refund",
    "send back": "return",
    "give back": "return",
    "never arrived": "lost missing delivered",
    "not arrived": "lost missing delivered",
    "has not arrived": "lost missing delivered",
    "hasn't arrived": "lost missing delivered",
    "never came": "lost missing delivered",
    "did not arrive": "lost missing delivered",
    "nothing arrived": "lost missing delivered",
}


class ThesaurusExpander:
    def __init__(self, groups=SYNONYM_GROUPS, phrases=PHRASE_RULES):
        self._by_stem: dict[str, tuple[str, ...]] = {}
        for g in groups:
            for w in g:
                self._by_stem.setdefault(legacy_stem(w), g)
        self.phrases = phrases

    def expand(self, query: str) -> str:
        q = query.lower()
        extra: list[str] = []
        for phrase, add in self.phrases.items():
            if phrase in q:
                extra.append(add)
        seen: set[str] = set()
        for w in words(q):
            g = self._by_stem.get(legacy_stem(w))
            if g and w not in STOP and id(g) not in seen:
                seen.add(id(g))
                extra.extend(x for x in g if legacy_stem(x) != legacy_stem(w))
        return query + (" " + " ".join(extra) if extra else "")


class VocabCorrector:
    """Replace out-of-vocabulary words with the closest corpus word (similarity >= cutoff). The corrected word is appended,
    the original kept (it may be a legitimate word we simply do not use)."""

    def __init__(self, vocabulary: set[str], cutoff: float = 0.8, min_len: int = 5):
        self.vocab, self.cutoff, self.min_len = vocabulary, cutoff, min_len
        self._pool = sorted(vocabulary)

    def expand(self, query: str) -> str:
        fixes = []
        for w in words(query):
            if len(w) >= self.min_len and w not in self.vocab and w not in STOP and not w.isdigit():
                m = difflib.get_close_matches(w, self._pool, n=1, cutoff=self.cutoff)
                if m:
                    fixes.append(m[0])
        return query + (" " + " ".join(fixes) if fixes else "")


class CallableRewriter:
    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def expand(self, query: str) -> str:
        out = (self.fn(query) or "").strip()
        return f"{query} {out}" if out and out != query else query


class ChainExpander:
    def __init__(self, *expanders: Expander):
        self.expanders = expanders

    def expand(self, query: str) -> str:
        for e in self.expanders:
            query = e.expand(query)
        return re.sub(r"\s+", " ", query).strip()
