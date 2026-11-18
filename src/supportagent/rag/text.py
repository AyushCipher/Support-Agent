"""Text normalization for retrieval: stdlib-only Porter stemmer, a small irregular-form lemma table, tokenizers.

Two analyzers exist on purpose:
  * "legacy"  - the original suffix stripper (kept byte-for-byte so the shipped BM25 numbers do not move)
  * "porter"  - Porter (1980) stemming plus irregular-verb lemmas (bought->buy, sent->send, lost->lose ...)
"""
from __future__ import annotations

import re

STOP = set("a an the of to is are in on for and or my i me can do does how what it be with this that your you if at as by have has had having something anything get got will would should could not was were been but from they them their there when where which who".split())
# Extra function words only used by the "porter" analyzer; the legacy analyzer is frozen.
STOP_EXTRA = set("please just also any so am us we our up out more some am im ive thing want need like very much many customer customers".split())

_IRREGULAR = {
    "bought": "buy", "paid": "pay", "sent": "send", "lost": "lose", "broke": "break", "broken": "break", "stolen": "steal",
    "took": "take", "taken": "take", "left": "leave", "held": "hold", "gone": "go", "went": "go", "came": "come",
    "ran": "run", "ordered": "order", "kept": "keep", "told": "tell", "won": "win", "made": "make", "spent": "spend",
    "charged": "charge", "billed": "bill", "froze": "freeze", "frozen": "freeze", "wrote": "write", "written": "write",
    "children": "child", "men": "man", "women": "woman",
}


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _measure(stem: str) -> int:
    n, i, ln = 0, 0, len(stem)
    while i < ln and _cons(stem, i):
        i += 1
    while i < ln:
        while i < ln and not _cons(stem, i):
            i += 1
        if i >= ln:
            break
        n += 1
        while i < ln and _cons(stem, i):
            i += 1
    return n


def _has_vowel(stem: str) -> bool:
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double_c(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    return len(w) >= 3 and _cons(w, len(w) - 3) and not _cons(w, len(w) - 2) and _cons(w, len(w) - 1) and w[-1] not in "wxy"


_STEP2 = {"ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize", "abli": "able", "alli": "al",
          "entli": "ent", "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al",
          "iveness": "ive", "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble"}
_STEP3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": ""}
_STEP4 = ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize")


def _replace_longest(w: str, table: dict[str, str], min_m: int) -> str:
    for suf in sorted(table, key=len, reverse=True):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            return stem + table[suf] if _measure(stem) >= min_m else w
    return w


def porter_stem(w: str) -> str:
    """Classic Porter stemmer (steps 1a-5b)."""
    if len(w) <= 2:
        return w
    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf) and _has_vowel(w[: -len(suf)]):
                w = w[: -len(suf)]
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif _double_c(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _cvc(w):
                    w += "e"
                break
    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    w = _replace_longest(w, _STEP2, 1)
    w = _replace_longest(w, _STEP3, 1)
    # step 4
    for suf in sorted(_STEP4, key=len, reverse=True):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 1 and (suf != "ion" or stem.endswith(("s", "t"))):
                w = stem
            break
    # step 5
    if w.endswith("e"):
        m = _measure(w[:-1])
        if m > 1 or (m == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    if _measure(w) > 1 and _double_c(w) and w.endswith("l"):
        w = w[:-1]
    return w


def lemma_stem(w: str) -> str:
    """Irregular-form lookup, then Porter."""
    return porter_stem(_IRREGULAR.get(w, w))


def legacy_stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def tokenize(s: str, analyzer: str = "legacy") -> list[str]:
    if analyzer == "legacy":
        return [legacy_stem(w) for w in words(s) if w not in STOP]
    if analyzer == "porter":
        return [lemma_stem(w) for w in words(s) if w not in STOP and w not in STOP_EXTRA]
    raise ValueError(f"unknown analyzer {analyzer!r}")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
