"""User simulators. ScriptedUser is deterministic (CI); LLMUser drives the same personas with a real model (live evals)."""
from __future__ import annotations

import re

ASKS_ORDER = re.compile(r"order number|which order|order id", re.I)
ASKS_SUB = re.compile(r"which subscription|which plan", re.I)
DENIAL = re.compile(r"can't|cannot|unable|isn't eligible|not eligible|wasn't able|unfortunately", re.I)


class ScriptedUser:
    def __init__(self, persona: dict, max_turns: int = 5):
        self.p, self.pushback, self.turn, self.max = persona, list(persona.get("pushback", [])), 0, max_turns
        self.follow = list(persona.get("followups", []))

    def next(self, agent_reply: str | None) -> str | None:
        if agent_reply is None:
            self.turn += 1
            return self.p["opening"]
        if self.turn >= self.max:
            return None
        facts = self.p.get("facts", {})
        msg = None
        if ASKS_ORDER.search(agent_reply) and "order_id" in facts:
            msg = f"It's {facts['order_id']}."
        elif ASKS_SUB.search(agent_reply) and "sub_id" in facts:
            msg = f"The one I mean is {facts['sub_id']}."
        elif DENIAL.search(agent_reply) and self.pushback:
            msg = self.pushback.pop(0)
        if msg is None and self.follow:
            msg = self.follow.pop(0)
        if msg:
            self.turn += 1
        return msg


class LLMUser:
    """Live-eval simulator: an LLM role-plays the persona. Ends when it replies [END]."""
    def __init__(self, persona: dict, llm, max_turns: int = 6):
        self.p, self.llm, self.max, self.turn, self.history = persona, llm, max_turns, 0, []

    def next(self, agent_reply: str | None) -> str | None:
        from supportagent.harness.types import Message
        if agent_reply is None:
            self.turn += 1
            self.history.append(Message("assistant", self.p["opening"]))
            return self.p["opening"]
        if self.turn >= self.max:
            return None
        self.history.append(Message("user", agent_reply))
        system = (f"You role-play a customer ({self.p['style']}). Facts you know: {self.p['facts']}. Possible follow-ups: {self.p['pushback']}. "
                  "Reply with ONLY the customer's next message, or [END] if satisfied or the conversation is over.")
        out = self.llm.complete(system, self.history, [], 200).text.strip()
        if out.startswith("[END]") or not out:
            return None
        self.turn += 1
        self.history.append(Message("assistant", out))
        return out
