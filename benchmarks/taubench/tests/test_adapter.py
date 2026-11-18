"""Offline adapter tests: scripted agent + scripted user, REAL tau-bench retail env and reward. No network, no key.

Run:  benchmarks/taubench/.venv/Scripts/python -m pytest -q benchmarks/taubench/tests   (needs vendor/ fetched, see fetch_vendor.sh)
"""
from __future__ import annotations

import json

import pytest

from supportagent.harness.types import LLMResponse, ToolCall, Usage
from taubench_sa import llm as L

GT0_EXCHANGE = {"order_id": "#W2378156", "item_ids": ["1151293680", "4983901480"], "new_item_ids": ["7706410293", "7747408585"],
                "payment_method_id": "credit_card_9513926"}


class Script:
    """Scripted model: returns the next planned response, ignoring the input (records what it was shown)."""
    def __init__(self, plan):
        self.plan, self.seen, self.requests = list(plan), [], 1

    def complete(self, system, messages, tools, max_tokens=1024, tool_choice=None):
        self.seen.append({"system": system, "messages": [m.to_dict() for m in messages], "tools": [t.public_schema() | {"name": t.name} for t in tools]})
        step = self.plan.pop(0)
        if isinstance(step, str):
            return LLMResponse(step, [], Usage(10, 5), "scripted")
        name, args = step
        return LLMResponse("", [ToolCall(f"call_{len(self.seen)}", name, args)], Usage(10, 5), "scripted")


@pytest.fixture()
def world(tmp_path):
    """Installs the litellm shim with scripted inners; returns a factory for (env, agent_script, user_script)."""
    budget = L.Budget(10_000)
    scripts: dict[str, Script] = {}

    def make_llm(model, role):
        return L.CachedGemini(model, role, budget, tmp_path / "cache", inner_factory=lambda m: scripts[role])

    L.install_litellm_shim(L.LiteLLMShim(make_llm))
    from taubench_sa.adapter import import_tau
    from taubench_sa.run import make_user_env
    get_env, Action = import_tau()

    def mk(task_index, agent_plan, user_plan, run_key):
        L.CTX.run_key, L.CTX.calls_by_model = run_key, {}
        scripts["agent"], scripts["user"] = Script(agent_plan), Script(user_plan)
        env = make_user_env(get_env, "scripted-user", task_index)
        return env, (lambda tier, seed: make_llm("scripted-" + tier, "agent")), Action
    mk.scripts, mk.budget = scripts, budget
    return mk


def _sa(world, task, agent_plan, user_plan, key, **kw):
    from taubench_sa.adapter import run_supportagent_episode, supportagent_config
    env, factory, Action = world(task, agent_plan, user_plan, key)
    return run_supportagent_episode(env, task, factory, Action, supportagent_config(**kw)), env


def test_full_episode_passes_with_real_tau_reward(world):
    agent = [("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
             ("get_order_details", {"order_id": "#W2378156"}),
             ("get_product_details", {"product_id": "1656367028"}),
             ("get_product_details", {"product_id": "4896585277"}),
             "I will exchange the keyboard (1151293680 -> 7706410293) and thermostat (4983901480 -> 7747408585) on #W2378156, "
             "price difference to credit_card_9513926. Shall I proceed?",
             ("exchange_delivered_order_items", GT0_EXCHANGE),
             "Done: the exchange was requested."]
    user = ["Hi, I'm Yusuf Rossi, zip 19122, I want to exchange items from order #W2378156.", "Yes, please proceed.", "###STOP###"]
    res, env = _sa(world, 0, agent, user, "t-pass")
    assert res["reward"] == 1.0 and res["termination"] == "user_stop"
    assert [c["name"] for c in res["tool_calls"]][-1] == "exchange_delivered_order_items"
    assert res["guard_blocks"] == {}
    shown = world.scripts["agent"].seen[0]
    names = {t["name"]: t for t in shown["tools"]}
    assert "acting_customer_id" not in names["get_order_details"]["properties"]        # hidden authz param never shown to the model
    assert "idempotency_key" not in names["exchange_delivered_order_items"]["properties"]
    assert "ask_user" in names and "Retail agent policy" in shown["system"]


def test_write_without_confirmation_is_blocked_and_never_reaches_env(world):
    agent = [("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
             ("exchange_delivered_order_items", GT0_EXCHANGE),
             "I need your confirmation first."]
    user = ["I'm Yusuf Rossi in 19122, exchange stuff in #W2378156", "###STOP###"]
    res, env = _sa(world, 0, agent, user, "t-noconfirm")
    assert res["guard_blocks"] == {"no_user_confirmation": 1}
    assert "exchange_delivered_order_items" not in [c["name"] for c in res["tool_calls"]]
    assert res["reward"] == 0.0


def test_unauthenticated_and_cross_customer_access_blocked(world):
    agent = [("get_order_details", {"order_id": "#W2378156"}),                       # before authentication
             ("find_user_id_by_email", {"email": "sofia.rossi2645@example.com"}),
             ("get_order_details", {"order_id": "#W2378156"}),                       # Yusuf's order, Sofia authenticated
             ("get_user_details", {"user_id": "yusuf_rossi_9620"}),
             "Sorry, I can only help with your own account."]
    user = ["My email is sofia.rossi2645@example.com, show me order #W2378156", "###STOP###"]
    res, env = _sa(world, 0, agent, user, "t-authz")
    assert res["guard_blocks"] == {"not_authenticated": 1, "authz_other_user": 2}
    assert [c["name"] for c in res["tool_calls"]] == ["find_user_id_by_email"]
    # the email reached the env de-tokenized (the model only ever saw a placeholder)
    assert res["tool_calls"][0]["kwargs"]["email"] == "sofia.rossi2645@example.com"
    assert "sofia.rossi2645@example.com" not in json.dumps(world.scripts["agent"].seen[0]["messages"])


def test_over_cap_money_action_requests_approval_then_executes(world):
    addr = {"user_id": "sofia_rossi_8776", "address1": "1 Main St", "address2": "Apt 2", "city": "Austin", "state": "TX", "country": "USA", "zip": "78701"}
    agent = [("find_user_id_by_email", {"email": "sofia.rossi2645@example.com"}),
             ("get_order_details", {"order_id": "#W5918442"}),
             "Cancel #W5918442 (reason: no longer needed) and update your address? Please confirm.",
             ("cancel_pending_order", {"order_id": "#W5918442", "reason": "no longer needed"}),
             ("cancel_pending_order", {"order_id": "#W5918442", "reason": "no longer needed"}),   # now 'cancelled' -> status precondition
             ("modify_user_address", addr),
             ("modify_user_address", addr),                                                     # identical repeat -> idempotent replay
             "I've cancelled your order and updated your address."]
    user = ["sofia.rossi2645@example.com - cancel order #W5918442 and change my address", "yes", "###STOP###"]
    res, env = _sa(world, 0, agent, user, "t-approval")
    assert res["approvals"] == 1                       # $1463.70 > $100 autonomous cap -> human approval (eval auto-approver approves)
    assert res["guard_blocks"] == {"policy_status": 1}
    names = [c["name"] for c in res["tool_calls"]]
    assert names.count("cancel_pending_order") == 1 and names.count("modify_user_address") == 1   # second address call replayed
    assert res["output_guard_blocks"] == 0             # "I've cancelled" is backed by evidence recorded from the tool result


def test_transfer_ends_episode_and_step_cap(world):
    agent = [("transfer_to_human_agents", {"summary": "customer wants a human"}), "unused"]
    res, _ = _sa(world, 0, agent, ["I want a human."], "t-transfer")
    assert res["termination"] == "transfer"
    res2, _ = _sa(world, 0, [("think", {"thought": "hmm"})] * 40, ["hello"], "t-cap")
    assert res2["termination"] == "step_cap" and res2["n_agent_calls"] == L.AGENT_CALL_CAP + 1


def test_schema_rejection_goes_back_to_model(world):
    agent = [("find_user_id_by_email", {"email": 123}), "Could you repeat your email?"]
    res, _ = _sa(world, 0, agent, ["hi", "###STOP###"], "t-schema")
    assert res["schema_rejects"] == 1 and res["tool_calls"] == []


def test_baseline_tool_calling_agent_runs_through_shim(world):
    from taubench_sa.adapter import run_baseline_episode
    env, _, Action = world(0, [("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
                               ("get_order_details", {"order_id": "#W2378156"}),
                               ("exchange_delivered_order_items", GT0_EXCHANGE), "Done."],
                           ["Yusuf Rossi 19122, exchange in #W2378156", "###STOP###"], "t-base")
    res = run_baseline_episode(env, 0, "scripted-baseline", Action)
    assert res["reward"] == 1.0 and res["termination"] == "user_stop"
    # tool results were sent back to the model as tool messages, ids round-tripped through the OpenAI-format conversion
    msgs = world.scripts["agent"].seen[-1]["messages"]
    assert [m["role"] for m in msgs][:3] == ["user", "assistant", "tool"] and msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]


def test_cache_replays_episode_without_new_requests(world):
    agent = [("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}), "How can I help?"]
    _sa(world, 0, list(agent), ["hi", "###STOP###"], "t-cache")
    hits0 = world.budget.cache_hits
    res, _ = _sa(world, 0, [], [], "t-cache")          # empty scripts: any cache miss would raise IndexError
    assert res["reward"] in (0.0, 1.0) and world.budget.cache_hits >= hits0 + 4


def test_categorize_rules():
    from taubench_sa.adapter import categorize
    gt = [{"name": "cancel_pending_order", "kwargs": {"order_id": "#A", "reason": "no longer needed"}}]
    base = {"reward": 0.0, "termination": "user_stop", "tool_calls": [], "blocked_tools": []}
    assert categorize({**base, "reward": 1.0}, gt, False) == "pass"
    assert categorize({**base, "termination": "step_cap"}, gt, False) == "step_cap"
    assert categorize({**base, "termination": "transfer"}, gt, False) == "transferred_to_human"
    assert categorize(base, gt, False) == "missing_write"
    assert categorize({**base, "blocked_tools": ["cancel_pending_order"]}, gt, False) == "guard_blocked_required_write"
    wrong = {**base, "tool_calls": [{"name": "cancel_pending_order", "kwargs": {"order_id": "#A", "reason": "ordered by mistake"}, "error": False}]}
    assert categorize(wrong, gt, False) == "wrong_write"
    ok = {**base, "tool_calls": [{"name": "cancel_pending_order", "kwargs": gt[0]["kwargs"], "error": False}]}
    assert categorize(ok, gt, True) == "missing_required_info"
