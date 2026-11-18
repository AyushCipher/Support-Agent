import shutil
import sqlite3

import pytest

from supportagent.config import PRESETS
from supportagent.guardrails.approval import ApprovalService
from supportagent.harness.checkpoint import CheckpointStore
from supportagent.harness.runtime import Runtime
from supportagent.mcp.client import build_registry
from supportagent.memory.store import CustomerMemory
from supportagent.obs.tracing import MemorySink, Tracer
from supportagent.shop.db import build_seed_db, connect

EMAILS = {"C001": "alice@example.com", "C002": "bob@example.com", "C005": "eve@example.com"}


@pytest.fixture(scope="session")
def seed_db(tmp_path_factory):
    return build_seed_db(tmp_path_factory.mktemp("seed") / "seed.db")


@pytest.fixture
def db(tmp_path, seed_db):
    p = tmp_path / "shop.db"
    shutil.copy(seed_db, p)
    return str(p)


@pytest.fixture
def make_rt(db):
    def _make(config="full", reviewer=lambda rec: (True, "ok"), async_approvals=False, llm_factory=None, **cfg_over):
        cfg = PRESETS[config].with_(**cfg_over)
        registry = build_registry(db, cfg.allow_anon_server)
        state = sqlite3.connect(":memory:", check_same_thread=False)
        approvals = ApprovalService(state, None if async_approvals else reviewer)
        sink = MemorySink()
        rt = Runtime(cfg, registry, CheckpointStore(state), approvals, Tracer([sink]), CustomerMemory(state), llm_factory=llm_factory)
        rt.sink, rt.state = sink, state
        return rt
    return _make


def rows(db, sql, *a):
    c = connect(db)
    try:
        return [dict(r) for r in c.execute(sql, a).fetchall()]
    finally:
        c.close()


@pytest.fixture
def q(db):
    return lambda sql, *a: rows(db, sql, *a)
