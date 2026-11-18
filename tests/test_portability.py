"""Sandbox portability: works on Windows (Job Object, best effort) and keeps POSIX rlimits."""
import subprocess
import sys

import pytest

from supportagent.guardrails import sandbox
from supportagent.guardrails.sandbox import SandboxError, safe_calc, sandbox_limits

WIN = sys.platform == "win32"


def test_calculation_works_on_this_platform():
    assert safe_calc("round(1000 * 0.15 / 3, 2)") == 50.0
    assert safe_calc("max(3, 7) + abs(-2)") == 9


def test_child_does_not_hard_depend_on_the_posix_resource_module():
    assert "import resource\nexcept ImportError" in sandbox._CHILD.replace("    ", "")  # guarded import, Windows-safe


def test_wall_clock_timeout_kills_a_runaway_child(monkeypatch):
    monkeypatch.setattr(sandbox, "_CHILD", "while True: pass")
    with pytest.raises(SandboxError, match="timeout"):
        safe_calc("1+1", timeout=1.0)


def test_failing_child_fails_closed(monkeypatch):
    monkeypatch.setattr(sandbox, "_CHILD", "import sys; sys.exit(3)")
    with pytest.raises(SandboxError, match="execution failed"):
        safe_calc("1+1")


def test_limits_report_is_honest_about_the_platform():
    lim = sandbox_limits()
    assert lim["wall_clock_timeout"] is True
    assert lim["platform"] == ("windows" if WIN else "posix")
    if not WIN:
        assert lim["rlimits"] is True                                                  # Linux/macOS must not be weaker than before


@pytest.mark.skipif(not WIN, reason="Windows Job Object")
def test_windows_job_object_enforces_memory_limit():
    job = sandbox._windows_job()
    if job is None:
        pytest.skip("Job Objects unavailable in this environment (documented best-effort)")
    p = subprocess.Popen([sys.executable, "-I", "-S", "-c", "x = bytearray(1 << 30)"], stderr=subprocess.PIPE, text=True)
    assert sandbox._assign(job, p)
    _, err = p.communicate(timeout=30)
    job[1].CloseHandle(job[0])
    assert p.returncode != 0 and "MemoryError" in err                                   # 1 GiB allocation refused under the 256 MiB limit


@pytest.mark.skipif(WIN, reason="POSIX rlimits")
def test_posix_rlimits_are_applied_in_the_child():
    code = sandbox._CHILD.replace("expr = sys.stdin.read()", "print(resource.getrlimit(resource.RLIMIT_CPU)[0], resource.getrlimit(resource.RLIMIT_AS)[0]); sys.exit(0)")
    out = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=30).stdout.split()
    assert out == ["2", str(256 * 1024 * 1024)]
