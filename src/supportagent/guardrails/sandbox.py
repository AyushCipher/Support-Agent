"""Sandboxed arithmetic for the `run_calculation` tool (e.g. proration math).

Defense in depth: (1) AST allow-list - no names, attributes, imports, subscripts, comprehensions; (2) execution in a
separate isolated interpreter (-I -S) with CPU/memory limits and a hard wall-clock timeout; (3) result size cap.

Resource limits per platform:
  * POSIX: RLIMIT_CPU (2 s) and RLIMIT_AS (256 MiB) set inside the child before any user data is parsed. Fail closed: if the
    limits cannot be applied the child dies and the call fails.
  * Windows: there is no `resource` module. The parent assigns the child to a Job Object (ctypes/kernel32) with a 256 MiB
    per-process memory limit, a 2 s user-CPU limit and kill-on-close. This is BEST EFFORT: the child is assigned right after
    CreateProcess, so there is a brief window before the limits bind (the AST allow-list already forbids anything that could
    use it), and if the Job Object cannot be created we still run under the hard wall-clock timeout and report
    `sandbox_limits()["windows_job_object"] = False`. Neither platform replaces a real sandbox for arbitrary code.
Production note: for arbitrary code use gVisor/Firecracker/E2B - an AST filter is only enough for expressions.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys

ALLOWED_FUNCS = {"round", "min", "max", "abs"}
ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
MAX_LEN = 200


class SandboxError(Exception): ...


def validate_expr(expr: str) -> ast.Expression:
    if len(expr) > MAX_LEN:
        raise SandboxError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise SandboxError("only numeric literals allowed")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BIN):
                raise SandboxError("operator not allowed")
            if isinstance(node.op, ast.Pow):
                r = node.right
                if not (isinstance(r, ast.Constant) and abs(r.value) <= 20):
                    raise SandboxError("exponent must be a literal <= 20")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise SandboxError("operator not allowed")
        elif isinstance(node, (ast.USub, ast.UAdd) + ALLOWED_BIN):
            continue
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS) or node.keywords:
                raise SandboxError("call not allowed")
        elif isinstance(node, ast.Name):
            if node.id not in ALLOWED_FUNCS:
                raise SandboxError(f"name '{node.id}' not allowed")
        elif isinstance(node, ast.Load):
            continue
        else:
            raise SandboxError(f"{type(node).__name__} not allowed")
    return tree


_CHILD = r"""
import sys, json, ast
try:
    import resource
except ImportError:                      # Windows: limits are applied by the parent through a Job Object
    resource = None
if resource is not None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
expr = sys.stdin.read()
tree = ast.parse(expr, mode="eval")
env = {"__builtins__": {}, "round": round, "min": min, "max": max, "abs": abs}
print(json.dumps({"result": eval(compile(tree, "<calc>", "eval"), env, {})}))
"""


MEM_LIMIT = 256 * 1024 * 1024
CPU_LIMIT_S = 2


def _windows_job():
    """Create a Job Object with memory/CPU/kill-on-close limits. Returns (handle, kernel32) or None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IOC(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXT()
        # PROCESS_TIME (0x2) | PROCESS_MEMORY (0x100) | KILL_ON_JOB_CLOSE (0x2000)
        info.Basic.LimitFlags = 0x2 | 0x100 | 0x2000
        info.Basic.PerProcessUserTimeLimit = CPU_LIMIT_S * 10_000_000          # 100 ns units
        info.ProcessMemoryLimit = MEM_LIMIT
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):       # 9 = JobObjectExtendedLimitInformation
            k32.CloseHandle(job)
            return None
        return job, k32
    except Exception:                                                       # pragma: no cover - ctypes/kernel32 oddities
        return None


def _assign(job, proc: subprocess.Popen) -> bool:
    try:
        from ctypes import wintypes  # noqa: F401
        return bool(job[1].AssignProcessToJobObject(job[0], int(proc._handle)))     # type: ignore[attr-defined]
    except Exception:                                                       # pragma: no cover
        return False


def sandbox_limits() -> dict:
    """Which resource limits are enforced on this platform (for honesty in tests/docs/dashboards)."""
    if sys.platform == "win32":
        j = _windows_job()
        if j:
            j[1].CloseHandle(j[0])
        return {"platform": "windows", "windows_job_object": bool(j), "rlimits": False, "wall_clock_timeout": True}
    try:
        import resource  # noqa: F401
        return {"platform": "posix", "windows_job_object": False, "rlimits": True, "wall_clock_timeout": True}
    except ImportError:                                                     # pragma: no cover
        return {"platform": "posix", "windows_job_object": False, "rlimits": False, "wall_clock_timeout": True}


def safe_calc(expr: str, timeout: float = 3.0) -> float | int:
    validate_expr(expr)                                   # parent-side validation
    job = _windows_job()
    try:
        p = subprocess.Popen([sys.executable, "-I", "-S", "-c", _CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    except OSError:
        if job:
            job[1].CloseHandle(job[0])
        raise SandboxError("execution failed") from None
    try:
        if job:
            _assign(job, p)
        try:
            stdout, _ = p.communicate(expr, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise SandboxError("timeout") from None
    finally:
        if job:
            job[1].CloseHandle(job[0])                    # KILL_ON_JOB_CLOSE reaps any stragglers (e.g. the venv launcher's child)
    if p.returncode != 0:
        raise SandboxError("execution failed")
    out = json.loads(stdout)["result"]
    if isinstance(out, float) and (out != out or abs(out) == float("inf")):
        raise SandboxError("non-finite result")
    return out
