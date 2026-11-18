"""Terminal labeler for the pack (alternative to label.html). Resumable: skips items this rater already labeled.

    python -m evals.labeling.label --rater ayush            # writes evals/labeling/labels.jsonl (appends)
Keys: p = pass, f = fail, s = skip, q = quit. After the reply label you may give outcome_ok (y/n/u, Enter = skip) and a note.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
PACK, LABELS = HERE / "pack.jsonl", HERE / "labels.jsonl"


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def show(it: dict, i: int, n: int) -> str:
    out = [f"\n{'=' * 78}\nItem {i}/{n}  {it['item_id']}  category={it['category']}  customer={it['customer_id']}  "
           f"final_status={it['final_status']}" + (f"  attack={it['attack']}" if it.get("attack") else "")]
    for m in it["transcript"]:
        out.append(textwrap.fill(m["text"], 100, initial_indent=f"  {m['role']:>8}: ", subsequent_indent=" " * 12))
    out.append("  -- ground truth --")
    out.append("  tool calls:       " + ", ".join(map(str, it.get("tool_calls", []))))
    out.append("  end_state_actual: " + json.dumps(it.get("end_state", {})))
    out.append("  policy_expected:  " + json.dumps(it.get("policy_expected", {})))
    return "\n".join(out)


def main(argv: list[str] | None = None, ask=input) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.labeling.label")
    ap.add_argument("--rater", required=True, help="your name; two raters on the same items give inter-rater kappa")
    a = ap.parse_args(argv)
    pack = read_jsonl(PACK)
    if not pack:
        print("No pack yet: run `python -m evals.labeling.build_pack` first.")
        return 1
    done = {r["item_id"] for r in read_jsonl(LABELS) if r.get("rater") == a.rater}
    todo = [it for it in pack if it["item_id"] not in done]
    print(f"{len(done)} already labeled by {a.rater}; {len(todo)} to go. Rubric: evals/labeling/RUBRIC.md")
    with LABELS.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, len(done) + 1):
            print(show(it, k, len(pack)))
            ans = ""
            while ans not in ("p", "f", "s", "q"):
                ans = ask("Reply label per rubric R1-R5: [p]ass / [f]ail / [s]kip / [q]uit > ").strip().lower()[:1]
            if ans == "q":
                break
            if ans == "s":
                continue
            row = {"item_id": it["item_id"], "rater": a.rater, "label": "pass" if ans == "p" else "fail"}
            o = ask("Optional O1 outcome handled correctly? [y]es/[n]o/[u]nsure/Enter=skip > ").strip().lower()[:1]
            if o in ("y", "n", "u"):
                row["outcome_ok"] = {"y": "yes", "n": "no", "u": "unsure"}[o]
            note = ask("Optional note > ").strip()
            if note:
                row["note"] = note[:500]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
