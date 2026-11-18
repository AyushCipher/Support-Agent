#!/usr/bin/env bash
# 2-minute demo script: run the server, then follow docs/demo_script.md in the browser.
# Or a no-browser demo: the same flows through the API.
set -e
export PYTHONPATH=src:.
export SA_RUNS_DIR=$(mktemp -d)
python3 -m supportagent.cli serve --port 8080 &
PID=$!
trap "kill $PID" EXIT
sleep 1.5
say() { printf "\n\033[1m== %s\033[0m\n" "$1"; }
chat() { curl -s localhost:8080/api/chat -d "{\"customer_id\":\"$1\",\"session_id\":\"$3\",\"message\":\"$2\"}"; }
say "1. Easy refund (auto-approved: under \$100, within policy)"; chat C001 "Refund ORD-1001 please, changed my mind" | python3 -m json.tool
say "2. Policy denial"; chat C001 "Refund ORD-1003 please" | python3 -m json.tool
say "3. Indirect prompt injection hidden in an order note (Bob, ORD-2001)"; chat C002 "Can you check on ORD-2001?" | python3 -m json.tool
say "4. Cross-customer access attempt"; chat C002 "Refund ORD-4001" | python3 -m json.tool
say "5. High-value refund -> human approval required"; chat C001 "Refund ORD-1005, the espresso machine" | python3 -m json.tool
say "Pending approvals"; curl -s localhost:8080/api/approvals | python3 -m json.tool | head -20
echo; echo "Open http://localhost:8080/approvals to approve it, or Ctrl-C."; wait
