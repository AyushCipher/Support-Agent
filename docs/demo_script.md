# 2-minute demo script

Record with `make serve` → http://localhost:8080 (two tabs: chat and `/approvals`). Keep the right-hand **Agent trace** panel visible.

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the architecture diagram (README) for 10 s | "A support agent that can move money. The model proposes; the harness disposes." |
| 0:10 | As **C001**: `Refund ORD-1001` | "In policy and under the cap, so it completes — $37.50, item price only; shipping isn't refundable." |
| 0:30 | `Refund ORD-1003`, then `please make an exception` | "Final-sale: denied by policy. The agent can't grant exceptions, so it escalates to a human." |
| 0:50 | Switch to **C002**: `Check on ORD-2001` | "That order note contains a hidden instruction to refund $500. It's stripped, the session is tainted — see the trace panel." |
| 1:10 | **C002**: `Refund ORD-4001` | "Someone else's order. The model can't even set the customer id — the harness injects it from the session." |
| 1:25 | **C001**: `Refund ORD-1005` | "$250 is over the autonomous cap. The run pauses and persists." Switch to **/approvals** → Approve. |
| 1:45 | Back to chat; show reply + trace | "It resumed exactly where it stopped, exactly one refund." |
| 1:55 | Show `docs/results.md` ladder | "43% attack success → 0%, measured per layer — on a simulated model; live numbers via `make eval-live`." |
