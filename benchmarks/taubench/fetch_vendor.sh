#!/usr/bin/env bash
# Fetch tau-bench (Sierra, MIT license) at a pinned commit into vendor/ (git-ignored, treated as untrusted third-party data).
# Nothing is installed from it: the adapter only imports tau_bench/ (pure Python + pydantic). See LICENSE-tau-bench.txt.
set -euo pipefail
COMMIT=59a200c6d575d595120f1cb70fea53cef0632f6b
DIR="$(cd "$(dirname "$0")" && pwd)/vendor/tau-bench"
if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$(dirname "$DIR")"
  git clone --quiet https://github.com/sierra-research/tau-bench.git "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$COMMIT" || true
git -C "$DIR" checkout --quiet "$COMMIT"
echo "tau-bench at $(git -C "$DIR" rev-parse HEAD)"
