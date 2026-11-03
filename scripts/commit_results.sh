#!/usr/bin/env bash
# Commit result files from a long-running job without losing them to a moved main branch.
# usage: scripts/commit_results.sh "<commit message>" <path> [<path> ...]
# Copies the given paths aside, resets to the latest origin/<branch>, puts them back, regenerates the
# derived files (metrics, tables, charts, model card) and pushes; retries a few times if main moves again.
set -uo pipefail
MSG=${1:?message}; shift
BRANCH=${GITHUB_REF_NAME:-main}
STASH=$(mktemp -d)
for p in "$@"; do
  if [ -e "$p" ]; then mkdir -p "$STASH/$(dirname "$p")"; cp -r "$p" "$STASH/$(dirname "$p")/"; fi
done
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for i in 1 2 3 4 5; do
  git fetch -q origin "$BRANCH" && git reset -q --hard "origin/$BRANCH"
  cp -r "$STASH"/. . 2>/dev/null || true
  jobx eval > /dev/null && jobx results-table > /dev/null && jobx charts > /dev/null || true
  if [ -f results/training/qwen2.5-0.5b-lora-cpu/status.json ]; then jobx model-card > /dev/null || true; fi
  git add -A results README.md MODEL_CARD.md 2>/dev/null; git add -A "$@" 2>/dev/null
  if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
  git commit -q -m "$MSG"
  if git push -q origin "HEAD:$BRANCH"; then echo "pushed: $MSG"; exit 0; fi
  sleep $((i * 20))
done
echo "::error::could not push after 5 attempts"
exit 1
