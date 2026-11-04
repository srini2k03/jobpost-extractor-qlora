#!/usr/bin/env bash
# Fetch a file produced by an earlier workflow run, without depending on GitHub Releases
# (the workflow token may not be allowed to create releases in this repository).
#
#   scripts/fetch_artifact.sh <release-tag> <asset-glob> <workflow-file> <artifact-name> <dest-dir>
#
# Tries the release asset first, then the named artifact of the newest run of <workflow-file>
# that still has it (artifacts are kept for 30 days). Exits 1 if neither has it.
set -euo pipefail
TAG=$1 GLOB=$2 WF=$3 ART=$4 DEST=$5
mkdir -p "$DEST"
if gh release view "$TAG" > /dev/null 2>&1 && gh release download "$TAG" -p "$GLOB" -D "$DEST" --clobber 2>/dev/null; then
  echo "fetched $GLOB from release $TAG"; exit 0
fi
for id in $(gh run list --workflow "$WF" --limit 30 --json databaseId,status \
              --jq '.[] | select(.status=="completed") | .databaseId'); do
  [ "$id" = "${GITHUB_RUN_ID:-}" ] && continue
  if gh run download "$id" -n "$ART" -D "$DEST" 2>/dev/null; then
    echo "fetched artifact $ART from $WF run $id"; exit 0
  fi
done
echo "not found: release $TAG ($GLOB) or artifact $ART of $WF" >&2
exit 1
