#!/usr/bin/env bash
# Commit only the explicitly supplied state/archive paths. Never force-push.
set -euo pipefail
message="$1"
shift
git add -A -- "$@"
if git diff --cached --quiet; then
  echo "No digest state or archive changes to commit."
  exit 0
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git commit -m "$message"
for attempt in 1 2 3; do
  if git push origin HEAD:main; then
    exit 0
  fi
  if [ "$attempt" -eq 3 ]; then
    exit 1
  fi
  # Preserve unrelated user commits; conflicting edits fail safely.
  git pull --rebase origin main
  sleep 3
done
