#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script so it works anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
DEV_BRANCH="dev"
MAIN_BRANCH="main"

usage() {
  cat <<'USAGE'
Usage: sync-dev.sh

Syncs the dev branch with changes from main.
This is useful when main has received commits (e.g., from merged PRs)
that dev doesn't have yet.

Uses merge strategy to integrate changes (creates a merge commit).
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"

# Ensure clean working tree
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree has uncommitted changes. Please commit or stash them first." >&2
  exit 1
fi

# Fetch latest from origin
printf '\n==> Fetching from origin\n'
git fetch origin

# Check current status
AHEAD=$(git rev-list --count "origin/$MAIN_BRANCH..$DEV_BRANCH" 2>/dev/null || echo "0")
BEHIND=$(git rev-list --count "$DEV_BRANCH..origin/$MAIN_BRANCH" 2>/dev/null || echo "0")

printf '\n📊 %s is %s commits ahead, %s commits behind %s\n' "$DEV_BRANCH" "$AHEAD" "$BEHIND" "$MAIN_BRANCH"

if [[ "$BEHIND" -eq 0 ]]; then
  echo "✅ $DEV_BRANCH is already up to date with $MAIN_BRANCH"
  exit 0
fi

# Checkout dev branch
printf '\n==> Switching to %s\n' "$DEV_BRANCH"
git checkout "$DEV_BRANCH"
git pull --ff-only origin "$DEV_BRANCH" 2>/dev/null || true

# Merge main into dev
printf '\n==> Merging %s into %s\n' "$MAIN_BRANCH" "$DEV_BRANCH"
if ! git merge "origin/$MAIN_BRANCH" --no-edit -m "chore: sync $DEV_BRANCH with $MAIN_BRANCH"; then
  echo ""
  echo "❌ Merge conflict! Resolve conflicts manually, then:"
  echo "   git add ."
  echo "   git commit"
  echo "   git push origin $DEV_BRANCH"
  exit 1
fi

# Push the merge
printf '\n==> Pushing %s\n' "$DEV_BRANCH"
git push origin "$DEV_BRANCH"

printf '\n✅ %s synced with %s (%s commits integrated)\n' "$DEV_BRANCH" "$MAIN_BRANCH" "$BEHIND"
