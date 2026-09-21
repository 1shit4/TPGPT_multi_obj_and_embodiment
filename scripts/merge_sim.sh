#!/usr/bin/env bash
# Bring the simulation's latest origin/main into the hardware branch.
#
# CLAUDE.md is resolved to the hardware branch's own copy (.gitattributes,
# merge=ours); the simulation's CLAUDE.md is then copied to docs/sim/SIM_CLAUDE.md
# so the two can still be compared. Any other conflict stops the merge for you to
# resolve by hand, as usual.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ "$(git branch --show-current)" != hardware ]]; then
  echo "run this on the hardware branch" >&2; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "commit or stash your changes first" >&2; exit 1
fi
git config merge.ours.driver true
git fetch origin
git merge --no-edit origin/main
git show origin/main:CLAUDE.md > docs/sim/SIM_CLAUDE.md
if ! git diff --quiet -- docs/sim/SIM_CLAUDE.md; then
  git add docs/sim/SIM_CLAUDE.md
  git commit -m "Refresh docs/sim/SIM_CLAUDE.md from origin/main"
fi
git log --oneline -3
