#!/bin/bash
# ONE command to sync with GitHub (replaces "git stash / pull / stash pop", which caused the conflicts).
#   bash scripts/sync.sh                 # commit your edits, pull mine, push yours
#   bash scripts/sync.sh "my message"    # same, with your own commit message
# Data are never committed (.gitignore blocks csv/pkl/npy/models/reruns_*; only share/ is allowed).
set -e
BR=eye-grouped-rerun
if [ -n "$(git ls-files -u)" ]; then
  echo "STOP: unresolved conflict in:"; git ls-files -u | awk '{print "  "$4}' | sort -u
  echo "Send me this output (or keep the GitHub version with: git checkout origin/$BR -- <file> && git add <file>)"
  exit 1
fi
[ "$(git branch --show-current)" = "$BR" ] || { echo "STOP: you are on branch $(git branch --show-current), expected $BR"; exit 1; }
git add -u                                   # edits/deletions of tracked files
git add configs/ share/ 2>/dev/null || true  # new configs and shared (aggregate) results
if ! git diff --cached --quiet; then
  echo "Committing:"; git diff --cached --name-status | sed 's/^/  /'
  git commit -q -m "${1:-sync from HPC $(date +%F_%H%M)}"
fi
git pull --rebase origin $BR
git push origin $BR
echo "SYNCED: $(git log --oneline -1)"
