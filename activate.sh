#!/bin/bash
# Start a Paper 2 work session:   source activate.sh      (must be SOURCED, not run with bash)
#
# Paper 2 lives in its OWN folder (git worktree) next to your main clone, so the other branch/experiment in the
# main clone is never switched under a running job:
#   <parent>/codebase-AMD   main clone  (your other branch; data/export lives here)
#   <parent>/AMD-P2         Paper 2 worktree (branch AMD-P2; outputs in reruns_paper2/)
# First call creates the worktree (once). Every call: cd there, conda activate ardes, set paths/GPU, sync status.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Use:  source activate.sh   (running it with bash cannot activate conda in your shell)"; exit 1
fi

_p2_main() {
  local BR=AMD-P2 ENV=ardes here common MAIN WT
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  common="$(cd "$here" && git rev-parse --git-common-dir 2>/dev/null)" \
    || { echo "STOP: $here is not inside the git repo"; return 1; }
  case "$common" in /*) ;; *) common="$(cd "$here/$common" && pwd)";; esac   # old git returns a relative path
  MAIN="$(dirname "$common")"                      # main clone (owner of .git)
  WT="$(dirname "$MAIN")/$BR"

  # ---- one-time: create the worktree
  if [ ! -d "$WT" ]; then
    if [ "$(git -C "$MAIN" branch --show-current)" = "$BR" ]; then
      if [ -n "$(git -C "$MAIN" status --porcelain --untracked-files=no)" ]; then
        echo "STOP: main clone is on $BR with uncommitted changes. Commit them (bash scripts/sync.sh), then re-run."
        return 1
      fi
      echo "Main clone is on $BR; switching it back to your previous branch so $BR can get its own folder:"
      git -C "$MAIN" checkout - || { echo "STOP: could not switch back; run: git -C $MAIN checkout <your-branch>"; return 1; }
    fi
    git -C "$MAIN" fetch -q origin "$BR" || { echo "STOP: cannot fetch $BR"; return 1; }
    git -C "$MAIN" show-ref -q --verify "refs/heads/$BR" || git -C "$MAIN" branch -q --track "$BR" "origin/$BR"
    git -C "$MAIN" worktree add -q "$WT" "$BR" || { echo "STOP: worktree add failed"; return 1; }
    echo "Created Paper 2 folder: $WT"
  fi

  cd "$WT" || return 1
  [ "$(git branch --show-current)" = "$BR" ] || { echo "STOP: $WT is not on $BR"; return 1; }

  # ---- conda
  if ! type conda >/dev/null 2>&1 || ! conda activate "$ENV" 2>/dev/null; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
      [ -f "$c/etc/profile.d/conda.sh" ] && . "$c/etc/profile.d/conda.sh" && break
    done
    conda activate "$ENV" || { echo "STOP: conda activate $ENV failed"; return 1; }
  fi

  # ---- paths (data stay in the main clone; never copied into the worktree) and GPU
  export EXPORT="$MAIN/data/export"
  unset RAW
  for f in "$MAIN/data/demographics_raw.csv" "$MAIN/demographics_raw.csv" "$MAIN/data/export/demographics_raw.csv"; do
    [ -f "$f" ] && export RAW="$f" && break
  done
  unset BANDS44
  for f in "$MAIN/data/export/ofa44_layout.csv" "$MAIN/data/ofa44_layout.csv" "$MAIN/ofa44_layout.csv"; do
    [ -f "$f" ] && export BANDS44="$f" && break
  done
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # RTX A6000
  mkdir -p reruns_paper2/logs

  # ---- sync status (fast-forward only: never rewrites your work)
  git fetch -q origin "$BR" 2>/dev/null
  if [ -z "$(git status --porcelain --untracked-files=no)" ]; then
    local before; before="$(git rev-parse HEAD)"
    git merge -q --ff-only "origin/$BR" 2>/dev/null || echo "NOTE: local and GitHub differ -> run: bash scripts/sync.sh"
    if [ -z "${_P2_RELOADED:-}" ] && ! git diff --quiet "$before" HEAD -- activate.sh; then
      echo "activate.sh was updated by the pull -> reloading it"
      export _P2_RELOADED=1; source "$WT/activate.sh"; unset _P2_RELOADED; return
    fi
  else
    echo "NOTE: you have uncommitted changes -> bash scripts/sync.sh when ready"
  fi

  echo "-----------------------------------------------------------------"
  echo " Paper 2 ready    folder : $WT"
  echo "                  branch : $(git branch --show-current) @ $(git log --oneline -1 | cut -c1-60)"
  echo "                  python : $(python -c 'import sys;print(sys.version.split()[0])') (conda: $CONDA_DEFAULT_ENV)"
  echo "                  EXPORT : $EXPORT $( [ -f "$EXPORT/demographics.csv" ] && echo '[demographics OK]' || echo '[demographics.csv MISSING]')"
  echo "                  RAW    : ${RAW:-not found (S0 visit counts will be skipped)}"
  echo "                  LAYOUT : ${BANDS44:-ofa44_layout.csv not found (44-region rings unavailable)}"
  echo "                  plan   : $(grep -m1 -o '^\*\*Status:[^*]*' docs/ANALYSIS_PLAN_PAPER2.md | tr -d '*')"
  echo "-----------------------------------------------------------------"
}
_p2_main; unset -f _p2_main
