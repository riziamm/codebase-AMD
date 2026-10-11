# Paper 2 runbook (branch `AMD-P2`)

**Start every session:** `source activate.sh` (own folder `../AMD-P2` via git worktree, conda env `ardes`,
paths to `data/export` in the main clone). Your other branch in the main clone is never touched.

Plan: `docs/ANALYSIS_PLAN_PAPER2.md`. Code: `src/p2_*.py`, `scripts/p2_*.sh`. Test: `bash tests/run_synthetic.sh`.
Outputs on the HPC: `reruns_paper2/` (git-ignored). Shared aggregates: `share/paper2_run_<date>/`.

| Step | Command | Needs sign-off | Time |
|---|---|---|---|
| 0 | `bash tests/run_synthetic.sh` (pipeline test, synthetic data) | no | ~5 min |
| S0 | `nohup bash scripts/p2_0_baseline.sh > reruns_paper2/logs/p2_0.log 2>&1 &` | no | ~1 min |
| P1+P2 | `nohup bash scripts/p2_1_main.sh > reruns_paper2/logs/p2_1.log 2>&1 &` | **yes** | ~20-40 min |
| share | `bash scripts/p2_package.sh && bash scripts/sync.sh "p2 <step>"` | no | 1 min |

Env overrides: `EXPORT` (default data/export), `RAW` (demographics_raw.csv), `BANDS44` (44-region band map).

## Run log (fill in after every real-data run)
| Date | Step | Commit (`run_info.json`) | Plan SHA-256 (first 8) | Share folder | Notes |
|---|---|---|---|---|---|
| | | | | | |
