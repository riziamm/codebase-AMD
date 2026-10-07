# Eye-grouped rerun (branch `eye-grouped-rerun`)

The original code is preserved on `main` and on the tag `v1-original-submission`. This branch changes **one thing scientifically**: every split groups by eye, so both repeat sessions of an eye always land in the same partition and the same CV fold. Everything else (models, grids, preprocessing, SHAP) is unchanged.

## What changed

| File | Change |
|---|---|
| `src/eye_split.py` | **new.** Grouped drop-ins: `grouped_train_test_split`, `grouped_cv`, `EyeGroupedKFold` |
| `src/make_split.py` | **new.** Creates the single frozen holdout and writes `train_mpod.csv` / `test_mpod.csv` (replaces `--mode create_test_set`) |
| `src/collect_preds.py` | **new.** Gathers DL and ML holdout predictions (y_true, y_prob, y_pred only) |
| `src/evaluate_all.py` | **new.** Identical metrics for every model: AUC, sensitivity/specificity, PPV/NPV, balanced accuracy, and three F1 definitions; eye-clustered bootstrap CIs; paired ΔAUC |
| `src/training_pipeline.py`, `src/core_logic.py` | ML internal validation split, grid-search CV and 10-fold CV summary are grouped by eye |
| `src/evaluation.py` | Syntax fix (stray line); exports per-row holdout predictions |
| `src/core_logic.py` | Import fix (`report_generation` → `src/reporting/utils`) |
| `dl_pipeline_gen.py` | Holdout and internal splits grouped by eye (the holdout equals the `make_split.py` holdout); grouped CV in training, tuning and multi-GPU workers; `--split_seed` (default 42) separate from `--seed`; `--no_eye_grouping` reproduces the old split; removed the `verbose` arg from `ReduceLROnPlateau` (newer PyTorch rejects it) |

The whole chain was tested end to end on a synthetic file with the identical schema (116 × 184, same column names, random values).

Not changed: the learning-curve diagnostic plots (`plot_learning_curve`) still use row-level CV. They are diagnostic figures only and are not reported in the manuscript.

## 0. Setup on the HPC

```bash
cd ~/codebase-AMD                     # or: git clone https://github.com/riziamm/codebase-AMD.git
git fetch origin
git checkout eye-grouped-rerun
conda activate ardes
mkdir -p reruns_eyegrouped/{env,logs}
pip freeze > reruns_eyegrouped/env/pip_freeze.txt
git rev-parse HEAD > reruns_eyegrouped/env/commit.txt
# put mpod.csv in data/  (data/*.csv is git-ignored)
```

Run anything longer than a few minutes inside `tmux`, and log it:
`<command> 2>&1 | tee -a reruns_eyegrouped/logs/<name>_$(date +%Y%m%d_%H%M%S).log`

## 1. Frozen split (already done; repeat here only if needed)

```bash
python src/make_split.py --data data/mpod.csv --out reruns_eyegrouped/split --seed 42 --write_csvs
```

The manifest SHA-256 must be `effe59b09c4c52a5f3049c416b78303e9f917e2176af79f839fb007703a9d3a7`. Never regenerate the split with other settings.

## 2. ML (binary)

**Preferred:** reuse your *original* experiment definitions exactly. Every original batch folder contains `all_configurations.json`; the manuscript tables came from these batches:
`batch_20250428_142957_base_bin`, `batch_20250527_165720` (best ML), `batch_20250428_155932_fi` (feature indexing), `batch_20250428_162319_fs` and `batch_20250702_101156_fs_all` (feature sorting).

**Fallback:** if you can't find them, use `configs/rerun_tier1_ml_binary.json` (the Table 3 / Table A2 feature-group rows).

```bash
python -m src.main --mode batch \
  --data_path reruns_eyegrouped/split/train_mpod.csv \
  --config_file configs/rerun_tier1_ml_binary.json \
  --report_dir reruns_eyegrouped/ml_batches
```

Check the log for `[eye_split] ... OK, no shared eyes` and `Using eye-grouped 5-fold CV`.

Holdout, run once per batch:

```bash
# edit BATCH_DIRS in scripts/1_create_holdout.sh to the new batch folder(s), and set
# HOLDOUT_CSV="reruns_eyegrouped/split/test_mpod.csv"; then:
bash scripts/1_create_holdout.sh

# evaluate every saved model of every experiment on its own test_test.pkl
for EXP in reruns_eyegrouped/ml_batches/batch_*/experiment_*; do
  python -m src.main --mode evaluate \
    --model_paths $EXP/models/*.pkl \
    --eval_data_path $EXP/data/test_test.pkl \
    --report_dir reruns_eyegrouped/ml_eval
done
```

## 3. DL (binary)

Use the full `data/mpod.csv`, which must include the `Subject`/`Eye` columns. The DL code builds the identical eye-grouped holdout itself. Copy epochs, CV splits and batch size from the original run's `experiment_config.json`, so that the only change is the grouping.

```bash
# Main comparison, tuned (Table A3 / Fig A2); SHAP on this seed-42 run (Figs 4, 6)
python dl_pipeline_gen.py --dataset mpod --data_path data/mpod.csv \
  --experiment_name EG_bin_all_tuned_s42 --tune_hyperparameters --use_gpu \
  --seed 42 --split_seed 42 --report_base_dir reruns_eyegrouped/dl

# Stability of the hybrid: same procedure, model seeds 0-4, same split (report mean ± SD)
for S in 0 1 2 3 4; do
  python dl_pipeline_gen.py --dataset mpod --data_path data/mpod.csv \
    --experiment_name EG_bin_hybrid_tuned_s$S --models_to_run CNNTransformer_parallel \
    --tune_hyperparameters --use_gpu --seed $S --split_seed 42 --report_base_dir reruns_eyegrouped/dl
done

# Exclusion experiments (Fig 5, A4-A6). CSV feature order: 0 mean,1 median,2 std,3 iqr,4 idr,5 skew,6 kurt,7 Del,8 Amp
python dl_pipeline_gen.py --dataset mpod --data_path data/mpod.csv --models_to_run CNNTransformer_parallel \
  --tune_hyperparameters --use_gpu --seed 42 --report_base_dir reruns_eyegrouped/dl \
  --experiment_name EG_bin_hybrid_no_mean --select_features 1 2 3 4 5 6 7 8
python dl_pipeline_gen.py --dataset mpod --data_path data/mpod.csv --models_to_run CNNTransformer_parallel \
  --tune_hyperparameters --use_gpu --seed 42 --report_base_dir reruns_eyegrouped/dl \
  --experiment_name EG_bin_hybrid_no_mean_skew --select_features 1 2 3 4 6 7 8
python dl_pipeline_gen.py --dataset mpod --data_path data/mpod.csv --models_to_run CNNTransformer_parallel \
  --tune_hyperparameters --use_gpu --seed 42 --report_base_dir reruns_eyegrouped/dl \
  --experiment_name EG_bin_hybrid_no_mean_amp --select_features 1 2 3 4 5 6 7
```

**Multiclass (Table A4):** passing `--binary_classification` *switches multiclass ON* (the flag is store_false). For ML, use the same configs with `"is_binary": false`.

## 4. Evaluate (on the HPC)

```bash
python -m src.collect_preds \
  --dl_runs reruns_eyegrouped/dl/EG_bin_* \
  --ml_dirs reruns_eyegrouped/ml_eval \
  --out reruns_eyegrouped/preds_holdout
ls reruns_eyegrouped/preds_holdout        # pick names for --pairs below

python src/evaluate_all.py \
  --pred_dir reruns_eyegrouped/preds_holdout \
  --manifest reruns_eyegrouped/split/split_manifest.csv \
  --out reruns_eyegrouped/results_holdout.csv \
  --pairs "EG_bin_all_tuned_s42__CNNTransformer_parallel:<ML best RF file stem>" \
          "<ML best RF file stem>:<ML LR baseline file stem>"
```

`evaluate_all.py` checks that every prediction file's labels match the frozen holdout order; a mismatch stops with an error.

## 5. What to share (aggregates only)

**Share:**
- `results_holdout.csv` and `results_holdout_paired_dAUC.csv`;
- the selected hyperparameters (from the logs or each run's `experiment_config.json`);
- SHAP figures;
- the batch summary CSVs.

**Keep on the HPC:**
- the data CSVs, `split_manifest.csv`, prediction files, `.npy` files and models.
