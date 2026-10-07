#!/bin/bash
# Step C: all DL reruns, one after another.  Run with nohup (see RERUN.md):
#   nohup bash scripts/3_run_dl.sh > $ROOT/logs/dl_all.log 2>&1 &
# Re-running skips any experiment whose folder already has results.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID                          # index 0 = same GPU as nvidia-smi (RTX A6000)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"      # one GPU for every DL run; Blackwell (sm_120) is NOT supported by this PyTorch
python -c "import torch;print('DL GPU:', torch.cuda.get_device_name(0))" 2>&1 | grep "DL GPU"
ROOT="${ROOT:-reruns_eyegrouped}"   # output root (pilot uses ROOT=reruns_pilot)

# ---- EDIT: copy these from the ORIGINAL hybrid run's experiment_config.json ----
EPOCHS_TRAIN=30
EPOCHS_TUNE=30
CV_TRAIN=3
CV_TUNE=3
BATCH=16
# --------------------------------------------------------------------------------

# [frozen split] same development/holdout files as the ML pipeline (made by src/make_split.py)
DEV="reruns_eyegrouped/split/train_mpod.csv"
HOLDOUT="reruns_eyegrouped/split/test_mpod.csv"
OUT="$ROOT/dl"
COMMON="--dataset mpod --data_path $DEV --test_data_path $HOLDOUT --report_base_dir $OUT --use_gpu --tune_hyperparameters \
 --split_seed 42 --epochs_training $EPOCHS_TRAIN --epochs_tuning $EPOCHS_TUNE \
 --cv_splits_training $CV_TRAIN --cv_splits_tuning $CV_TUNE --batch_size $BATCH"

run () {   # run <name> <extra args...>
  local NAME=$1; shift
  if [ -f "$OUT/$NAME/post_analysis/metrics/CNNTransformer_parallel_final_best_holdout_eval_classification_report.json" ]; then
    echo "== skip (done): $NAME"; return
  fi
  echo "== START $NAME  $(date)"
  python dl_pipeline_gen.py $COMMON --experiment_name "$NAME" "$@" || echo "!! FAILED: $NAME"
  echo "== END   $NAME  $(date)"
}

# 1) main comparison, all 4 models, seed 42  (Table A3, Fig A2; SHAP Figs 4, 6)
run EG_bin_all_tuned_s42 --seed 42

# 2) hybrid stability, seeds 0-4, same split  (report mean +/- SD)
for S in 0 1 2 3 4; do
  run EG_bin_hybrid_tuned_s$S --seed $S --models_to_run CNNTransformer_parallel
done

# 3) exclusion experiments  (Fig 5, A4-A6). CSV order: 0 mean 1 median 2 std 3 iqr 4 idr 5 skew 6 kurt 7 Del 8 Amp
run EG_bin_hybrid_no_mean       --seed 42 --models_to_run CNNTransformer_parallel --select_features 1 2 3 4 5 6 7 8
run EG_bin_hybrid_no_mean_skew  --seed 42 --models_to_run CNNTransformer_parallel --select_features 1 2 3 4 6 7 8
run EG_bin_hybrid_no_mean_amp   --seed 42 --models_to_run CNNTransformer_parallel --select_features 1 2 3 4 5 6 7

echo "ALL DL RUNS FINISHED $(date)"
