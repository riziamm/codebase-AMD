#!/bin/bash
# PILOT: quick end-to-end check on the real data (~15-30 min). Nothing here goes into the paper.
# Run:  nohup bash scripts/0_pilot.sh > reruns_pilot.log 2>&1 &
# Then: tail -n 40 reruns_pilot.log      (look at the CHECKLIST at the end)
set -u
export ROOT=reruns_pilot
rm -rf "$ROOT"; mkdir -p "$ROOT"/logs
[ -f reruns_eyegrouped/split/train_mpod.csv ] || { echo "ERROR: run src/make_split.py first"; exit 1; }

cat > "$ROOT"/pilot_ml.json <<'J'
[{"_name":"pilot_all9","normalization":"standard","sampling_method":"smote","is_binary":true,"preserve_zones":true,"sort_features":"none","tune_hyperparams":false,"analyze_shap":true,"transform_features":false},
 {"_name":"pilot_lowval","normalization":"standard","sampling_method":"smote","is_binary":true,"preserve_zones":true,"sort_features":"none","tune_hyperparams":false,"analyze_shap":true,"transform_features":false,"feature_indices":[2,5,7]}]
J
echo "== ML"
python -m src.main --mode batch --data_path reruns_eyegrouped/split/train_mpod.csv \
  --config_file "$ROOT"/pilot_ml.json --report_dir "$ROOT"/ml_batches > "$ROOT"/logs/ml.log 2>&1
echo "== holdout + ML eval"
bash scripts/1_create_holdout.sh > "$ROOT"/logs/holdout.log 2>&1
bash scripts/2_eval_ml_holdout.sh > "$ROOT"/logs/ml_eval.log 2>&1
echo "== DL (2 epochs, no tuning)"
python dl_pipeline_gen.py --dataset mpod --data_path reruns_eyegrouped/split/train_mpod.csv \
  --test_data_path reruns_eyegrouped/split/test_mpod.csv --report_base_dir "$ROOT"/dl \
  --experiment_name EG_bin_pilot --seed 42 --split_seed 42 --use_gpu \
  --models_to_run CNN CNNTransformer_parallel --epochs_training 2 --shap_num_samples 5 > "$ROOT"/logs/dl.log 2>&1
echo "== collect + evaluate"
NBOOT=200 bash scripts/4_collect_evaluate.sh > "$ROOT"/logs/eval.log 2>&1

ok () { if eval "$2"; then echo "  PASS  $1"; else echo "  FAIL  $1"; fi; }
echo; echo "=========== CHECKLIST ==========="
ok "ML eye-grouped split"        "grep -q 'no shared eyes' $ROOT/logs/ml.log"
ok "ML eye-grouped CV"           "! grep -q 'tune_hyperparams\": true' $ROOT/pilot_ml.json || grep -q 'eye-grouped' $ROOT/logs/ml.log"
ok "ML feature names (std,skew,Del)" "grep -q \"\\['std', 'skew', 'Del'\\]\" $ROOT/logs/ml.log"
ok "ML all models saved (8 per exp)" "[ \$(ls $ROOT/ml_batches/*/experiment_1/models/*.pkl 2>/dev/null | wc -l) -ge 8 ]"
ok "ML AUC not zero in summary"  "python -c \"import json,glob,math;v=[m['roc_auc_score'] for f in glob.glob('$ROOT/ml_batches/*/*/metrics/all_model_metrics.json') for m in json.load(open(f))['models'].values()];assert v and all(x>0 for x in v if not math.isnan(x))\""
ok "ML SHAP figures (RF + LR)"   "ls $ROOT/ml_batches/*/experiment_1/figures | grep -q 'Random Forest_shap' && ls $ROOT/ml_batches/*/experiment_1/figures | grep -q 'Logistic Regression_shap'"
ok "ML holdout predictions"      "[ \$(ls $ROOT/ml_eval/predictions/*.csv 2>/dev/null | wc -l) -ge 16 ]"
ok "DL uses frozen split files"  "grep -q 'DL holdout split (frozen files): OK' $ROOT/logs/dl.log"
ok "DL holdout predictions"      "[ \$(ls $ROOT/dl/EG_bin_pilot/post_analysis/metrics/*holdout_eval_classification_report.json 2>/dev/null | wc -l) -ge 2 ]"
ok "DL SHAP circular maps"       "find $ROOT/dl -name '*circ*.png' | grep -q ."
ok "Final table (AUC+sens+CI)"   "[ -s $ROOT/results_holdout.csv ]"
echo "================================="
echo "If any FAIL: send me the matching log in $ROOT/logs/"
