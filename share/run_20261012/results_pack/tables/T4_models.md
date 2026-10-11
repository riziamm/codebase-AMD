### Models and configurations

| Model | Parameters | Tuned | Notes |
|---|---|---|---|
| Logistic regression | 181 | C in {0.001, 0.01, 0.1, 1} (inner 3-fold, AUC) | L2, class-balanced, liblinear |
| Random forest | - | max_depth in {3, None} x min_samples_leaf in {1, 3} | 300 trees, balanced_subsample |
| Zone-token CNN-Transformer | 3,265 | none (pre-specified) | 20 zone tokens x 9 features; zone + ring embeddings; 1 encoder layer (d=16, 2 heads); Conv1d over zones (16 ch); AdamW lr 3e-3, wd 1e-2, 80 epochs, batch 16, pos_weight = n_neg/n_pos |
| CNN-Transformer (original) | 602,993 | none (configuration of the original study) | parallel CNN + Transformer on a single 180-d token; d=16, 2 heads, 1 layer, dropout 0.3 |
