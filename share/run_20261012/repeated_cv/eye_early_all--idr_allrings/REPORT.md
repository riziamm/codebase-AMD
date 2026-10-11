# Repeated nested eye-grouped CV

task = early, groups = all,-idr, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.579 |    0.419 |    0.728 |       nan |       nan |         0.58  |         0.587 |           0.474 | n/a (no permutation test) |
| rf      | 0.594 |    0.416 |    0.755 |       nan |       nan |         0.353 |         0.794 |           0.26  | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.015 -0.075 0.044             0.653
