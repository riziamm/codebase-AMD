# Repeated nested eye-grouped CV

task = early, groups = all,-iqr, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.59  |    0.428 |    0.738 |       nan |       nan |         0.57  |         0.59  |           0.468 | n/a (no permutation test) |
| rf      | 0.606 |    0.431 |    0.767 |       nan |       nan |         0.367 |         0.812 |           0.254 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.016 -0.078 0.042              0.61
