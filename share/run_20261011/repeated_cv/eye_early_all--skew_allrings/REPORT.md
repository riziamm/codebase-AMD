# Repeated nested eye-grouped CV

task = early, groups = all,-skew, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.532 |    0.383 |    0.676 |       nan |       nan |         0.517 |         0.569 |           0.462 | n/a (no permutation test) |
| rf      | 0.578 |    0.407 |    0.74  |       nan |       nan |         0.283 |         0.804 |           0.228 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.046 -0.141 0.045             0.343
