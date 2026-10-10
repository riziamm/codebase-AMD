# Repeated nested eye-grouped CV

task = early, groups = all,-Del, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.499 |    0.365 |     0.63 |       nan |       nan |         0.443 |         0.548 |           0.449 | n/a (no permutation test) |
| rf      | 0.537 |    0.385 |     0.68 |       nan |       nan |         0.25  |         0.794 |           0.222 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.038 -0.116 0.046             0.369
