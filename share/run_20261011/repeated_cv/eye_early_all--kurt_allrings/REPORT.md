# Repeated nested eye-grouped CV

task = early, groups = all,-kurt, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.584 |    0.433 |    0.72  |       nan |       nan |         0.55  |         0.633 |           0.434 | n/a (no permutation test) |
| rf      | 0.58  |    0.417 |    0.731 |       nan |       nan |         0.273 |         0.79  |           0.233 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.004 -0.06 0.069             0.892
