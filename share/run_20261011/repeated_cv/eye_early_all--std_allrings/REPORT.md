# Repeated nested eye-grouped CV

task = early, groups = all,-std, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.577 |    0.419 |    0.727 |       nan |       nan |         0.583 |         0.583 |           0.478 | n/a (no permutation test) |
| rf      | 0.593 |    0.418 |    0.757 |       nan |       nan |         0.353 |         0.802 |           0.255 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.016 -0.079 0.044             0.647
