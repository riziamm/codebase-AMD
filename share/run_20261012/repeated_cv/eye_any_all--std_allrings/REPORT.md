# Repeated nested eye-grouped CV

task = any, groups = all,-std, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.651 |    0.533 |    0.753 |       nan |       nan |         0.623 |         0.61  |           0.519 | n/a (no permutation test) |
| rf      | 0.706 |    0.584 |    0.822 |       nan |       nan |         0.67  |         0.646 |           0.528 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.055 -0.118 0.006             0.078
