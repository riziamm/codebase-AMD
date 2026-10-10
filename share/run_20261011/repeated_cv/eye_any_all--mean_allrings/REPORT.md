# Repeated nested eye-grouped CV

task = any, groups = all,-mean, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.67  |    0.554 |    0.773 |       nan |       nan |         0.628 |         0.648 |           0.504 | n/a (no permutation test) |
| rf      | 0.714 |    0.589 |    0.827 |       nan |       nan |         0.708 |         0.623 |           0.559 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.043 -0.106 0.017             0.161
