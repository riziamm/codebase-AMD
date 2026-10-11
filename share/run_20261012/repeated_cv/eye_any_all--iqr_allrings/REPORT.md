# Repeated nested eye-grouped CV

task = any, groups = all,-iqr, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.663 |    0.544 |    0.765 |       nan |       nan |         0.631 |         0.623 |           0.517 | n/a (no permutation test) |
| rf      | 0.714 |    0.59  |    0.828 |       nan |       nan |         0.686 |         0.637 |           0.541 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.051 -0.112 0.008             0.098
