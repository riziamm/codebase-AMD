# Repeated nested eye-grouped CV

task = any, groups = all,coupling, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.676 |    0.56  |    0.779 |       nan |       nan |         0.652 |         0.662 |           0.511 | n/a (no permutation test) |
| rf      | 0.716 |    0.591 |    0.827 |       nan |       nan |         0.698 |         0.629 |           0.552 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.04 -0.093 0.013             0.138
