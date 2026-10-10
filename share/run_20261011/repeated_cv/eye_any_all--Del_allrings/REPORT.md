# Repeated nested eye-grouped CV

task = any, groups = all,-Del, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.683 |    0.56  |    0.789 |       nan |       nan |         0.633 |         0.644 |           0.509 | n/a (no permutation test) |
| rf      | 0.723 |    0.602 |    0.834 |       nan |       nan |         0.667 |         0.671 |           0.516 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.04 -0.097 0.013             0.149
