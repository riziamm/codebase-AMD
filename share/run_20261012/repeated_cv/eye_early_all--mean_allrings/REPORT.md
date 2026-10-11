# Repeated nested eye-grouped CV

task = early, groups = all,-mean, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.609 |    0.44  |    0.76  |       nan |       nan |         0.593 |         0.621 |           0.457 | n/a (no permutation test) |
| rf      | 0.605 |    0.432 |    0.765 |       nan |       nan |         0.347 |         0.81  |           0.248 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.003 -0.05 0.058             0.863
