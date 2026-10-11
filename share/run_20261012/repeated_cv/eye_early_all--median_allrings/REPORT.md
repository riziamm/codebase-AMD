# Repeated nested eye-grouped CV

task = early, groups = all,-median, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.61  |    0.441 |    0.765 |       nan |       nan |         0.593 |         0.615 |           0.461 | n/a (no permutation test) |
| rf      | 0.605 |    0.433 |    0.767 |       nan |       nan |         0.347 |         0.81  |           0.248 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf 0.005 -0.049 0.058             0.825
