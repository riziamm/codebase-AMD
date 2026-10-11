# Repeated nested eye-grouped CV

task = any, groups = all,-median, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.673 |    0.557 |    0.774 |       nan |       nan |         0.622 |         0.64  |           0.504 | n/a (no permutation test) |
| rf      | 0.712 |    0.586 |    0.825 |       nan |       nan |         0.692 |         0.612 |           0.556 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.039 -0.102 0.018             0.195
