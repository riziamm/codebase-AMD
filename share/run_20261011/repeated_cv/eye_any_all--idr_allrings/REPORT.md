# Repeated nested eye-grouped CV

task = any, groups = all,-idr, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.655 |    0.54  |    0.759 |       nan |       nan |         0.627 |         0.608 |           0.522 | n/a (no permutation test) |
| rf      | 0.709 |    0.587 |    0.825 |       nan |       nan |         0.681 |         0.64  |           0.537 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.053 -0.113 0.005             0.082
