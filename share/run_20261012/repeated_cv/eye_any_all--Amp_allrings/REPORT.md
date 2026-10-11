# Repeated nested eye-grouped CV

task = any, groups = all,-Amp, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.702 |    0.586 |    0.81  |       nan |       nan |         0.67  |         0.665 |           0.52  | n/a (no permutation test) |
| rf      | 0.726 |    0.604 |    0.837 |       nan |       nan |         0.694 |         0.656 |           0.537 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC    lo    hi  boot_p_two_sided
   lr - rf -0.024 -0.09 0.043             0.519
