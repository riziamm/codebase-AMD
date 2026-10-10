# Repeated nested eye-grouped CV

task = any, groups = all,-kurt, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.697 |    0.583 |    0.802 |       nan |       nan |         0.652 |         0.673 |           0.506 | n/a (no permutation test) |
| rf      | 0.721 |    0.601 |    0.834 |       nan |       nan |         0.688 |         0.637 |           0.542 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.024 -0.081 0.033              0.42
