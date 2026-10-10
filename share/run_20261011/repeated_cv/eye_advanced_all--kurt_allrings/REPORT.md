# Repeated nested eye-grouped CV

task = advanced, groups = all,-kurt, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.88  |    0.793 |    0.948 |       nan |       nan |         0.853 |         0.792 |           0.463 | n/a (no permutation test) |
| rf      | 0.924 |    0.851 |    0.973 |       nan |       nan |         0.729 |         0.919 |           0.337 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.043 -0.106 0.007             0.099
