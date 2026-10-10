# Repeated nested eye-grouped CV

task = advanced, groups = all,-skew, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.883 |    0.796 |    0.948 |       nan |       nan |         0.835 |         0.804 |           0.449 | n/a (no permutation test) |
| rf      | 0.938 |    0.876 |    0.981 |       nan |       nan |         0.75  |         0.929 |           0.34  | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.055 -0.105 -0.014             0.007
