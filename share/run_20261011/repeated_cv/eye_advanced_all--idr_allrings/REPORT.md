# Repeated nested eye-grouped CV

task = advanced, groups = all,-idr, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.887 |    0.806 |    0.949 |       nan |       nan |         0.859 |         0.783 |           0.471 | n/a (no permutation test) |
| rf      | 0.941 |    0.877 |    0.984 |       nan |       nan |         0.747 |         0.933 |           0.336 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.054 -0.105 -0.014             0.007
