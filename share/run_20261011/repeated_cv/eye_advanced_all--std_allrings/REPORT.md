# Repeated nested eye-grouped CV

task = advanced, groups = all,-std, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.884 |    0.801 |    0.947 |       nan |       nan |         0.856 |         0.787 |           0.467 | n/a (no permutation test) |
| rf      | 0.939 |    0.874 |    0.983 |       nan |       nan |         0.747 |         0.935 |           0.335 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.055 -0.109 -0.014             0.007
