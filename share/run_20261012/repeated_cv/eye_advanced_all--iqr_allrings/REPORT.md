# Repeated nested eye-grouped CV

task = advanced, groups = all,-iqr, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.885 |    0.805 |    0.947 |       nan |       nan |         0.862 |         0.777 |           0.476 | n/a (no permutation test) |
| rf      | 0.938 |    0.873 |    0.983 |       nan |       nan |         0.738 |         0.931 |           0.334 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.053 -0.105 -0.013             0.007
