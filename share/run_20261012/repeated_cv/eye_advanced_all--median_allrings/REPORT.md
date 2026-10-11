# Repeated nested eye-grouped CV

task = advanced, groups = all,-median, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.889 |    0.809 |    0.951 |       nan |       nan |         0.856 |         0.804 |           0.457 | n/a (no permutation test) |
| rf      | 0.941 |    0.877 |    0.982 |       nan |       nan |         0.759 |         0.925 |           0.345 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.052 -0.109 -0.008              0.02
