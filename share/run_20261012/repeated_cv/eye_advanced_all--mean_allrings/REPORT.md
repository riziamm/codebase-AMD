# Repeated nested eye-grouped CV

task = advanced, groups = all,-mean, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.886 |    0.802 |    0.95  |       nan |       nan |         0.865 |         0.802 |           0.462 | n/a (no permutation test) |
| rf      | 0.945 |    0.884 |    0.985 |       nan |       nan |         0.756 |         0.931 |           0.341 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.058 -0.115 -0.015             0.007
