# Repeated nested eye-grouped CV

task = early, groups = all,coupling, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.586 |    0.422 |    0.745 |       nan |       nan |         0.583 |         0.613 |           0.459 | n/a (no permutation test) |
| rf      | 0.602 |    0.424 |    0.767 |       nan |       nan |         0.36  |         0.815 |           0.249 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo   hi  boot_p_two_sided
   lr - rf -0.015 -0.069 0.04             0.613
