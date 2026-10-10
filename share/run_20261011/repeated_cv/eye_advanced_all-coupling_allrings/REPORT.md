# Repeated nested eye-grouped CV

task = advanced, groups = all,coupling, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.88  |    0.794 |    0.946 |       nan |       nan |         0.844 |         0.788 |           0.462 | n/a (no permutation test) |
| rf      | 0.939 |    0.876 |    0.982 |       nan |       nan |         0.738 |         0.929 |           0.335 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC    lo    hi  boot_p_two_sided
   lr - rf -0.059 -0.12 -0.01             0.016
