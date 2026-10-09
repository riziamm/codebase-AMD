# Repeated nested eye-grouped CV

task = advanced, groups = all, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:----------|
| lr      | 0.886 |    0.803 |    0.947 |     0.005 |         1 |         0.862 |         0.779 |           0.474 | SIGNAL    |
| rf      | 0.938 |    0.872 |    0.982 |     0.005 |         1 |         0.753 |         0.925 |           0.343 | SIGNAL    |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.052 -0.103 -0.012             0.007
