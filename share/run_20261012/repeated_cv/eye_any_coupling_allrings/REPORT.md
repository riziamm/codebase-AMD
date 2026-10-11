# Repeated nested eye-grouped CV

task = any, groups = coupling, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.63  |    0.516 |    0.742 |     0.149 |     0.856 |         0.6   |          0.56 |           0.528 | NO DETECTABLE SIGNAL      |
| rf      | 0.603 |    0.482 |    0.712 |   nan     |   nan     |         0.625 |          0.51 |           0.565 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf 0.027 -0.044 0.102             0.462
