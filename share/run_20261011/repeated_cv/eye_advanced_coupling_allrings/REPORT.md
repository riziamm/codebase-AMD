# Repeated nested eye-grouped CV

task = advanced, groups = coupling, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.574 |    0.443 |    0.7   |     0.493 |     0.512 |         0.518 |         0.602 |           0.445 | NO DETECTABLE SIGNAL      |
| rf      | 0.624 |    0.509 |    0.735 |   nan     |   nan     |         0.306 |         0.771 |           0.259 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.05 -0.128 0.028             0.216
