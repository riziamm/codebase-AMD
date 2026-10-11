# Repeated nested eye-grouped CV

task = early, groups = Del, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.638 |    0.475 |    0.786 |     0.149 |     0.856 |         0.64  |         0.567 |           0.509 | NO DETECTABLE SIGNAL      |
| rf      | 0.557 |    0.405 |    0.713 |   nan     |   nan     |         0.397 |         0.617 |           0.388 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf 0.081 -0.001 0.164             0.051
