# Repeated nested eye-grouped CV

task = advanced, groups = Del,Amp, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.559 |    0.447 |    0.669 |     0.383 |     0.622 |         0.474 |         0.619 |           0.417 | NO DETECTABLE SIGNAL      |
| rf      | 0.55  |    0.382 |    0.711 |   nan     |   nan     |         0.426 |         0.731 |           0.331 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf  0.01 -0.094 0.114             0.882
