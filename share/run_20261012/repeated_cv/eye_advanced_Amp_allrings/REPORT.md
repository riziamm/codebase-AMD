# Repeated nested eye-grouped CV

task = advanced, groups = Amp, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.591 |    0.487 |    0.701 |     0.239 |     0.766 |         0.544 |         0.573 |           0.473 | NO DETECTABLE SIGNAL      |
| rf      | 0.548 |    0.411 |    0.686 |   nan     |   nan     |         0.374 |         0.681 |           0.341 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo   hi  boot_p_two_sided
   lr - rf 0.043 -0.039 0.13             0.293
