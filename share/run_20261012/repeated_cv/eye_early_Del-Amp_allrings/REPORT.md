# Repeated nested eye-grouped CV

task = early, groups = Del,Amp, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.584 |    0.42  |    0.74  |     0.328 |     0.677 |         0.633 |         0.513 |           0.54  | NO DETECTABLE SIGNAL      |
| rf      | 0.516 |    0.363 |    0.673 |   nan     |   nan     |         0.323 |         0.644 |           0.344 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.068 0.007 0.137              0.03
