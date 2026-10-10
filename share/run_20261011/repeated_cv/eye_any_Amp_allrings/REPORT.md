# Repeated nested eye-grouped CV

task = any, groups = Amp, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.461 |    0.372 |    0.55  |      0.91 |     0.095 |         0.48  |         0.419 |           0.525 | NO DETECTABLE SIGNAL      |
| rf      | 0.374 |    0.287 |    0.467 |    nan    |   nan     |         0.566 |         0.275 |           0.637 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.087 0.008 0.164             0.027
