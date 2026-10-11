# Repeated nested eye-grouped CV

task = early, groups = Amp, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.391 |    0.285 |    0.5   |      0.98 |     0.025 |         0.337 |         0.477 |           0.455 | BELOW CHANCE              |
| rf      | 0.296 |    0.193 |    0.411 |    nan    |   nan     |         0.107 |         0.733 |           0.209 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.095 0.015 0.176             0.016
