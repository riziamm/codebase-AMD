# Repeated nested eye-grouped CV

task = early, groups = all, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr      | 0.581 |    0.422 |    0.731 |     0.428 |     0.577 |         0.57  |         0.602 |           0.461 | NO DETECTABLE SIGNAL |
| rf      | 0.589 |    0.416 |    0.752 |     0.234 |     0.771 |         0.353 |         0.806 |           0.252 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC    lo    hi  boot_p_two_sided
   lr - rf -0.008 -0.07 0.055             0.815
