# Repeated nested eye-grouped CV

task = advanced, groups = Del, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.496 |    0.373 |    0.612 |     0.448 |     0.557 |         0.488 |         0.542 |           0.47  | NO DETECTABLE SIGNAL      |
| rf      | 0.503 |    0.343 |    0.657 |   nan     |   nan     |         0.376 |         0.683 |           0.341 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.007 -0.138 0.131             0.897
