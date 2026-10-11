# Repeated nested eye-grouped CV

task = any, groups = Del, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.509 |    0.415 |    0.606 |     0.488 |     0.517 |         0.553 |         0.454 |           0.55  | NO DETECTABLE SIGNAL      |
| rf      | 0.457 |    0.356 |    0.561 |   nan     |   nan     |         0.609 |         0.375 |           0.616 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.052 -0.05 0.148             0.307
