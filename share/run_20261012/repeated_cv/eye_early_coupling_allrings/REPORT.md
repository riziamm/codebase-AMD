# Repeated nested eye-grouped CV

task = early, groups = coupling, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.604 |    0.459 |    0.735 |     0.299 |     0.706 |         0.547 |         0.581 |           0.466 | NO DETECTABLE SIGNAL      |
| rf      | 0.558 |    0.425 |    0.688 |   nan     |   nan     |         0.363 |         0.671 |           0.341 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo   hi  boot_p_two_sided
   lr - rf 0.047 -0.072 0.15             0.397
