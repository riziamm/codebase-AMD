# Repeated nested eye-grouped CV

task = any, groups = all,-skew, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.683 |    0.563 |    0.788 |       nan |       nan |         0.625 |         0.671 |           0.492 | n/a (no permutation test) |
| rf      | 0.709 |    0.587 |    0.819 |       nan |       nan |         0.675 |         0.644 |           0.532 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo   hi  boot_p_two_sided
   lr - rf -0.026 -0.083 0.03             0.412
