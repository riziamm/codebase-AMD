# Repeated nested eye-grouped CV

task = any, groups = mean,median,std,iqr,idr,skew,kurt,coupling, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.716 |    0.606 |    0.82  |       nan |       nan |         0.675 |         0.677 |           0.517 | n/a (no permutation test) |
| rf      | 0.729 |    0.609 |    0.838 |       nan |       nan |         0.68  |         0.665 |           0.525 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.013 -0.073 0.049             0.705
