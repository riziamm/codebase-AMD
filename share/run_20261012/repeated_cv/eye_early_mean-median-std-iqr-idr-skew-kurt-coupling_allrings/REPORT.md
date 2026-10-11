# Repeated nested eye-grouped CV

task = early, groups = mean,median,std,iqr,idr,skew,kurt,coupling, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.546 |    0.404 |    0.679 |       nan |       nan |          0.5  |         0.563 |           0.46  | n/a (no permutation test) |
| rf      | 0.585 |    0.426 |    0.732 |       nan |       nan |          0.29 |         0.808 |           0.228 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.038 -0.119 0.037              0.35
