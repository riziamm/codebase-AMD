# Repeated nested eye-grouped CV

task = advanced, groups = mean,median,std,iqr,idr,skew,kurt,coupling, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.906 |    0.829 |    0.964 |       nan |       nan |         0.888 |         0.769 |           0.491 | n/a (no permutation test) |
| rf      | 0.941 |    0.877 |    0.984 |       nan |       nan |         0.744 |         0.927 |           0.338 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.035 -0.083 0.004             0.083
