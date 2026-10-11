# Repeated nested eye-grouped CV

task = advanced, groups = mean,median,std,iqr,idr,skew,kurt,Del, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.883 |    0.797 |    0.952 |       nan |       nan |         0.859 |         0.775 |           0.476 | n/a (no permutation test) |
| rf      | 0.939 |    0.875 |    0.983 |       nan |       nan |         0.744 |         0.923 |           0.341 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo     hi  boot_p_two_sided
   lr - rf -0.056 -0.108 -0.014             0.005
