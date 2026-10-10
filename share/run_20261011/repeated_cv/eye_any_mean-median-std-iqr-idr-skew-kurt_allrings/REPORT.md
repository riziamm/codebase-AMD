# Repeated nested eye-grouped CV

task = any, groups = mean,median,std,iqr,idr,skew,kurt, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.723 |    0.607 |    0.827 |      0.01 |     0.995 |         0.67  |         0.656 |           0.524 | SIGNAL                    |
| rf      | 0.733 |    0.61  |    0.842 |    nan    |   nan     |         0.691 |         0.656 |           0.535 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.01 -0.067 0.051             0.783
