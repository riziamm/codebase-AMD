# Repeated nested eye-grouped CV

task = early, groups = mean,median,std,iqr,idr,skew,kurt, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.538 |    0.407 |    0.668 |     0.458 |     0.547 |         0.51  |         0.562 |           0.465 | NO DETECTABLE SIGNAL      |
| rf      | 0.576 |    0.424 |    0.716 |   nan     |   nan     |         0.287 |         0.788 |           0.239 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.038 -0.127 0.048             0.405
