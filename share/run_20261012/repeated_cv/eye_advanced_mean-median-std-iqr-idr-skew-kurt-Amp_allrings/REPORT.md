# Repeated nested eye-grouped CV

task = advanced, groups = mean,median,std,iqr,idr,skew,kurt,Amp, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.911 |    0.838 |    0.963 |       nan |       nan |         0.888 |         0.798 |           0.473 | n/a (no permutation test) |
| rf      | 0.942 |    0.878 |    0.984 |       nan |       nan |         0.747 |         0.929 |           0.338 | n/a (no permutation test) |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

comparison   dAUC     lo    hi  boot_p_two_sided
   lr - rf -0.031 -0.074 0.002             0.066
