# Repeated nested eye-grouped CV

task = advanced, groups = mean,median,std,iqr,idr,skew,kurt, rings = all, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:----------|
| lr      |  0.91 |    0.829 |    0.969 |     0.005 |         1 |         0.891 |         0.796 |           0.476 | SIGNAL    |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

n/a
