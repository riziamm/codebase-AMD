# Repeated nested eye-grouped CV

task = early, groups = mean,median,std,iqr,idr,skew,kurt, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr      | 0.538 |    0.407 |    0.668 |     0.458 |     0.547 |          0.51 |         0.562 |           0.465 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

n/a
