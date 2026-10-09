# Repeated nested eye-grouped CV

task = advanced, groups = all, rings = outer, group_by = eye, eyes = 43, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:----------|
| lr      | 0.864 |    0.754 |    0.951 |     0.015 |      0.99 |         0.829 |         0.808 |           0.444 | SIGNAL    |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

n/a
