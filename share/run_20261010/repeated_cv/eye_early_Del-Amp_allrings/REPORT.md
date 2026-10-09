# Repeated nested eye-grouped CV

task = early, groups = Del,Amp, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr      | 0.584 |     0.42 |     0.74 |     0.328 |     0.677 |         0.633 |         0.513 |            0.54 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

n/a
