# Repeated nested eye-grouped CV

task = any, groups = Del,Amp, rings = all, group_by = eye, eyes = 58, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.463 |    0.378 |    0.547 |     0.831 |     0.174 |         0.48  |         0.437 |           0.517 | NO DETECTABLE SIGNAL      |
| rf      | 0.367 |    0.271 |    0.473 |   nan     |   nan     |         0.588 |         0.262 |           0.655 | n/a (no permutation test) |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC    lo    hi  boot_p_two_sided
   lr - rf 0.096 0.013 0.173             0.026
