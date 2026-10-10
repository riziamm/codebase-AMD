# Repeated nested eye-grouped CV

task = early, groups = all,-Amp, rings = all, group_by = eye, eyes = 41, repeats = 10

| model   |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict                   |
|:--------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:--------------------------|
| lr      | 0.617 |    0.461 |    0.761 |       nan |       nan |         0.587 |         0.621 |           0.455 | n/a (no permutation test) |
| rf      | 0.616 |    0.447 |    0.773 |       nan |       nan |         0.39  |         0.802 |           0.268 | n/a (no permutation test) |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

comparison  dAUC     lo    hi  boot_p_two_sided
   lr - rf 0.001 -0.063 0.067             0.948
