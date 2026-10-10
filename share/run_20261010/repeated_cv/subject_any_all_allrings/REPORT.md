# Repeated nested eye-grouped CV

task = any, groups = all, rings = all, group_by = subject, eyes = 58, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.632 |    0.51  |    0.745 |     0.04  |     0.965 |         0.633 |         0.56  |           0.547 | NO DETECTABLE SIGNAL |
| rf          | 0.658 |    0.537 |    0.771 |     0.04  |     0.965 |         0.678 |         0.51  |           0.594 | SIGNAL               |
| hybrid_zone | 0.569 |    0.467 |    0.667 |     0.069 |     0.941 |         0.625 |         0.483 |           0.577 | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.594 |    0.473 |    0.703 |     0.059 |     0.95  |         0.658 |         0.487 |           0.593 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo    hi  boot_p_two_sided
                  lr - rf -0.027 -0.107 0.054             0.521
         lr - hybrid_zone  0.062 -0.015 0.144             0.143
         lr - hybrid_orig  0.038 -0.034 0.115             0.318
         rf - hybrid_zone  0.089  0.019 0.165             0.008
         rf - hybrid_orig  0.064 -0.011 0.145             0.098
hybrid_zone - hybrid_orig -0.024 -0.087 0.038             0.450
