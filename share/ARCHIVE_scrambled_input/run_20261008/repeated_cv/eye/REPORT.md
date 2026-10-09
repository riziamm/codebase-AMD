# Repeated nested eye-grouped CV

group_by = eye, eyes = 58, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.391 |    0.316 |    0.469 |     0.99  |     0.015 |         0.439 |         0.404 |           0.509 | BELOW CHANCE         |
| rf          | 0.409 |    0.328 |    0.496 |     0.886 |     0.119 |         0.67  |         0.212 |           0.723 | NO DETECTABLE SIGNAL |
| hybrid_zone | 0.499 |    0.425 |    0.565 |     0.545 |     0.465 |         0.538 |         0.433 |           0.551 | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.514 |    0.429 |    0.596 |     0.416 |     0.594 |         0.575 |         0.429 |           0.573 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo     hi  boot_p_two_sided
                  lr - rf -0.018 -0.096  0.061             0.651
         lr - hybrid_zone -0.108 -0.198 -0.009             0.037
         lr - hybrid_orig -0.122 -0.203 -0.031             0.008
         rf - hybrid_zone -0.090 -0.160 -0.015             0.019
         rf - hybrid_orig -0.105 -0.178 -0.022             0.013
hybrid_zone - hybrid_orig -0.015 -0.081  0.053             0.656
