# Repeated nested eye-grouped CV

group_by = subject, eyes = 58, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.396 |    0.319 |    0.478 |     0.995 |     0.01  |         0.434 |         0.392 |           0.512 | BELOW CHANCE         |
| rf          | 0.378 |    0.29  |    0.477 |     0.935 |     0.07  |         0.597 |         0.238 |           0.671 | NO DETECTABLE SIGNAL |
| hybrid_zone | 0.483 |    0.409 |    0.556 |     0.861 |     0.149 |         0.53  |         0.448 |           0.54  | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.466 |    0.374 |    0.557 |     0.851 |     0.158 |         0.509 |         0.423 |           0.54  | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo     hi  boot_p_two_sided
                  lr - rf  0.018 -0.059  0.094             0.659
         lr - hybrid_zone -0.087 -0.166  0.002             0.059
         lr - hybrid_orig -0.070 -0.151  0.021             0.131
         rf - hybrid_zone -0.105 -0.172 -0.034             0.002
         rf - hybrid_orig -0.088 -0.162 -0.006             0.032
hybrid_zone - hybrid_orig  0.018 -0.047  0.081             0.588
