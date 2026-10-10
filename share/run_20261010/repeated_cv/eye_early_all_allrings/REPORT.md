# Repeated nested eye-grouped CV

task = early, groups = all, rings = all, group_by = eye, eyes = 41, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.581 |    0.422 |    0.731 |     0.428 |     0.577 |         0.57  |         0.602 |           0.461 | NO DETECTABLE SIGNAL |
| rf          | 0.589 |    0.416 |    0.752 |     0.234 |     0.771 |         0.353 |         0.806 |           0.252 | NO DETECTABLE SIGNAL |
| hybrid_zone | 0.525 |    0.393 |    0.649 |     0.465 |     0.554 |         0.41  |         0.627 |           0.387 | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.51  |    0.358 |    0.651 |     0.455 |     0.554 |         0.36  |         0.677 |           0.337 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo    hi  boot_p_two_sided
                  lr - rf -0.008 -0.070 0.055             0.815
         lr - hybrid_zone  0.056 -0.024 0.149             0.185
         lr - hybrid_orig  0.071 -0.015 0.158             0.108
         rf - hybrid_zone  0.064 -0.029 0.157             0.179
         rf - hybrid_orig  0.078 -0.009 0.168             0.073
hybrid_zone - hybrid_orig  0.015 -0.054 0.085             0.734
