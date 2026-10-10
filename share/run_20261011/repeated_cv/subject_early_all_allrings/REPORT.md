# Repeated nested eye-grouped CV

task = early, groups = all, rings = all, group_by = subject, eyes = 41, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.491 |    0.349 |    0.636 |     0.259 |     0.746 |         0.5   |         0.512 |           0.493 | NO DETECTABLE SIGNAL |
| rf          | 0.502 |    0.344 |    0.661 |     0.184 |     0.821 |         0.26  |         0.735 |           0.263 | NO DETECTABLE SIGNAL |
| hybrid_zone | 0.432 |    0.313 |    0.555 |     0.693 |     0.317 |         0.373 |         0.527 |           0.437 | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.403 |    0.271 |    0.537 |     0.673 |     0.337 |         0.35  |         0.498 |           0.446 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.536

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo    hi  boot_p_two_sided
                  lr - rf -0.011 -0.072 0.047             0.788
         lr - hybrid_zone  0.059 -0.010 0.133             0.099
         lr - hybrid_orig  0.088  0.011 0.166             0.027
         rf - hybrid_zone  0.070 -0.018 0.152             0.118
         rf - hybrid_orig  0.099  0.013 0.182             0.025
hybrid_zone - hybrid_orig  0.029 -0.036 0.095             0.398
