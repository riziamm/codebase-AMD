# Repeated nested eye-grouped CV

task = advanced, groups = all, rings = all, group_by = eye, eyes = 43, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict   |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:----------|
| lr          | 0.886 |    0.803 |    0.947 |     0.005 |         1 |         0.862 |         0.779 |           0.474 | SIGNAL    |
| rf          | 0.938 |    0.872 |    0.982 |     0.005 |         1 |         0.753 |         0.925 |           0.343 | SIGNAL    |
| hybrid_zone | 0.891 |    0.806 |    0.955 |     0.02  |         1 |         0.762 |         0.856 |           0.388 | SIGNAL    |
| hybrid_orig | 0.907 |    0.836 |    0.958 |     0.01  |         1 |         0.821 |         0.852 |           0.414 | SIGNAL    |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo     hi  boot_p_two_sided
                  lr - rf -0.052 -0.103 -0.012             0.007
         lr - hybrid_zone -0.005 -0.050  0.035             0.867
         lr - hybrid_orig -0.021 -0.047  0.001             0.060
         rf - hybrid_zone  0.047  0.005  0.098             0.021
         rf - hybrid_orig  0.031 -0.004  0.073             0.087
hybrid_zone - hybrid_orig -0.016 -0.051  0.019             0.387
