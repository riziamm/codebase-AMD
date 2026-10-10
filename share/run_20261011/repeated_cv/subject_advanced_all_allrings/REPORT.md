# Repeated nested eye-grouped CV

task = advanced, groups = all, rings = all, group_by = subject, eyes = 43, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict   |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:----------|
| lr          | 0.827 |    0.721 |    0.909 |     0.015 |      0.99 |         0.791 |         0.729 |           0.477 | SIGNAL    |
| rf          | 0.893 |    0.799 |    0.963 |     0.005 |      1    |         0.712 |         0.915 |           0.333 | SIGNAL    |
| hybrid_zone | 0.851 |    0.755 |    0.926 |     0.01  |      1    |         0.721 |         0.838 |           0.383 | SIGNAL    |
| hybrid_orig | 0.839 |    0.751 |    0.913 |     0.02  |      0.99 |         0.718 |         0.813 |           0.397 | SIGNAL    |

All-positive F1 baseline: 0.567

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo     hi  boot_p_two_sided
                  lr - rf -0.065 -0.135 -0.007             0.026
         lr - hybrid_zone -0.024 -0.082  0.031             0.381
         lr - hybrid_orig -0.012 -0.046  0.016             0.469
         rf - hybrid_zone  0.041 -0.020  0.109             0.200
         rf - hybrid_orig  0.053 -0.007  0.122             0.089
hybrid_zone - hybrid_orig  0.012 -0.038  0.064             0.634
