# Repeated nested eye-grouped CV

task = any, groups = all, rings = all, group_by = eye, eyes = 58, repeats = 10

| model       |   AUC |   AUC_lo |   AUC_hi |   p_above |   p_below |   sensitivity |   specificity |   pred_pos_rate | verdict              |
|:------------|------:|---------:|---------:|----------:|----------:|--------------:|--------------:|----------------:|:---------------------|
| lr          | 0.672 |    0.553 |    0.777 |     0.114 |     0.891 |         0.628 |         0.635 |           0.51  | NO DETECTABLE SIGNAL |
| rf          | 0.714 |    0.591 |    0.826 |     0.02  |     0.985 |         0.7   |         0.623 |           0.555 | SIGNAL               |
| hybrid_zone | 0.582 |    0.482 |    0.674 |     0.505 |     0.505 |         0.58  |         0.577 |           0.509 | NO DETECTABLE SIGNAL |
| hybrid_orig | 0.665 |    0.543 |    0.774 |     0.178 |     0.832 |         0.669 |         0.606 |           0.546 | NO DETECTABLE SIGNAL |

All-positive F1 baseline: 0.711

Paired dAUC (eye-clustered bootstrap):

               comparison   dAUC     lo     hi  boot_p_two_sided
                  lr - rf -0.042 -0.096  0.009             0.113
         lr - hybrid_zone  0.090  0.034  0.146             0.004
         lr - hybrid_orig  0.006 -0.044  0.059             0.817
         rf - hybrid_zone  0.132  0.070  0.195             0.000
         rf - hybrid_orig  0.048 -0.019  0.119             0.153
hybrid_zone - hybrid_orig -0.084 -0.143 -0.024             0.005
