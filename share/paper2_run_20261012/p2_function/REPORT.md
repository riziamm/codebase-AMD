# Paper 2, P1: functional validity (out-of-fold, subject-grouped repeated CV)

R2 = out-of-fold R^2 (can be < 0); rho = Spearman(prediction, observed); 95% CI = subject-clustered bootstrap.
Plan v3: P1 is the only confirmatory test; secondary and supplementary contrasts are reported with CIs and no significance claims.

## Contrasts (d = A - B, paired)
~~~
 id          role outcome model visit             set_A       set_B  n_eyes   R2_A   R2_B    dR2  dR2_lo  dR2_hi   drho  drho_lo  drho_hi             conclusion
 P1       PRIMARY    VAlc ridge  mean       AS+MPOD+OFA     AS+MPOD      58 -0.059 -0.126  0.067  -0.035   0.157 -0.037   -0.084    0.011 no difference detected
 S1     secondary      VA ridge  mean       AS+MPOD+OFA     AS+MPOD      56 -0.050 -0.072  0.022  -0.002   0.058 -0.018   -0.048    0.011 no difference detected
S2a     secondary    VAlc ridge  mean     AS+MPOD+OFA44 AS+MPOD+OFA      58 -0.058 -0.059  0.001  -0.021   0.018 -0.023   -0.048    0.004 no difference detected
S2b     secondary      VA ridge  mean     AS+MPOD+OFA44 AS+MPOD+OFA      56 -0.045 -0.050  0.005   0.001   0.009 -0.009   -0.018    0.002     A > B (dR2 CI > 0)
 S3     secondary    VAlc ridge  last       AS+MPOD+OFA     AS+MPOD      58 -0.090 -0.114  0.024  -0.082   0.087 -0.054   -0.095   -0.013 no difference detected
 S4     secondary    VAlc ridge first       AS+MPOD+OFA     AS+MPOD      58 -0.046 -0.039 -0.007  -0.045   0.008 -0.049   -0.096   -0.002 no difference detected
X1a supplementary    VAlc ridge  mean            AS+OFA          AS      58 -0.054 -0.070  0.017  -0.040   0.048 -0.075   -0.190    0.044 no difference detected
X1b supplementary      VA ridge  mean            AS+OFA          AS      56 -0.075 -0.087  0.012  -0.032   0.022 -0.062   -0.124    0.000 no difference detected
X2a supplementary    VAlc ridge  mean           AS+MPOD          AS      58 -0.126 -0.070 -0.056  -0.186   0.078  0.210   -0.043    0.424 no difference detected
X2b supplementary      VA ridge  mean           AS+MPOD          AS      56 -0.072 -0.087  0.015  -0.093   0.036 -0.044   -0.130    0.050 no difference detected
X3a supplementary    VAlc ridge  mean      AS+grade+OFA    AS+grade      58 -0.080 -0.090  0.011  -0.072   0.054 -0.146   -0.281   -0.013 no difference detected
X3b supplementary      VA ridge  mean      AS+grade+OFA    AS+grade      56 -0.090 -0.090 -0.000  -0.067   0.013 -0.067   -0.136    0.000 no difference detected
X4a supplementary    VAlc ridge  mean       AS+MPOD+Del     AS+MPOD      58 -0.105 -0.126  0.021  -0.032   0.060 -0.004   -0.035    0.028 no difference detected
X4b supplementary    VAlc ridge  mean       AS+MPOD+Amp     AS+MPOD      58 -0.087 -0.126  0.039  -0.004   0.080 -0.009   -0.048    0.030 no difference detected
X5a supplementary    VAlc ridge  mean AS+MPOD+OFA+OFA44 AS+MPOD+OFA      58 -0.071 -0.059 -0.012  -0.060   0.012 -0.056   -0.090   -0.017 no difference detected
X5b supplementary      VA ridge  mean AS+MPOD+OFA+OFA44 AS+MPOD+OFA      56 -0.045 -0.050  0.005  -0.004   0.007 -0.022   -0.048    0.003 no difference detected
 X6 supplementary    VAlc    rf  mean       AS+MPOD+OFA     AS+MPOD      58  0.034  0.043 -0.009  -0.055   0.007 -0.062   -0.119   -0.015 no difference detected
 X7 supplementary    VAlc ridge  mean     AS+MPODz+OFAz    AS+MPODz      58 -0.011 -0.008 -0.003  -0.074   0.023 -0.022   -0.067    0.025 no difference detected
~~~

**Visit rule (pre-specified):** CONSISTENT across mean/last/first visit: no difference detected

## All runs
~~~
outcome               set model visit  n_features  n_eyes     R2  R2_lo  R2_hi    rho  rho_lo  rho_hi   MAE
     VA                AS ridge  mean           2      56 -0.087 -0.270 -0.033 -0.190  -0.325  -0.037 6.576
     VA           AS+MPOD    rf  mean          23      56 -0.119 -0.613 -0.006 -0.138  -0.378   0.116 6.961
     VA           AS+MPOD ridge  mean          23      56 -0.072 -0.351 -0.057 -0.233  -0.379  -0.058 6.704
     VA       AS+MPOD+Amp ridge  mean          26      56 -0.068 -0.342 -0.055 -0.242  -0.384  -0.068 6.689
     VA       AS+MPOD+Del ridge  mean          26      56 -0.058 -0.329 -0.048 -0.254  -0.395  -0.078 6.651
     VA       AS+MPOD+OFA    rf  mean          29      56 -0.124 -0.651 -0.031 -0.178  -0.400   0.069 7.070
     VA       AS+MPOD+OFA ridge  mean          29      56 -0.050 -0.301 -0.044 -0.252  -0.390  -0.081 6.619
     VA AS+MPOD+OFA+OFA44 ridge  mean          39      56 -0.045 -0.297 -0.040 -0.274  -0.411  -0.106 6.613
     VA     AS+MPOD+OFA44 ridge  mean          33      56 -0.045 -0.290 -0.040 -0.260  -0.397  -0.089 6.607
     VA          AS+MPODz ridge  mean         142      56 -0.051 -0.324 -0.039 -0.197  -0.352  -0.019 6.622
     VA     AS+MPODz+OFAz ridge  mean         182      56 -0.044 -0.303 -0.036 -0.221  -0.369  -0.051 6.605
     VA            AS+OFA ridge  mean           8      56 -0.075 -0.278 -0.050 -0.252  -0.364  -0.111 6.603
     VA          AS+grade ridge  mean           3      56 -0.090 -0.273 -0.025 -0.175  -0.319  -0.019 6.577
     VA      AS+grade+OFA ridge  mean           9      56 -0.090 -0.316 -0.048 -0.241  -0.358  -0.093 6.639
   VAlc                AS ridge  mean           2      58 -0.070 -0.234 -0.029 -0.110  -0.255   0.050 9.317
   VAlc           AS+MPOD    rf  mean          23      58  0.043 -0.144  0.214  0.271  -0.046   0.531 8.699
   VAlc           AS+MPOD ridge first          23      58 -0.039 -0.259  0.021  0.081  -0.190   0.340 9.299
   VAlc           AS+MPOD ridge  last          23      58 -0.114 -0.348  0.028  0.111  -0.149   0.346 9.524
   VAlc           AS+MPOD ridge  mean          23      58 -0.126 -0.354  0.040  0.100  -0.171   0.358 9.447
   VAlc       AS+MPOD+Amp ridge  mean          26      58 -0.087 -0.287  0.044  0.091  -0.178   0.350 9.347
   VAlc       AS+MPOD+Del ridge  mean          26      58 -0.105 -0.314  0.025  0.096  -0.171   0.351 9.488
   VAlc       AS+MPOD+OFA    rf  mean          29      58  0.034 -0.157  0.190  0.209  -0.116   0.482 8.923
   VAlc       AS+MPOD+OFA ridge first          29      58 -0.046 -0.304  0.004  0.031  -0.234   0.286 9.428
   VAlc       AS+MPOD+OFA ridge  last          29      58 -0.090 -0.388  0.012  0.058  -0.197   0.295 9.650
   VAlc       AS+MPOD+OFA ridge  mean          29      58 -0.059 -0.253  0.024  0.062  -0.205   0.317 9.421
   VAlc AS+MPOD+OFA+OFA44 ridge  mean          39      58 -0.071 -0.302 -0.005  0.007  -0.245   0.246 9.591
   VAlc     AS+MPOD+OFA44 ridge  mean          33      58 -0.058 -0.261  0.012  0.040  -0.217   0.295 9.444
   VAlc          AS+MPODz ridge  mean         142      58 -0.008 -0.176  0.091  0.194  -0.067   0.423 9.112
   VAlc     AS+MPODz+OFAz ridge  mean         182      58 -0.011 -0.220  0.062  0.172  -0.068   0.392 9.298
   VAlc            AS+OFA ridge  mean           8      58 -0.054 -0.256 -0.038 -0.185  -0.296  -0.050 9.438
   VAlc          AS+grade ridge  mean           3      58 -0.090 -0.284  0.054  0.150  -0.090   0.358 9.377
   VAlc      AS+grade+OFA ridge  mean           9      58 -0.080 -0.256  0.020  0.005  -0.182   0.176 9.492
~~~
