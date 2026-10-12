# Calibration and decision curves (out-of-fold, repeated nested CV)

Brier: lower is better; scaled Brier: 1 = perfect, 0 = no better than predicting the prevalence.
Slope < 1 = predictions too extreme (overfitting); intercept != 0 = systematic over/under-estimation
(expected here: all models used class weighting). dBrier < 0 = better calibrated than the random forest.

| task | grouping | model | Brier | scaled Brier | slope (ideal 1) | intercept (ideal 0) | dBrier vs RF |
|---|---|---|---|---|---|---|---|
| advanced | eye | Logistic regression | 0.139 [0.097, 0.190] | 0.42 [0.14, 0.59] | 0.89 [0.65, 1.75] | -0.59 [-1.36, 0.14] | 0.034 [-0.005, 0.074] |
| advanced | eye | Random forest | 0.110 [0.074, 0.149] | 0.54 [0.37, 0.68] | 2.01 [1.49, 3.38] | 0.01 [-0.55, 0.59] | - |
| advanced | eye | Zone-token hybrid | 0.162 [0.097, 0.242] | 0.32 [-0.03, 0.57] | 0.26 [0.18, 0.40] | -1.37 [-4.35, 3.56] | 0.060 [0.013, 0.111] |
| advanced | eye | Original hybrid | 0.146 [0.088, 0.215] | 0.39 [0.06, 0.62] | 0.28 [0.21, 0.40] | -0.38 [-4.09, 3.39] | 0.042 [-0.000, 0.089] |
| advanced | subject | Logistic regression | 0.174 [0.125, 0.225] | 0.27 [-0.01, 0.48] | 0.76 [0.48, 1.39] | -0.52 [-1.34, 0.16] | 0.046 [0.001, 0.089] |
| advanced | subject | Random forest | 0.129 [0.091, 0.175] | 0.46 [0.26, 0.61] | 1.67 [1.19, 2.89] | 0.04 [-0.56, 0.68] | - |
| advanced | subject | Zone-token hybrid | 0.197 [0.127, 0.276] | 0.18 [-0.19, 0.46] | 0.23 [0.16, 0.34] | 1.49 [-4.24, 3.99] | 0.066 [0.013, 0.122] |
| advanced | subject | Original hybrid | 0.217 [0.139, 0.300] | 0.09 [-0.29, 0.39] | 0.20 [0.14, 0.30] | 0.30 [-4.50, 3.87] | 0.081 [0.027, 0.141] |
| any | eye | Logistic regression | 0.238 [0.199, 0.277] | 0.04 [-0.17, 0.19] | 0.53 [0.09, 1.14] | 0.24 [-0.37, 0.85] | 0.022 [0.004, 0.041] |
| any | eye | Random forest | 0.218 [0.185, 0.252] | 0.12 [-0.06, 0.24] | 1.05 [0.37, 2.11] | 0.06 [-0.47, 0.64] | - |
| any | eye | Zone-token hybrid | 0.393 [0.325, 0.467] | -0.59 [-1.00, -0.34] | 0.05 [-0.01, 0.11] | 0.78 [-2.60, 3.76] | 0.173 [0.129, 0.223] |
| any | eye | Original hybrid | 0.340 [0.256, 0.431] | -0.38 [-0.82, -0.06] | 0.10 [0.03, 0.17] | 0.22 [-4.17, 4.53] | 0.124 [0.064, 0.192] |
| any | subject | Logistic regression | 0.254 [0.212, 0.300] | -0.03 [-0.26, 0.13] | 0.24 [0.04, 0.56] | 0.17 [-0.51, 0.89] | 0.026 [-0.003, 0.058] |
| any | subject | Random forest | 0.226 [0.198, 0.257] | 0.09 [-0.07, 0.18] | 0.94 [0.28, 1.83] | 0.07 [-0.46, 0.64] | - |
| any | subject | Zone-token hybrid | 0.393 [0.325, 0.472] | -0.59 [-0.99, -0.34] | 0.06 [-0.02, 0.11] | -0.64 [-3.81, 2.93] | 0.171 [0.121, 0.227] |
| any | subject | Original hybrid | 0.398 [0.302, 0.487] | -0.61 [-1.00, -0.26] | 0.06 [-0.01, 0.13] | -1.94 [-4.68, 3.06] | 0.167 [0.101, 0.234] |

Figures: calibration_dca_<task>_<grouping>.png
