# Calibration and decision curves (out-of-fold, repeated nested CV)

Brier: lower is better; scaled Brier: 1 = perfect, 0 = no better than predicting the prevalence.
Slope < 1 = predictions too extreme (overfitting); intercept != 0 = systematic over/under-estimation
(expected here: all models used class weighting). dBrier < 0 = better calibrated than the random forest.

| task | grouping | model | Brier | scaled Brier | slope (ideal 1) | intercept (ideal 0) | dBrier vs RF |
|---|---|---|---|---|---|---|---|
| advanced | eye | Logistic regression | 0.143 [0.096, 0.191] | 0.40 [0.13, 0.59] | 1.03 [0.71, 1.85] | -0.62 [-1.40, 0.14] | 0.034 [-0.005, 0.074] |
| advanced | eye | Random forest | 0.109 [0.074, 0.148] | 0.55 [0.37, 0.68] | 2.05 [1.51, 3.44] | 0.01 [-0.55, 0.59] | - |
| advanced | eye | Zone-token hybrid | 0.169 [0.104, 0.243] | 0.29 [-0.05, 0.53] | 0.23 [0.16, 0.37] | -2325996.81 [-170882070.91, 97654879.37] | 0.060 [0.013, 0.111] |
| advanced | eye | Original hybrid | 0.151 [0.093, 0.217] | 0.37 [0.06, 0.61] | 0.17 [0.13, 0.26] | -2325475.02 [-98029583.52, 47221577.32] | 0.042 [-0.000, 0.089] |
| advanced | subject | Logistic regression | 0.177 [0.129, 0.230] | 0.26 [-0.02, 0.46] | 0.84 [0.55, 1.52] | -0.59 [-1.41, 0.16] | 0.046 [0.001, 0.089] |
| advanced | subject | Random forest | 0.131 [0.093, 0.175] | 0.45 [0.26, 0.60] | 1.67 [1.20, 2.87] | 0.06 [-0.54, 0.68] | - |
| advanced | subject | Zone-token hybrid | 0.197 [0.129, 0.276] | 0.18 [-0.17, 0.44] | 0.20 [0.13, 0.31] | 0.73 [-90645436.14, 86722233.43] | 0.066 [0.013, 0.122] |
| advanced | subject | Original hybrid | 0.212 [0.139, 0.290] | 0.11 [-0.26, 0.40] | 0.12 [0.09, 0.20] | -0.10 [-88372206.95, 43799776.31] | 0.081 [0.027, 0.141] |
| any | eye | Logistic regression | 0.239 [0.205, 0.280] | 0.03 [-0.18, 0.16] | 0.55 [0.15, 1.18] | 0.21 [-0.40, 0.86] | 0.022 [0.004, 0.041] |
| any | eye | Random forest | 0.217 [0.185, 0.252] | 0.12 [-0.05, 0.24] | 1.06 [0.35, 2.13] | 0.06 [-0.48, 0.63] | - |
| any | eye | Zone-token hybrid | 0.391 [0.324, 0.462] | -0.58 [-0.97, -0.33] | 0.04 [-0.01, 0.10] | 0.85 [-2.58, 4.05] | 0.173 [0.129, 0.223] |
| any | eye | Original hybrid | 0.341 [0.262, 0.430] | -0.38 [-0.81, -0.09] | 0.06 [0.02, 0.11] | 0.20 [-5.93, 21008227.41] | 0.124 [0.064, 0.192] |
| any | subject | Logistic regression | 0.257 [0.214, 0.302] | -0.04 [-0.27, 0.12] | 0.32 [0.06, 0.70] | 0.17 [-0.53, 0.89] | 0.026 [-0.003, 0.058] |
| any | subject | Random forest | 0.230 [0.201, 0.259] | 0.07 [-0.09, 0.17] | 0.90 [0.25, 1.75] | 0.05 [-0.47, 0.63] | - |
| any | subject | Zone-token hybrid | 0.401 [0.328, 0.477] | -0.62 [-0.98, -0.36] | 0.04 [-0.01, 0.10] | -0.66 [-4.03, 2.72] | 0.171 [0.121, 0.227] |
| any | subject | Original hybrid | 0.397 [0.309, 0.484] | -0.61 [-1.00, -0.29] | 0.03 [-0.01, 0.08] | -1.81 [-24928719.78, 17241597.88] | 0.167 [0.101, 0.234] |

Figures: calibration_dca_<task>_<grouping>.png
