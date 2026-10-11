### Discrimination by task and model (repeated nested CV)

| Task | Model | AUC [95% CI] (eye-grouped) | p (perm.) | Sens / Spec at 0.5 | AUC [95% CI] (subject-grouped) | p (perm.)  | n eyes |
|---|---|---|---|---|---|---|---|
| Any AMD (AREDS 2-4 vs 1) | Logistic regression | 0.67 [0.55-0.78] | 0.11 | 0.63 / 0.63 | 0.63 [0.51-0.74] | 0.04 | 58 |
| Any AMD (AREDS 2-4 vs 1) | Random forest | 0.71 [0.59-0.83] | 0.02 | 0.70 / 0.62 | 0.66 [0.54-0.77] | 0.04 | 58 |
| Any AMD (AREDS 2-4 vs 1) | Zone-token CNN-Transformer | 0.58 [0.48-0.67] | 0.50 | 0.58 / 0.58 | 0.57 [0.47-0.67] | 0.07 | 58 |
| Any AMD (AREDS 2-4 vs 1) | CNN-Transformer (original) | 0.67 [0.54-0.77] | 0.18 | 0.67 / 0.61 | 0.59 [0.47-0.70] | 0.06 | 58 |
| Early AMD (AREDS 2 vs 1) | Logistic regression | 0.58 [0.42-0.73] | 0.43 | 0.57 / 0.60 | 0.49 [0.35-0.64] | 0.26 | 41 |
| Early AMD (AREDS 2 vs 1) | Random forest | 0.59 [0.42-0.75] | 0.23 | 0.35 / 0.81 | 0.50 [0.34-0.66] | 0.18 | 41 |
| Early AMD (AREDS 2 vs 1) | Zone-token CNN-Transformer | 0.52 [0.39-0.65] | 0.47 | 0.41 / 0.63 | 0.43 [0.31-0.56] | 0.69 | 41 |
| Early AMD (AREDS 2 vs 1) | CNN-Transformer (original) | 0.51 [0.36-0.65] | 0.46 | 0.36 / 0.68 | 0.40 [0.27-0.54] | 0.67 | 41 |
| Intermediate/advanced (AREDS 3-4 vs 1) | Logistic regression | 0.89 [0.80-0.95] | <0.01 | 0.86 / 0.78 | 0.83 [0.72-0.91] | 0.01 | 43 |
| Intermediate/advanced (AREDS 3-4 vs 1) | Random forest | 0.94 [0.87-0.98] | <0.01 | 0.75 / 0.93 | 0.89 [0.80-0.96] | <0.01 | 43 |
| Intermediate/advanced (AREDS 3-4 vs 1) | Zone-token CNN-Transformer | 0.89 [0.81-0.96] | 0.02 | 0.76 / 0.86 | 0.85 [0.75-0.93] | <0.01 | 43 |
| Intermediate/advanced (AREDS 3-4 vs 1) | CNN-Transformer (original) | 0.91 [0.84-0.96] | <0.01 | 0.82 / 0.85 | 0.84 [0.75-0.91] | 0.02 | 43 |
