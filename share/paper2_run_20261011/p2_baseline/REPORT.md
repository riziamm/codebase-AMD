# Paper 2 step S0: baseline and safety checks

data: {'n_eyes': 58, 'n_subjects': 29, 'sessions_per_eye': [2], 'areds_per_eye': {1: 26, 2: 15, 3: 10, 4: 7}, 'ring_map_default': True, 'bands44': None, 'sex_coding': {'1': 0, '2': 1}, 'has_ofa44': True}

## S0.1 Table 1 (median [IQR]; age/sex at subject level)
~~~
  group  n_eyes  n_subjects       age_median_IQR sex_counts_subjects        VA_median_IQR  VA_missing      VAlc_median_IQR  VAlc_missing
    all      58          29 70.00 [68.00, 73.00]       {2: 22, 1: 7} 83.00 [80.75, 85.00]           2 73.00 [68.00, 76.00]             0
AREDS 1      26          16 68.50 [66.00, 69.25]       {2: 12, 1: 4} 83.00 [81.00, 85.00]           0 74.00 [71.25, 76.75]             0
AREDS 2      15          11 69.00 [68.50, 71.50]        {2: 8, 1: 3} 85.00 [82.50, 89.50]           0 76.00 [73.00, 78.00]             0
AREDS 3      10           9 75.00 [71.00, 81.00]        {2: 6, 1: 3} 83.00 [80.00, 84.00]           1 68.00 [56.00, 72.25]             0
AREDS 4       7           7 76.00 [72.50, 84.00]        {2: 5, 1: 2} 74.50 [65.50, 81.25]           1 63.00 [53.50, 68.00]             0
~~~

## S0.2 Age-only discrimination (AUC of age; subject-clustered bootstrap 95% CI)
~~~
               task  n_eyes  n_pos  AUC_age    lo    hi                                                   verdict
     any (2-4 vs 1)      58     32    0.800 0.647 0.927 FLAG: age alone discriminates (report to Paper 1 authors)
     early (2 vs 1)      41     15    0.705 0.511 0.867 FLAG: age alone discriminates (report to Paper 1 authors)
advanced (3-4 vs 1)      43     17    0.885 0.726 0.991 FLAG: age alone discriminates (report to Paper 1 authors)
~~~

## S0.3 Spearman correlations with age
~~~
variable                sample  n_eyes  spearman_age     lo    hi
 Del|all normal eyes (AREDS 1)      26        -0.048 -0.576 0.480
 Amp|all normal eyes (AREDS 1)      26         0.239 -0.389 0.723
Del2|all normal eyes (AREDS 1)      26        -0.065 -0.599 0.479
Amp2|all normal eyes (AREDS 1)      26         0.188 -0.445 0.706
mean|all normal eyes (AREDS 1)      26        -0.263 -0.700 0.240
~~~

## S0.4 Follow-up fields (counts of subjects)
~~~
demographics_raw.csv not given
~~~
