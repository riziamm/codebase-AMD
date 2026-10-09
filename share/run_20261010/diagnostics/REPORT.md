# Diagnostics  (eyes=58, subjects=29, bootstrap B=1000, subject-clustered)

## A. Strongest feature x ring effects per contrast (AUC>0.5 = higher in disease)

### early (2 vs 1)
```
    feature   AUC    lo    hi         direction  ci_excludes_05
 skew|outer 0.718 0.520 0.884 higher in disease            True
 std|centre 0.710 0.549 0.864 higher in disease            True
 idr|centre 0.695 0.527 0.854 higher in disease            True
kurt|centre 0.692 0.472 0.881 higher in disease           False
skew|middle 0.679 0.518 0.831 higher in disease            True
 Del|middle 0.321 0.150 0.497  lower in disease            True
  Del|outer 0.326 0.150 0.518  lower in disease           False
 kurt|outer 0.328 0.170 0.527  lower in disease           False
```

### intermediate (3 vs 1)
```
      feature   AUC    lo    hi         direction  ci_excludes_05
 median|outer 0.088 0.000 0.206  lower in disease            True
   mean|outer 0.127 0.010 0.277  lower in disease            True
median|middle 0.127 0.006 0.274  lower in disease            True
  mean|middle 0.135 0.006 0.297  lower in disease            True
   kurt|outer 0.712 0.517 0.894 higher in disease            True
  mean|centre 0.296 0.110 0.506  lower in disease           False
median|centre 0.296 0.090 0.503  lower in disease           False
  skew|middle 0.677 0.413 0.878 higher in disease           False
```

### advanced (4 vs 1)
```
      feature   AUC    lo    hi         direction  ci_excludes_05
 median|outer 0.044 0.000 0.148  lower in disease            True
   mean|outer 0.077 0.000 0.232  lower in disease            True
median|middle 0.077 0.000 0.198  lower in disease            True
  mean|middle 0.093 0.000 0.240  lower in disease            True
    iqr|outer 0.110 0.000 0.281  lower in disease            True
   kurt|outer 0.863 0.720 0.971 higher in disease            True
    idr|outer 0.181 0.010 0.413  lower in disease            True
median|centre 0.192 0.026 0.412  lower in disease            True
```

### advanced (3-4 vs 1)
```
      feature   AUC    lo    hi         direction  ci_excludes_05
 median|outer 0.070 0.000 0.176  lower in disease            True
   mean|outer 0.106 0.005 0.245  lower in disease            True
median|middle 0.106 0.009 0.237  lower in disease            True
  mean|middle 0.118 0.009 0.260  lower in disease            True
   kurt|outer 0.774 0.613 0.913 higher in disease            True
median|centre 0.253 0.080 0.452  lower in disease            True
  mean|centre 0.258 0.091 0.462  lower in disease            True
    iqr|outer 0.260 0.033 0.493  lower in disease            True
```

### any AMD (2-4 vs 1)
```
      feature   AUC    lo    hi         direction  ci_excludes_05
 median|outer 0.267 0.121 0.444  lower in disease            True
   mean|outer 0.292 0.142 0.460  lower in disease            True
median|middle 0.302 0.143 0.472  lower in disease            True
  mean|middle 0.314 0.149 0.484  lower in disease            True
  skew|middle 0.637 0.475 0.799 higher in disease           False
   skew|outer 0.635 0.466 0.789 higher in disease           False
  kurt|centre 0.633 0.462 0.789 higher in disease           False
    iqr|outer 0.393 0.222 0.583  lower in disease           False
```

### Significant direction reversals (early vs advanced, both CIs exclude 0.5, opposite sides): 0 of 27 feature x ring cells
```
none
```

## B. Positive control: OFA total deviations
```
             contrast            feature   AUC    lo    hi  ci_excludes_05
       early (2 vs 1)        Del mean TD 0.313 0.142 0.493            True
       early (2 vs 1) Del worst (max) TD 0.359 0.190 0.529           False
       early (2 vs 1)        Amp mean TD 0.559 0.332 0.763           False
       early (2 vs 1) Amp worst (min) TD 0.590 0.386 0.781           False
       early (2 vs 1) Del mean TD centre 0.328 0.166 0.506           False
       early (2 vs 1) Amp mean TD centre 0.579 0.384 0.764           False
       early (2 vs 1) Del mean TD middle 0.321 0.150 0.497            True
       early (2 vs 1) Amp mean TD middle 0.585 0.359 0.778           False
       early (2 vs 1)  Del mean TD outer 0.326 0.150 0.518           False
       early (2 vs 1)  Amp mean TD outer 0.531 0.296 0.740           False
intermediate (3 vs 1)        Del mean TD 0.573 0.316 0.815           False
intermediate (3 vs 1) Del worst (max) TD 0.612 0.376 0.838           False
intermediate (3 vs 1)        Amp mean TD 0.427 0.153 0.704           False
intermediate (3 vs 1) Amp worst (min) TD 0.458 0.246 0.679           False
intermediate (3 vs 1) Del mean TD centre 0.604 0.344 0.829           False
intermediate (3 vs 1) Amp mean TD centre 0.515 0.234 0.759           False
intermediate (3 vs 1) Del mean TD middle 0.581 0.315 0.819           False
intermediate (3 vs 1) Amp mean TD middle 0.419 0.134 0.696           False
intermediate (3 vs 1)  Del mean TD outer 0.565 0.316 0.805           False
intermediate (3 vs 1)  Amp mean TD outer 0.350 0.133 0.614           False
    advanced (4 vs 1)        Del mean TD 0.643 0.368 0.880           False
    advanced (4 vs 1) Del worst (max) TD 0.632 0.356 0.871           False
    advanced (4 vs 1)        Amp mean TD 0.341 0.101 0.648           False
    advanced (4 vs 1) Amp worst (min) TD 0.385 0.185 0.631           False
    advanced (4 vs 1) Del mean TD centre 0.659 0.376 0.905           False
    advanced (4 vs 1) Amp mean TD centre 0.396 0.143 0.673           False
    advanced (4 vs 1) Del mean TD middle 0.593 0.320 0.847           False
    advanced (4 vs 1) Amp mean TD middle 0.335 0.098 0.642           False
    advanced (4 vs 1)  Del mean TD outer 0.637 0.371 0.875           False
    advanced (4 vs 1)  Amp mean TD outer 0.335 0.090 0.628           False
  advanced (3-4 vs 1)        Del mean TD 0.602 0.345 0.830           False
  advanced (3-4 vs 1) Del worst (max) TD 0.620 0.388 0.835           False
  advanced (3-4 vs 1)        Amp mean TD 0.391 0.145 0.664           False
  advanced (3-4 vs 1) Amp worst (min) TD 0.428 0.242 0.643           False
  advanced (3-4 vs 1) Del mean TD centre 0.627 0.373 0.846           False
  advanced (3-4 vs 1) Amp mean TD centre 0.466 0.236 0.704           False
  advanced (3-4 vs 1) Del mean TD middle 0.586 0.328 0.824           False
  advanced (3-4 vs 1) Amp mean TD middle 0.385 0.136 0.652           False
  advanced (3-4 vs 1)  Del mean TD outer 0.595 0.346 0.822           False
  advanced (3-4 vs 1)  Amp mean TD outer 0.344 0.127 0.616           False
   any AMD (2-4 vs 1)        Del mean TD 0.466 0.283 0.662           False
   any AMD (2-4 vs 1) Del worst (max) TD 0.498 0.312 0.685           False
   any AMD (2-4 vs 1)        Amp mean TD 0.470 0.288 0.678           False
   any AMD (2-4 vs 1) Amp worst (min) TD 0.504 0.336 0.679           False
   any AMD (2-4 vs 1) Del mean TD centre 0.487 0.296 0.687           False
   any AMD (2-4 vs 1) Amp mean TD centre 0.519 0.361 0.691           False
   any AMD (2-4 vs 1) Del mean TD middle 0.462 0.272 0.657           False
   any AMD (2-4 vs 1) Amp mean TD middle 0.478 0.297 0.691           False
   any AMD (2-4 vs 1)  Del mean TD outer 0.469 0.287 0.662           False
   any AMD (2-4 vs 1)  Amp mean TD outer 0.431 0.248 0.635           False
```

## C. Test-retest reliability ICC(2,1)  (<0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent)
```
feature   ring  ICC_ring_mean  ICC_median_per_zone  fellow_eye_spearman
   mean centre          0.749                0.753                0.839
   mean middle          0.792                0.767                0.839
   mean  outer          0.736                0.610                0.839
 median centre          0.750                0.750                0.824
 median middle          0.795                0.764                0.824
 median  outer          0.713                0.601                0.824
    std centre          0.787                0.786                0.811
    std middle          0.515                0.528                0.811
    std  outer          0.463                0.483                0.811
    iqr centre          0.792                0.765                0.867
    iqr middle          0.518                0.504                0.867
    iqr  outer          0.515                0.507                0.867
    idr centre          0.788                0.789                0.773
    idr middle          0.521                0.529                0.773
    idr  outer          0.489                0.503                0.773
   skew centre          0.766                0.719                0.549
   skew middle          0.772                0.713                0.549
   skew  outer          0.617                0.589                0.549
   kurt centre          0.837                0.759                0.243
   kurt middle          0.538                0.582                0.243
   kurt  outer          0.567                0.542                0.243
    Del centre          0.851                0.697                0.954
    Del middle          0.868                0.670                0.954
    Del  outer          0.916                0.731                0.954
    Amp centre          0.722                0.621                0.926
    Amp middle          0.779                0.602                0.926
    Amp  outer          0.811                0.569                0.926
```

## 44-region OFA (wider field)
See D_ofa44_summary.csv and D_ofa44_regions.csv.


Figure: figures/A_stage_direction_heatmap.png
