# Age check (pre-specified: docs/ANALYSIS_PLAN_AGE.md)

## AUC by feature set
~~~
grouping     task model      set   AUC    lo    hi     p
     eye advanced    lr     MPOD 0.910 0.829 0.969 0.005
     eye advanced    lr MPOD+age 0.908 0.825 0.967   NaN
     eye advanced    lr      age 0.875 0.756 0.960 0.010
     eye advanced    lr      all 0.886 0.803 0.947 0.005
     eye advanced    lr  all+age 0.900 0.820 0.958   NaN
     eye advanced    rf     MPOD 0.943 0.879 0.984   NaN
     eye advanced    rf MPOD+age 0.946 0.885 0.985   NaN
     eye advanced    rf      age 0.872 0.761 0.961   NaN
     eye advanced    rf      all 0.938 0.872 0.982 0.005
     eye advanced    rf  all+age 0.944 0.883 0.983   NaN
     eye      any    lr     MPOD 0.723 0.607 0.827 0.010
     eye      any    lr MPOD+age 0.725 0.609 0.831   NaN
     eye      any    lr      age 0.788 0.665 0.899 0.005
     eye      any    lr      all 0.672 0.553 0.777 0.114
     eye      any    lr  all+age 0.691 0.572 0.797   NaN
     eye      any    rf     MPOD 0.733 0.610 0.842   NaN
     eye      any    rf MPOD+age 0.743 0.620 0.852   NaN
     eye      any    rf      age 0.733 0.613 0.841   NaN
     eye      any    rf      all 0.714 0.591 0.826 0.020
     eye      any    rf  all+age 0.731 0.606 0.844   NaN
     eye    early    lr     MPOD 0.538 0.407 0.668 0.458
     eye    early    lr MPOD+age 0.527 0.394 0.658   NaN
     eye    early    lr      age 0.677 0.491 0.830 0.095
     eye    early    lr      all 0.581 0.422 0.731 0.428
     eye    early    lr  all+age 0.579 0.420 0.727   NaN
     eye    early    rf     MPOD 0.576 0.424 0.716   NaN
     eye    early    rf MPOD+age 0.581 0.431 0.718   NaN
     eye    early    rf      age 0.578 0.398 0.762   NaN
     eye    early    rf      all 0.589 0.416 0.752 0.234
     eye    early    rf  all+age 0.597 0.423 0.758   NaN
 subject advanced    lr     MPOD 0.881 0.791 0.947 0.005
 subject advanced    lr MPOD+age 0.885 0.796 0.953   NaN
 subject advanced    lr      age 0.860 0.736 0.952 0.015
 subject advanced    lr      all 0.827 0.721 0.909 0.015
 subject advanced    lr  all+age 0.832 0.723 0.918   NaN
 subject advanced    rf     MPOD 0.905 0.818 0.969   NaN
 subject advanced    rf MPOD+age 0.911 0.828 0.969   NaN
 subject advanced    rf      age 0.752 0.609 0.881   NaN
 subject advanced    rf      all 0.893 0.799 0.963 0.005
 subject advanced    rf  all+age 0.905 0.819 0.966   NaN
 subject      any    lr     MPOD 0.671 0.547 0.782 0.030
 subject      any    lr MPOD+age 0.696 0.568 0.812   NaN
 subject      any    lr      age 0.784 0.658 0.896 0.010
 subject      any    lr      all 0.632 0.510 0.745 0.040
 subject      any    lr  all+age 0.649 0.527 0.761   NaN
 subject      any    rf     MPOD 0.685 0.564 0.794   NaN
 subject      any    rf MPOD+age 0.688 0.568 0.800   NaN
 subject      any    rf      age 0.668 0.539 0.789   NaN
 subject      any    rf      all 0.658 0.537 0.771 0.040
 subject      any    rf  all+age 0.681 0.560 0.795   NaN
 subject    early    lr     MPOD 0.455 0.321 0.593 0.532
 subject    early    lr MPOD+age 0.462 0.319 0.609   NaN
 subject    early    lr      age 0.679 0.496 0.834 0.075
 subject    early    lr      all 0.491 0.349 0.636 0.259
 subject    early    lr  all+age 0.497 0.356 0.639   NaN
 subject    early    rf     MPOD 0.457 0.314 0.597   NaN
 subject    early    rf MPOD+age 0.478 0.334 0.621   NaN
 subject    early    rf      age 0.542 0.373 0.720   NaN
 subject    early    rf      all 0.502 0.344 0.661 0.184
 subject    early    rf  all+age 0.502 0.341 0.664   NaN
~~~

## Paired comparisons (dAUC = A - B; 95% CI eye-clustered bootstrap)
~~~
    task model                                         comparison  AUC_A  AUC_B   dAUC     lo     hi             conclusion
advanced    rf [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.911  0.752  0.159  0.021  0.311               A better
advanced    rf                         [subject] MPOD+age vs MPOD  0.911  0.905  0.006 -0.006  0.019 no difference detected
advanced    rf                              [subject] MPOD vs age  0.905  0.752  0.153  0.005  0.312               A better
advanced    rf                           [subject] all+age vs age  0.905  0.752  0.153  0.013  0.307               A better
advanced    rf                           [subject] all+age vs all  0.905  0.893  0.013 -0.002  0.030 no difference detected
advanced    lr [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.885  0.860  0.025 -0.067  0.132 no difference detected
advanced    lr                         [subject] MPOD+age vs MPOD  0.885  0.881  0.005 -0.016  0.024 no difference detected
advanced    lr                              [subject] MPOD vs age  0.881  0.860  0.020 -0.080  0.137 no difference detected
advanced    lr                           [subject] all+age vs age  0.832  0.860 -0.028 -0.136  0.089 no difference detected
advanced    lr                           [subject] all+age vs all  0.832  0.827  0.005 -0.021  0.030 no difference detected
advanced    rf     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.946  0.872  0.074 -0.023  0.191 no difference detected
advanced    rf                             [eye] MPOD+age vs MPOD  0.946  0.943  0.003 -0.004  0.011 no difference detected
advanced    rf                                  [eye] MPOD vs age  0.943  0.872  0.071 -0.031  0.190 no difference detected
advanced    rf                               [eye] all+age vs age  0.944  0.872  0.072 -0.025  0.187 no difference detected
advanced    rf                               [eye] all+age vs all  0.944  0.938  0.006 -0.001  0.015 no difference detected
advanced    lr     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.908  0.875  0.033 -0.050  0.136 no difference detected
advanced    lr                             [eye] MPOD+age vs MPOD  0.908  0.910 -0.002 -0.021  0.020 no difference detected
advanced    lr                                  [eye] MPOD vs age  0.910  0.875  0.035 -0.059  0.144 no difference detected
advanced    lr                               [eye] all+age vs age  0.900  0.875  0.025 -0.060  0.125 no difference detected
advanced    lr                               [eye] all+age vs all  0.900  0.886  0.014 -0.003  0.032 no difference detected
     any    rf [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.688  0.668  0.020 -0.107  0.149 no difference detected
     any    rf                         [subject] MPOD+age vs MPOD  0.688  0.685  0.003 -0.021  0.029 no difference detected
     any    rf                              [subject] MPOD vs age  0.685  0.668  0.017 -0.129  0.158 no difference detected
     any    rf                           [subject] all+age vs age  0.681  0.668  0.014 -0.118  0.148 no difference detected
     any    rf                           [subject] all+age vs all  0.681  0.658  0.023  0.001  0.047               A better
     any    lr [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.696  0.784 -0.088 -0.196  0.010 no difference detected
     any    lr                         [subject] MPOD+age vs MPOD  0.696  0.671  0.026  0.007  0.046               A better
     any    lr                              [subject] MPOD vs age  0.671  0.784 -0.114 -0.231 -0.007               B better
     any    lr                           [subject] all+age vs age  0.649  0.784 -0.136 -0.250 -0.022               B better
     any    lr                           [subject] all+age vs all  0.649  0.632  0.017  0.001  0.035               A better
     any    rf     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.743  0.733  0.010 -0.103  0.123 no difference detected
     any    rf                             [eye] MPOD+age vs MPOD  0.743  0.733  0.010 -0.005  0.027 no difference detected
     any    rf                                  [eye] MPOD vs age  0.733  0.733 -0.000 -0.124  0.120 no difference detected
     any    rf                               [eye] all+age vs age  0.731  0.733 -0.002 -0.116  0.114 no difference detected
     any    rf                               [eye] all+age vs all  0.731  0.714  0.018  0.001  0.037               A better
     any    lr     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.725  0.788 -0.063 -0.160  0.028 no difference detected
     any    lr                             [eye] MPOD+age vs MPOD  0.725  0.723  0.003 -0.013  0.020 no difference detected
     any    lr                                  [eye] MPOD vs age  0.723  0.788 -0.066 -0.174  0.037 no difference detected
     any    lr                               [eye] all+age vs age  0.691  0.788 -0.098 -0.197  0.002 no difference detected
     any    lr                               [eye] all+age vs all  0.691  0.672  0.019  0.000  0.039               A better
   early    rf [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.478  0.542 -0.064 -0.299  0.166 no difference detected
   early    rf                         [subject] MPOD+age vs MPOD  0.478  0.457  0.021  0.001  0.042               A better
   early    rf                              [subject] MPOD vs age  0.457  0.542 -0.085 -0.327  0.154 no difference detected
   early    rf                           [subject] all+age vs age  0.502  0.542 -0.040 -0.269  0.191 no difference detected
   early    rf                           [subject] all+age vs all  0.502  0.502  0.000 -0.014  0.014 no difference detected
   early    lr [subject] MPOD+age vs age (PRIMARY if subject, RF)  0.462  0.679 -0.217 -0.387 -0.039               B better
   early    lr                         [subject] MPOD+age vs MPOD  0.462  0.455  0.007 -0.021  0.033 no difference detected
   early    lr                              [subject] MPOD vs age  0.455  0.679 -0.225 -0.401 -0.034               B better
   early    lr                           [subject] all+age vs age  0.497  0.679 -0.182 -0.351 -0.009               B better
   early    lr                           [subject] all+age vs all  0.497  0.491  0.006 -0.008  0.021 no difference detected
   early    rf     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.581  0.578  0.003 -0.226  0.220 no difference detected
   early    rf                             [eye] MPOD+age vs MPOD  0.581  0.576  0.006 -0.016  0.026 no difference detected
   early    rf                                  [eye] MPOD vs age  0.576  0.578 -0.002 -0.238  0.224 no difference detected
   early    rf                               [eye] all+age vs age  0.597  0.578  0.019 -0.205  0.244 no difference detected
   early    rf                               [eye] all+age vs all  0.597  0.589  0.008 -0.007  0.024 no difference detected
   early    lr     [eye] MPOD+age vs age (PRIMARY if subject, RF)  0.527  0.677 -0.149 -0.319  0.030 no difference detected
   early    lr                             [eye] MPOD+age vs MPOD  0.527  0.538 -0.011 -0.031  0.009 no difference detected
   early    lr                                  [eye] MPOD vs age  0.538  0.677 -0.139 -0.315  0.047 no difference detected
   early    lr                               [eye] all+age vs age  0.579  0.677 -0.097 -0.265  0.082 no difference detected
   early    lr                               [eye] all+age vs all  0.579  0.581 -0.001 -0.012  0.010 no difference detected
~~~

## PRIMARY (RF, participant-grouped, MPOD+age vs age)
~~~
    task   dAUC     lo    hi             conclusion
advanced  0.159  0.021 0.311               A better
     any  0.020 -0.107 0.149 no difference detected
   early -0.064 -0.299 0.166 no difference detected
~~~
