# OFA value analyses (pre-specified; docs/ANALYSIS_PLAN_OFA.md)

dAUC = A - B, paired on the same eyes and CV splits; 95% CI = eye-clustered bootstrap.

## task: any
~~~
model                                 comparison  AUC_A  AUC_B   dAUC     lo     hi             conclusion
   lr                        A: MPOD+Del vs MPOD  0.702  0.723 -0.021 -0.057  0.011 no difference detected
   lr                        A: MPOD+Amp vs MPOD  0.683  0.723 -0.040 -0.081  0.002 no difference detected
   lr                    A: MPOD+Del+Amp vs MPOD  0.672  0.723 -0.051 -0.104 -0.000               B better
   lr                  A: adding Amp to MPOD+Del  0.672  0.702 -0.030 -0.081  0.015 no difference detected
   lr                  A: adding Del to MPOD+Amp  0.672  0.683 -0.011 -0.049  0.024 no difference detected
   lr                   C: MPOD+coupling vs MPOD  0.716  0.723 -0.006 -0.031  0.020 no difference detected
   lr                     C: all+coupling vs all  0.676  0.672  0.004 -0.020  0.030 no difference detected
   lr     B: importance of mean (all minus mean)  0.672  0.670  0.001 -0.022  0.023 no difference detected
   lr B: importance of median (all minus median)  0.672  0.673 -0.001 -0.019  0.017 no difference detected
   lr       B: importance of std (all minus std)  0.672  0.651  0.021  0.002  0.041               A better
   lr       B: importance of iqr (all minus iqr)  0.672  0.663  0.009 -0.010  0.029 no difference detected
   lr       B: importance of idr (all minus idr)  0.672  0.655  0.016  0.001  0.034               A better
   lr     B: importance of skew (all minus skew)  0.672  0.683 -0.011 -0.038  0.016 no difference detected
   lr     B: importance of kurt (all minus kurt)  0.672  0.697 -0.026 -0.058  0.005 no difference detected
   lr       B: importance of Del (all minus Del)  0.672  0.683 -0.011 -0.049  0.024 no difference detected
   lr       B: importance of Amp (all minus Amp)  0.672  0.702 -0.030 -0.081  0.015 no difference detected
   rf                        A: MPOD+Del vs MPOD  0.726  0.733 -0.007 -0.023  0.010 no difference detected
   rf                        A: MPOD+Amp vs MPOD  0.723  0.733 -0.009 -0.023  0.004 no difference detected
   rf                    A: MPOD+Del+Amp vs MPOD  0.714  0.733 -0.019 -0.041  0.001 no difference detected
   rf                  A: adding Amp to MPOD+Del  0.714  0.726 -0.013 -0.028  0.002 no difference detected
   rf                  A: adding Del to MPOD+Amp  0.714  0.723 -0.010 -0.028  0.007 no difference detected
   rf                   C: MPOD+coupling vs MPOD  0.729  0.733 -0.004 -0.015  0.008 no difference detected
   rf                     C: all+coupling vs all  0.716  0.714  0.002 -0.014  0.019 no difference detected
   rf     B: importance of mean (all minus mean)  0.714  0.714  0.000 -0.018  0.018 no difference detected
   rf B: importance of median (all minus median)  0.714  0.712  0.002 -0.017  0.020 no difference detected
   rf       B: importance of std (all minus std)  0.714  0.706  0.008 -0.007  0.027 no difference detected
   rf       B: importance of iqr (all minus iqr)  0.714  0.714 -0.000 -0.013  0.014 no difference detected
   rf       B: importance of idr (all minus idr)  0.714  0.709  0.005 -0.009  0.020 no difference detected
   rf     B: importance of skew (all minus skew)  0.714  0.709  0.005 -0.015  0.026 no difference detected
   rf     B: importance of kurt (all minus kurt)  0.714  0.721 -0.007 -0.024  0.008 no difference detected
   rf       B: importance of Del (all minus Del)  0.714  0.723 -0.010 -0.028  0.007 no difference detected
   rf       B: importance of Amp (all minus Amp)  0.714  0.726 -0.013 -0.028  0.002 no difference detected
~~~

## task: early
~~~
model                                 comparison  AUC_A  AUC_B   dAUC     lo     hi             conclusion
   lr                        A: MPOD+Del vs MPOD  0.617  0.538  0.079 -0.009  0.161 no difference detected
   lr                        A: MPOD+Amp vs MPOD  0.499  0.538 -0.039 -0.076 -0.004               B better
   lr                    A: MPOD+Del+Amp vs MPOD  0.581  0.538  0.043 -0.044  0.131 no difference detected
   lr                  A: adding Amp to MPOD+Del  0.581  0.617 -0.037 -0.062 -0.012               B better
   lr                  A: adding Del to MPOD+Amp  0.581  0.499  0.081 -0.007  0.169 no difference detected
   lr                   C: MPOD+coupling vs MPOD  0.546  0.538  0.008 -0.037  0.056 no difference detected
   lr                     C: all+coupling vs all  0.586  0.581  0.006 -0.026  0.038 no difference detected
   lr     B: importance of mean (all minus mean)  0.581  0.609 -0.028 -0.045 -0.011               B better
   lr B: importance of median (all minus median)  0.581  0.610 -0.029 -0.048 -0.012               B better
   lr       B: importance of std (all minus std)  0.581  0.577  0.004 -0.012  0.021 no difference detected
   lr       B: importance of iqr (all minus iqr)  0.581  0.590 -0.009 -0.026  0.007 no difference detected
   lr       B: importance of idr (all minus idr)  0.581  0.579  0.002 -0.010  0.014 no difference detected
   lr     B: importance of skew (all minus skew)  0.581  0.532  0.049  0.015  0.085               A better
   lr     B: importance of kurt (all minus kurt)  0.581  0.584 -0.003 -0.037  0.026 no difference detected
   lr       B: importance of Del (all minus Del)  0.581  0.499  0.081 -0.007  0.169 no difference detected
   lr       B: importance of Amp (all minus Amp)  0.581  0.617 -0.037 -0.062 -0.012               B better
   rf                        A: MPOD+Del vs MPOD  0.616  0.576  0.041 -0.059  0.132 no difference detected
   rf                        A: MPOD+Amp vs MPOD  0.537  0.576 -0.038 -0.076 -0.007               B better
   rf                    A: MPOD+Del+Amp vs MPOD  0.589  0.576  0.013 -0.086  0.108 no difference detected
   rf                  A: adding Amp to MPOD+Del  0.589  0.616 -0.028 -0.056 -0.002               B better
   rf                  A: adding Del to MPOD+Amp  0.589  0.537  0.051 -0.038  0.138 no difference detected
   rf                   C: MPOD+coupling vs MPOD  0.585  0.576  0.009 -0.021  0.040 no difference detected
   rf                     C: all+coupling vs all  0.602  0.589  0.013 -0.012  0.040 no difference detected
   rf     B: importance of mean (all minus mean)  0.589  0.605 -0.017 -0.035  0.002 no difference detected
   rf B: importance of median (all minus median)  0.589  0.605 -0.016 -0.032 -0.001               B better
   rf       B: importance of std (all minus std)  0.589  0.593 -0.004 -0.030  0.023 no difference detected
   rf       B: importance of iqr (all minus iqr)  0.589  0.606 -0.018 -0.040  0.002 no difference detected
   rf       B: importance of idr (all minus idr)  0.589  0.594 -0.005 -0.031  0.021 no difference detected
   rf     B: importance of skew (all minus skew)  0.589  0.578  0.010 -0.037  0.050 no difference detected
   rf     B: importance of kurt (all minus kurt)  0.589  0.580  0.009 -0.028  0.044 no difference detected
   rf       B: importance of Del (all minus Del)  0.589  0.537  0.051 -0.038  0.138 no difference detected
   rf       B: importance of Amp (all minus Amp)  0.589  0.616 -0.028 -0.056 -0.002               B better
~~~

## task: advanced
~~~
model                                 comparison  AUC_A  AUC_B   dAUC     lo     hi             conclusion
   lr                        A: MPOD+Del vs MPOD  0.883  0.910 -0.026 -0.060 -0.002               B better
   lr                        A: MPOD+Amp vs MPOD  0.911  0.910  0.001 -0.032  0.031 no difference detected
   lr                    A: MPOD+Del+Amp vs MPOD  0.886  0.910 -0.024 -0.063  0.014 no difference detected
   lr                  A: adding Amp to MPOD+Del  0.886  0.883  0.003 -0.025  0.029 no difference detected
   lr                  A: adding Del to MPOD+Amp  0.886  0.911 -0.025 -0.044 -0.008               B better
   lr                   C: MPOD+coupling vs MPOD  0.906  0.910 -0.003 -0.038  0.022 no difference detected
   lr                     C: all+coupling vs all  0.880  0.886 -0.006 -0.038  0.016 no difference detected
   lr     B: importance of mean (all minus mean)  0.886  0.886 -0.000 -0.014  0.014 no difference detected
   lr B: importance of median (all minus median)  0.886  0.889 -0.003 -0.016  0.008 no difference detected
   lr       B: importance of std (all minus std)  0.886  0.884  0.002 -0.005  0.011 no difference detected
   lr       B: importance of iqr (all minus iqr)  0.886  0.885  0.001 -0.007  0.010 no difference detected
   lr       B: importance of idr (all minus idr)  0.886  0.887 -0.001 -0.008  0.006 no difference detected
   lr     B: importance of skew (all minus skew)  0.886  0.883  0.003 -0.017  0.024 no difference detected
   lr     B: importance of kurt (all minus kurt)  0.886  0.880  0.006 -0.014  0.027 no difference detected
   lr       B: importance of Del (all minus Del)  0.886  0.911 -0.025 -0.044 -0.008               B better
   lr       B: importance of Amp (all minus Amp)  0.886  0.883  0.003 -0.025  0.029 no difference detected
   rf                        A: MPOD+Del vs MPOD  0.939  0.943 -0.004 -0.009  0.001 no difference detected
   rf                        A: MPOD+Amp vs MPOD  0.942  0.943 -0.001 -0.007  0.005 no difference detected
   rf                    A: MPOD+Del+Amp vs MPOD  0.938  0.943 -0.005 -0.014  0.003 no difference detected
   rf                  A: adding Amp to MPOD+Del  0.938  0.939 -0.001 -0.008  0.004 no difference detected
   rf                  A: adding Del to MPOD+Amp  0.938  0.942 -0.004 -0.010  0.000 no difference detected
   rf                   C: MPOD+coupling vs MPOD  0.941  0.943 -0.002 -0.008  0.004 no difference detected
   rf                     C: all+coupling vs all  0.939  0.938  0.001 -0.003  0.007 no difference detected
   rf     B: importance of mean (all minus mean)  0.938  0.945 -0.007 -0.016  0.000 no difference detected
   rf B: importance of median (all minus median)  0.938  0.941 -0.003 -0.012  0.005 no difference detected
   rf       B: importance of std (all minus std)  0.938  0.939 -0.001 -0.007  0.004 no difference detected
   rf       B: importance of iqr (all minus iqr)  0.938  0.938 -0.000 -0.007  0.006 no difference detected
   rf       B: importance of idr (all minus idr)  0.938  0.941 -0.003 -0.009  0.002 no difference detected
   rf     B: importance of skew (all minus skew)  0.938  0.938  0.000 -0.010  0.010 no difference detected
   rf     B: importance of kurt (all minus kurt)  0.938  0.924  0.014  0.000  0.032               A better
   rf       B: importance of Del (all minus Del)  0.938  0.942 -0.004 -0.010  0.000 no difference detected
   rf       B: importance of Amp (all minus Amp)  0.938  0.939 -0.001 -0.008  0.004 no difference detected
~~~

## Single-modality and baseline sets (permutation p from LR)
~~~
    task                            groups       model   AUC  AUC_lo  AUC_hi  p_above                   verdict
advanced                               Amp          lr 0.591   0.487   0.701    0.239      NO DETECTABLE SIGNAL
advanced                               Amp          rf 0.548   0.411   0.686      NaN n/a (no permutation test)
advanced                               Del          lr 0.496   0.373   0.612    0.448      NO DETECTABLE SIGNAL
advanced                               Del          rf 0.503   0.343   0.657      NaN n/a (no permutation test)
advanced                           Del,Amp          lr 0.559   0.447   0.669    0.383      NO DETECTABLE SIGNAL
advanced                           Del,Amp          rf 0.550   0.382   0.711      NaN n/a (no permutation test)
advanced                               all hybrid_orig 0.907   0.836   0.958    0.010                    SIGNAL
advanced                               all hybrid_orig 0.839   0.751   0.913    0.020                    SIGNAL
advanced                               all hybrid_zone 0.891   0.806   0.955    0.020                    SIGNAL
advanced                               all hybrid_zone 0.851   0.755   0.926    0.010                    SIGNAL
advanced                               all          lr 0.886   0.803   0.947    0.005                    SIGNAL
advanced                               all          lr 0.827   0.721   0.909    0.015                    SIGNAL
advanced                               all          rf 0.938   0.872   0.982    0.005                    SIGNAL
advanced                               all          rf 0.893   0.799   0.963    0.005                    SIGNAL
advanced                          coupling          lr 0.574   0.443   0.700    0.493      NO DETECTABLE SIGNAL
advanced                          coupling          rf 0.624   0.509   0.735      NaN n/a (no permutation test)
advanced mean,median,std,iqr,idr,skew,kurt          lr 0.910   0.829   0.969    0.005                    SIGNAL
advanced mean,median,std,iqr,idr,skew,kurt          rf 0.943   0.879   0.984      NaN n/a (no permutation test)
     any                               Amp          lr 0.461   0.372   0.550    0.910      NO DETECTABLE SIGNAL
     any                               Amp          rf 0.374   0.287   0.467      NaN n/a (no permutation test)
     any                               Del          lr 0.509   0.415   0.606    0.488      NO DETECTABLE SIGNAL
     any                               Del          rf 0.457   0.356   0.561      NaN n/a (no permutation test)
     any                           Del,Amp          lr 0.463   0.378   0.547    0.831      NO DETECTABLE SIGNAL
     any                           Del,Amp          rf 0.367   0.271   0.473      NaN n/a (no permutation test)
     any                               all hybrid_orig 0.665   0.543   0.774    0.178      NO DETECTABLE SIGNAL
     any                               all hybrid_orig 0.594   0.473   0.703    0.059      NO DETECTABLE SIGNAL
     any                               all hybrid_zone 0.582   0.482   0.674    0.505      NO DETECTABLE SIGNAL
     any                               all hybrid_zone 0.569   0.467   0.667    0.069      NO DETECTABLE SIGNAL
     any                               all          lr 0.672   0.553   0.777    0.114      NO DETECTABLE SIGNAL
     any                               all          lr 0.632   0.510   0.745    0.040      NO DETECTABLE SIGNAL
     any                               all          rf 0.714   0.591   0.826    0.020                    SIGNAL
     any                               all          rf 0.658   0.537   0.771    0.040                    SIGNAL
     any                          coupling          lr 0.630   0.516   0.742    0.149      NO DETECTABLE SIGNAL
     any                          coupling          rf 0.603   0.482   0.712      NaN n/a (no permutation test)
     any mean,median,std,iqr,idr,skew,kurt          lr 0.723   0.607   0.827    0.010                    SIGNAL
     any mean,median,std,iqr,idr,skew,kurt          rf 0.733   0.610   0.842      NaN n/a (no permutation test)
   early                               Amp          lr 0.391   0.285   0.500    0.980              BELOW CHANCE
   early                               Amp          rf 0.296   0.193   0.411      NaN n/a (no permutation test)
   early                               Del          lr 0.638   0.475   0.786    0.149      NO DETECTABLE SIGNAL
   early                               Del          rf 0.557   0.405   0.713      NaN n/a (no permutation test)
   early                           Del,Amp          lr 0.584   0.420   0.740    0.328      NO DETECTABLE SIGNAL
   early                           Del,Amp          rf 0.516   0.363   0.673      NaN n/a (no permutation test)
   early                               all hybrid_orig 0.510   0.358   0.651    0.455      NO DETECTABLE SIGNAL
   early                               all hybrid_orig 0.403   0.271   0.537    0.673      NO DETECTABLE SIGNAL
   early                               all hybrid_zone 0.525   0.393   0.649    0.465      NO DETECTABLE SIGNAL
   early                               all hybrid_zone 0.432   0.313   0.555    0.693      NO DETECTABLE SIGNAL
   early                               all          lr 0.581   0.422   0.731    0.428      NO DETECTABLE SIGNAL
   early                               all          lr 0.491   0.349   0.636    0.259      NO DETECTABLE SIGNAL
   early                               all          rf 0.589   0.416   0.752    0.234      NO DETECTABLE SIGNAL
   early                               all          rf 0.502   0.344   0.661    0.184      NO DETECTABLE SIGNAL
   early                          coupling          lr 0.604   0.459   0.735    0.299      NO DETECTABLE SIGNAL
   early                          coupling          rf 0.558   0.425   0.688      NaN n/a (no permutation test)
   early mean,median,std,iqr,idr,skew,kurt          lr 0.538   0.407   0.668    0.458      NO DETECTABLE SIGNAL
   early mean,median,std,iqr,idr,skew,kurt          rf 0.576   0.424   0.716      NaN n/a (no permutation test)
~~~
