# Pre-specified analysis plan: value of objective perimetry (OFA) in combination with MPOD

**Status:** written 10 Oct 2026, before running analyses A-D on the corrected data (`docs/DATA_ISSUE_2026-10.md`).
**Code:** `scripts/9_ofa_value.sh` (A-D), `scripts/10_explain.sh` (SHAP). Commit hash of the run is recorded in `share/run_*/env/commit.txt`.

## Common design (unchanged from the main analysis)
- Data: 116 eye-sessions, 58 eyes, 29 participants; features = 7 MPOD statistics + OFA total deviations (Del, Amp), 20 regions each.
- Tasks: **any** (AREDS 2-4 vs 1), **early** (2 vs 1), **advanced** (3-4 vs 1).
- Evaluation: repeated (10x) nested 5-fold cross-validation, eye-grouped; scaling/imputation fitted in-fold;
  identical outer splits for every feature set, so comparisons are paired eye by eye.
- Models: logistic regression (L2) and random forest; hyperparameters tuned only on training folds (inner 3-fold, AUC).
- Metric: AUC (primary). Uncertainty: eye-clustered bootstrap 95% CI. Single-set significance: label-permutation test
  (block scheme, 200 permutations, LR). Set-vs-set: paired dAUC with eye-clustered bootstrap CI.

## Questions and analyses
| | Question | Analysis | Primary comparison |
|---|---|---|---|
| **A** | Does OFA add information to MPOD? | MPOD; MPOD+Del; MPOD+Amp; MPOD+Del+Amp | dAUC (MPOD+Del+Amp) - (MPOD), RF, tasks any and advanced |
| **B** | Which OFA measure contributes, Del or Amp? | Leave-one-group-out for all 9 groups | dAUC (all) - (all minus group) |
| **C** | Does structure-function coupling carry stage information? | Per eye-session Spearman rho across zones between each MPOD statistic and Del / Amp (14 features); coupling alone, MPOD+coupling, all+coupling | dAUC (MPOD+coupling) - (MPOD) |
| **D** | Is the functional signal in early AMD? | Del, Amp, Del+Amp alone in the *early* task; univariate stage-direction maps (`scripts/8_diagnostics.sh`) | permutation p of Del, Amp |

## Decision rules (fixed in advance)
- "Adds information": paired dAUC 95% CI excludes 0. Otherwise: "no difference detected" (not "no effect").
- Single-set signal: permutation p < 0.05 **and** AUC >= 0.65.
- Many comparisons are made (3 tasks x 2 models x 16 contrasts). A (RF, any/advanced) is primary; B-D are exploratory and
  reported with CIs, without multiplicity-adjusted claims.

## Reporting commitments
- Every pre-specified comparison above is reported (main text or supplement), whatever its direction.
- The paper does not make claims beyond what the CIs support, in either direction.
- Manuscript 1 (ML: Health): A and B (replacing the earlier feature-group ablation), SHAP from `scripts/10_explain.sh`.
- Short paper (MedInfo): C and D, plus test-retest reliability and 44-region OFA, with no overlap of reported results.
