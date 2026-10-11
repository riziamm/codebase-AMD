# Pre-specified analysis plan: age as a confounder (Paper 1)

**Status:** written 11 Oct 2026, before running. Trigger: AREDS 1 eyes are younger (median 68.5 y) than AREDS 3-4 eyes (75-76 y),
and age alone reached AUC 0.80 (any AMD) in the Paper 2 exploration. Code: `scripts/12_age_check.sh`.

## Question
Does MPOD add discriminative information **beyond age**?

## Design (identical to Paper 1)
Repeated nested CV (10 x 5-fold, same seeds and splits), LR and RF, tasks any / early / advanced.
Feature sets: age; MPOD (7 statistics); MPOD + age; MPOD + OFA (all); all + age. Sex is not used (descriptive only).
**Grouping:** participant-grouped is **primary** here, because age is a participant attribute and, in eye-grouped CV,
the fellow eye in training can let age act as a participant fingerprint. Eye-grouped results are reported as secondary.

## Primary comparison and decision rule
- Primary: paired dAUC = (MPOD + age) - (age), RF, participant-grouped, tasks advanced and any.
- "MPOD adds beyond age" if the 95% eye-clustered bootstrap CI excludes 0; otherwise "not shown in this cohort".
- Secondary: (MPOD + age) - (MPOD) (does age add?), (MPOD) - (age) head-to-head, (all + age) - (all); LR; eye-grouped; early task.

## Reporting commitment
All comparisons are reported (Paper 1 Table 3 row "Age only", Table A6 for the comparisons, Limitations).
- If MPOD adds beyond age: the main claims stand as "beyond age".
- If not: Paper 1 states that discrimination could not be separated from age in this non-age-matched cohort,
  and the ring/SHAP findings are presented as descriptive of the MPOD pattern, not as evidence of stage information beyond age.
