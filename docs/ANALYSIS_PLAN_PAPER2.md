# Pre-specified analysis plan, Paper 2 (short paper)

**Status: DRAFT, not signed off.** The P1/P2 code refuses to run on real data until this line reads
`Status: SIGNED OFF (date, names)`. Step S0 (descriptive) may run before sign-off.
**Written:** 11 Oct 2026, branch `AMD-P2`. **Sign-off:** ____________ (all co-authors, date)

Working title: *Test-retest reliability and functional validity of objective multifocal pupillographic
perimetry (OFA) in age-related macular degeneration.*
Target: IEEE EMBC 2027 (full paper; deadline reported as 24 Jan 2027, **verify on embc.embs.org**).
Paper 1 (submitted, *Machine Learning: Health*) is cited as "submitted"; none of its results are re-reported.

## 0. Transparency: what has already been seen
| Item | Seen before this plan? | Consequence |
|---|---|---|
| Visual acuity (VA, VAlc), age, sex vs any feature | **No** (never exported or analysed) | P1 is genuinely confirmatory |
| ICC point estimates, 20-region ring means (`share/run_20261011/diagnostics`) | Yes | P2 is **estimation**, not a test; paper states the point estimates were inspected |
| 44-region AUCs vs AREDS grade | Yes | not reused as a question |
| Coupling (C) and early-AMD OFA (D) vs AREDS grade | Yes, run 20261011 | reported as *previously pre-specified* (section 5), no new claims |
| Earlier DL results (F1 ~0.80) | Yes | **not used**: row-level splits put both sessions of an eye in train and test (leakage) and used the scrambled CSV |

## 1. Data and units
58 eyes (29 participants; AREDS 1/2/3/4 = 26/15/10/7 eyes), two test sessions per eye. Source: verified export
(`src.mat_to_csv`, max difference 0 vs MATLAB). Never `reruns_SCRAMBLED_archive/`.
Eye-level features = mean of the two sessions (except P2, which compares sessions).
Ring summaries: mean over zones of each ring (`d.p.ring`: centre 1-4, middle 5-12, outer 13-20).

**To fix before sign-off**
- [ ] VA and VAlc units (logMAR assumed: higher = worse) and the visit at which they were measured
      relative to the OFA/MPOD sessions.
- [ ] 44-region layout: region -> eccentricity band file `ofa44_bands.csv` (from device documentation).
      If unavailable, 44-region analyses use whole-field mean and worst value only.
- [ ] Confirm that no reliability result appears in Paper 1 (main text or supplement).

## 2. Step S0: baseline and safety checks (runs first; descriptive)
- Table 1 by AREDS grade: eyes, subjects, age, sex, VA, VAlc (median [IQR]), missingness.
- **Age-confound check for Paper 1:** AUC of age alone for any / early / advanced AMD, subject-clustered bootstrap.
  Rule: AUC >= 0.70 with CI excluding 0.5 -> **FLAG** to Paper 1 authors before its review returns; CI excluding 0.5
  but AUC < 0.70 -> note. Not a Paper 2 result.
- Age vs OFA total deviation (and MPOD mean) in AREDS 1 eyes (TD is normatively age-corrected; a residual
  correlation would argue for age adjustment).
- Counts of `VisitN` / `VisitLast` (decides whether a longitudinal paper is feasible; no analysis here).
- S0 contains no association between predictors and acuity.

## 3. P1 (primary, confirmatory): do OFA measures explain acuity beyond age, sex and MPOD?
**Hypothesis H1.** Adding OFA (Del, Amp; 3 ring means each) to age + sex + MPOD (7 statistics x 3 ring means)
increases out-of-fold R^2 for **low-contrast visual acuity (VAlc)**.
Rationale: AREDS grade is structural; VAlc is a functional outcome less affected by ceiling than high-contrast VA.

**Design.** Repeated (R = 20) 5-fold cross-validation, **subject-grouped** (age and sex are subject-level; fellow
eyes share acuity determinants), stratified on worst-eye AREDS {1, 2, 3+}. Identical splits for every feature set
of an outcome (paired). Ridge regression (alpha tuned by inner 3-fold subject-grouped CV); imputation and scaling
fitted within training folds. Seed 2027. No deep learning (58 eyes).
**Metric.** Out-of-fold R^2 averaged over repeats (primary); Spearman rho and MAE (secondary).
**Inference.** Subject-clustered bootstrap (B = 2000), paired.
**Decision rule.** dR^2 95% CI > 0: "OFA adds explanatory value"; CI includes 0: "no difference detected" (never
"no effect"); CI < 0: "OFA worsens out-of-sample prediction".
**Precision (synthetic simulation, same design):** CI half-width ~0.10-0.13, so only dR^2 of roughly 0.10-0.15
or more is detectable. This is stated in the paper.
**Missing data.** Eyes with missing outcome are excluded from that outcome (n reported); missing features are
median-imputed in-fold.

**Secondary (pre-specified, reported with CIs, no significance claims):**
| ID | Contrast (A vs B) | Outcome |
|---|---|---|
| S1 | AS+MPOD+OFA vs AS+MPOD | VA |
| S2a/b | AS+OFA vs AS | VAlc / VA |
| S3a/b (context) | AS+MPOD vs AS | VAlc / VA |
| S4a/b | AS+grade+OFA vs AS+grade (does function add beyond the structural grade?) | VAlc / VA |
| S5a/b | AS+MPOD+Del, AS+MPOD+Amp vs AS+MPOD | VAlc |
| S6a/b | AS+MPOD+OFA+OFA44 vs AS+MPOD+OFA (wider field) | VAlc / VA |
| SENS1 | primary contrast with random forest | VAlc |
| SENS2 | primary contrast at full 20-region resolution | VAlc |
AS = age + sex. Code: `src/p2_function.py` (`CONTRASTS`).

## 4. P2 (primary, estimation): test-retest reliability of OFA vs MPOD
**Estimands** (session 1 vs 2, eye level, n = 58), per measure (9 feature groups x ring, plus 44-region summaries):
ICC(2,1) absolute agreement; within-eye SD (Sw); repeatability coefficient RC = 2.77 Sw; bias (session 2 - 1)
and 95% limits of agreement; RC / between-eye SD; per-zone ICC (median [IQR], % zones >= 0.75).
Strata: all eyes (primary), AREDS 1, AREDS 2-4. CIs: subject-clustered bootstrap (B = 2000).
**Paired contrasts:** dICC (Del or Amp minus MPOD mean, same ring; Del2/Amp2 vs Del/Amp whole field).
**Interpretation (fixed):** Koo & Li bands (< 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent) applied
to the **CI**, not the point estimate. Monitoring implication is stated through RC (minimal detectable change),
separately for ring means and single zones. Code: `src/p2_reliability.py`.

## 5. Previously pre-specified analyses carried over (reporting commitment of `ANALYSIS_PLAN_OFA.md`)
C (structure-function coupling) and D (functional signal in early AMD) were pre-specified for the short paper and
have been run (`share/run_20261011/ofa_value`). They are reported in one table, labelled as such, with the
original conclusions ("no difference detected" where the CI includes 0). The "faster delay in early AMD"
direction was observed in these data, so it is **not** tested here as a new hypothesis; it is stated as a
hypothesis for an independent cohort.

## 6. Not in this paper
Longitudinal progression (decided after S0 visit counts); AREDS-grade classification (Paper 1); any model search
beyond section 3.

## 7. Reporting commitments
- P1 primary result is reported in the abstract whatever its direction.
- Every analysis in sections 3-5 is reported (main text or supplement).
- Claims never exceed the CIs. Only one confirmatory test (P1), so no multiplicity correction; all other results
  are descriptive.
- Deviations from this plan are logged below with date and reason before results are seen where possible.
- Provenance: every run writes `run_info.json` (git commit, plan SHA-256, input-file SHA-256, versions, arguments).

## 8. Deviation log
| Date | Change | Reason | Before/after seeing results |
|---|---|---|---|
| | | | |
