# Pre-specified analysis plan, Paper 2 (short paper)

**Status: DRAFT, not signed off.** The P1/P2 code refuses to run on real data until this line reads
`Status: SIGNED OFF (date, names)`. Step S0 (descriptive) may run before sign-off.
**Version 2, 11 Oct 2026**, branch `AMD-P2`. **Sign-off:** ____________ (all co-authors, date)

Working title: *Does objective pupillographic perimetry (OFA) explain visual function in age-related macular
degeneration beyond age and macular pigment?* (with first-to-last-visit stability as a descriptive aim)
Target: IEEE EMBC 2027 (full paper; deadline reported as 24 Jan 2027, **verify on embc.embs.org**).
Paper 1 (submitted, *Machine Learning: Health*) is cited as "submitted"; none of its results are re-reported.

## 0. Transparency: what has already been seen
| Item | Seen before this plan? | Consequence |
|---|---|---|
| Visual acuity (VA, VAlc) vs any feature | **No** | P1 is genuinely confirmatory |
| Age and sex vs AREDS grade (S0, run 11 Oct 2026) | Yes | age is in every P1 baseline (fixed before S0 was run) |
| First-vs-last-visit ICC point estimates, 20-region ring means (`share/run_20261011/diagnostics`, labelled "test-retest" there) | Yes | P2 is descriptive; the paper states these were inspected |
| 44-region AUCs vs AREDS grade | Yes | not reused as a question |
| Coupling (C) and early-AMD OFA (D) vs AREDS grade | Yes, run 20261011 | reported as *previously pre-specified* (section 5), no new claims |
| Earlier DL results (F1 ~0.80) | Yes | **not used**: row-level splits put both rows of an eye in train and test (leakage) and used the scrambled CSV |

## 1. Data
- 29 participants, 58 eyes (AREDS 1/2/3/4 = 26/15/10/7 eyes), single site. Verified export (`src.mat_to_csv`,
  max difference 0 vs MATLAB). Never `reruns_SCRAMBLED_archive/`.
- **The two rows per eye are two study visits, not test-retest sessions:** Repeat 1 = first visit, Repeat 2 = last
  visit (from `d.filesRe`: V1+V4 for 20, V1+V3 for 2, V1+V2 for 7 included participants; VisitN 4/3/2 = 20/2/7;
  last-visit years 2018/2019/2020 = 2/19/8). The interval is **unknown** (exam dates unavailable; likely months to
  about 2 years). One participant code in the file names is not among the 29 and is excluded.
- **Acuity:** one VA and one low-contrast VA (VAlc) per eye in `d.T`; the visit is **not recorded** (most likely
  the last visit; unconfirmed). Units: **letter scores, higher = better** (S0: VA 83 [81-85], VAlc 73 [68-76]).
  VA is near ceiling (IQR 4 letters), which supports VAlc as the primary outcome.
- **AREDS grade:** one per eye (`d.sevs`); its visit is unknown, so grade stability cannot be assessed.
- **Rings:** 20-region field from `d.p.ring` (centre 1-4, middle 5-12, outer 13-20). 44-region field from
  `d.p(1).pattern(1)`, exported as `ofa44_layout.csv` (region, ring, sector, ecc_mid, angle_mid, x/y centre).
  Only the given `ring` numbers are used (ring means), which makes the analysis independent of left/right
  mirroring (`regionfliplr`/`regionflipud`). Sector, eccentricity units and orientation are not interpreted.

**Open items (do not block sign-off; reported as limitations if unresolved)**
- [ ] Confirm with Ted: acuity visit; 44-region eccentricity units and eye orientation.
- [ ] Clinical co-author confirms letter-score units.
- [ ] Confirm that Paper 1 (manuscript and supplement) reports no ICC and does not call the two visits
      "test-retest sessions" (results pack checked: no ICC).

## 2. Step S0: baseline and safety checks (descriptive; done)
- Table 1 by AREDS grade; age-only AUC; age vs OFA total deviation and MPOD in AREDS 1 eyes; follow-up counts.
- **Age result (S0, `share/paper2_run_20261011/p2_baseline`):** AUC of raw age as the score, no model and no CV,
  eye level, participant-clustered bootstrap: any 0.80 [0.65, 0.93], early 0.71 [0.51, 0.87], advanced 0.89
  [0.73, 0.99]. **FLAG** passed to Paper 1 (its age check runs on the Paper 1 branch, `scripts/12_age_check.sh`,
  participant-grouped primary; not duplicated here). For Paper 2: every P1 model contains age and sex.
- S0 contains no association between predictors and acuity.

## 3. P1 (sole primary aim, confirmatory): does OFA explain acuity beyond age, sex and MPOD?
**Hypothesis H1.** Adding OFA (Del, Amp; 3 ring means each) to age + sex + MPOD (7 statistics x 3 ring means)
increases out-of-fold R^2 for **low-contrast visual acuity (VAlc)**.

**Design.** Eye level; features = **mean of first and last visit** (primary). Repeated (R = 20) 5-fold CV,
**participant-grouped** (age and sex are participant-level; fellow eyes share acuity determinants), stratified on
worst-eye AREDS {1, 2, 3+}. Identical splits for all feature sets of an outcome and visit (paired). Ridge
regression (alpha tuned by inner 3-fold participant-grouped CV); imputation and scaling inside training folds.
Seed 2027. No deep learning (58 eyes).
**Metric.** Out-of-fold R^2 averaged over repeats (primary); Spearman rho and MAE (secondary).
**Inference.** Participant-clustered bootstrap (B = 2000), paired.
**Decision rule.** dR^2 95% CI > 0: "OFA adds explanatory value"; CI includes 0: "no difference detected" (never
"no effect"); CI < 0: "OFA worsens out-of-sample prediction".
**Visit rule (because the acuity visit is unknown).** The primary contrast is repeated with last-visit features
only (SENS1) and first-visit features only (SENS2). If the three conclusions agree, report "consistent". If they
differ, report the **last visit as primary-for-interpretation** and flag the uncertainty in the abstract.
**Precision (synthetic simulation, same design):** CI half-width ~0.10-0.13, so only dR^2 of roughly 0.10-0.15
or more is detectable. Stated in the paper.
**Missing data.** Eyes with missing outcome are excluded from that outcome (n reported; VA: 2 eyes); missing
features are median-imputed in-fold.

**Secondary (pre-specified; reported with CIs, no significance claims)**
| ID | Contrast (A vs B), visit mean unless stated | Outcome |
|---|---|---|
| S1 | AS+MPOD+OFA vs AS+MPOD | VA |
| S2a/b | AS+OFA vs AS | VAlc / VA |
| S3a/b (context) | AS+MPOD vs AS | VAlc / VA |
| S4a/b | AS+grade+OFA vs AS+grade (function beyond the structural grade) | VAlc / VA |
| S5a/b | AS+MPOD+Del, AS+MPOD+Amp vs AS+MPOD | VAlc |
| S6a/b | AS+MPOD+OFA+OFA44 vs AS+MPOD+OFA (wider field: 44-region ring means, whole-field mean and worst) | VAlc / VA |
| SENS1 / SENS2 | primary contrast, last visit only / first visit only (visit rule above) | VAlc |
| SENS3 | primary contrast with random forest | VAlc |
| SENS4 | primary contrast at full 20-region resolution | VAlc |
AS = age + sex. Code: `src/p2_function.py` (`RUNS`, `CONTRASTS`).

## 4. P2 (secondary aim, descriptive): visit-to-visit stability, first vs last visit
**Not test-retest reliability.** With an unknown interval of months to years, every agreement statistic mixes
measurement noise with true change. **No claim about device reliability is made.** Wording in the paper:
"visit-to-visit stability"; the unknown interval is a stated limitation.
**Estimands** (n = 58 eyes), per measure (9 feature groups x ring, plus 44-region ring means and summaries):
ICC(2,1) absolute agreement; within-eye SD across visits (Sw); stability coefficient SC95 = 2.77 Sw; mean
first-to-last change with 95% limits; SC95 / between-eye SD; per-zone ICC (median [IQR], % zones >= 0.75).
**Strata:** all eyes; AREDS 1; AREDS 2; AREDS 3-4. The mean change per stratum is the **descriptive
first-to-last change by AREDS grade**; with unknown interval and grade timing it is hypothesis-generating only.
CIs: participant-clustered bootstrap (B = 2000). Paired dICC (OFA vs MPOD mean, same ring) is descriptive.
**Interpretation (fixed):** bands (< 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent) describe
stability, applied to the CI. A high ICC means OFA ranks eyes consistently across visits; a low ICC cannot separate
noise from progression. Code: `src/p2_stability.py`.

## 5. Previously pre-specified analyses carried over (reporting commitment of `ANALYSIS_PLAN_OFA.md`)
C (structure-function coupling) and D (functional signal in early AMD) were pre-specified for the short paper and
have been run (`share/run_20261011/ofa_value`). They are reported in one table, labelled as such, with the
original conclusions. The "faster delay in early AMD" direction was observed in these data, so it is **not**
tested here as a new hypothesis; it is stated as a hypothesis for an independent cohort.
Note: the ICC from that plan was framed as test-retest; it is re-framed here as visit-to-visit stability (section 4).

## 6. Not in this paper
Longitudinal progression: participants had 2-4 visits, but the export holds only the first and last, without
dates. A progression paper needs the intermediate visits and exam dates. AREDS-grade classification (Paper 1).
Any model search beyond section 3.

## 7. Reporting commitments
- P1 primary result (with the visit rule outcome) is reported in the abstract whatever its direction.
- Every analysis in sections 3-5 is reported (main text or supplement).
- Claims never exceed the CIs. One confirmatory test (P1), so no multiplicity correction; all else is descriptive.
- Deviations are logged below with date and reason.
- Provenance: every run writes `run_info.json` (git commit, plan SHA-256, input-file SHA-256, versions, arguments).
- Shared outputs are aggregates only; no dates or identifiers (packager blocks pre-2025 calendar dates).

## 8. Deviation log
| Date | Change | Reason | Before/after seeing P1/P2 results |
|---|---|---|---|
| 11 Oct 2026 | v2: two visits are first/last study visits, not test-retest; P2 renamed visit-to-visit stability, descriptive, strata AREDS 1/2/3-4; P1 is the sole primary aim; visit sensitivity (SENS1/SENS2) and visit rule added; 44-region rings from `d.p(1).pattern(1)` | data structure clarified from `d.filesRe`; acuity visit not recorded | before (no P1/P2 run on real data) |
