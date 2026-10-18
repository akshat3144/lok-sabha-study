# RQ2 exploratory: is the decline in responsiveness real?

## What holds and what does not (all CIs 95%)

**Holds**
1. The decline is not an LLM drift artefact. Human gold (design weighted, n=400 questions, 1,636 parts) falls in step with M2: late minus early gold -0.048 [-0.121, 0.025] vs M2 on the same questions -0.044 [-0.109, 0.025]; LS18 minus LS13 gold -0.115 [-0.212, -0.007] vs M2 -0.103 [-0.204, 0.006]. Gold alone is underpowered (LS level CIs about +-0.05 to +-0.09), so the evidence is "consistent and not contradicted", not gold-only proof.
2. Gemini's bias is stable over time. M2 minus gold is +0.024 [0.009, 0.041] overall (M2 lenient), homogeneity across LS13-18 p=0.96 (question level) and p=0.91 (part level), bias slope 0.000 [-0.014, 0.015] per decade. Part level accuracy 0.89 (early) / 0.86 (middle) / 0.87 (late), kappa 0.71 / 0.69 / 0.73; sensitivity for non-FULL about 0.80 in every period. M2 under-detects non-FULL parts if anything (gold non-FULL 22.5% -> 33.3%, M2 on same parts 20.7% -> 29.9%).
3. Bias-corrected (PPI++) means: LS13 0.832 [0.803, 0.860], LS17 0.797 [0.766, 0.829], LS18 0.738 [0.697, 0.780]; LS18 minus LS13 -0.093 [-0.135, -0.051] (naive -0.080 [-0.099, -0.062]); late minus early -0.060 [-0.096, -0.023]; trend -0.027 [-0.044, -0.009] per decade (naive -0.027 [-0.035, -0.019]). Correction leaves the decline intact (slightly larger).
4. Complexity does not explain all of it. LS18 minus LS13 is -0.080 [-0.097, -0.063] raw and -0.051 [-0.063, -0.038] with full controls (63% survives); trend -0.026 [-0.034, -0.018] per decade raw, -0.013 [-0.020, -0.006] controlled (51% survives; 47 to 57% across specifications; same for the share of FULL parts, 64%). Robust to flexible length (decile FE, parts FE), words per part, and asker bloc.
5. Deferral's decline is real. The formula marker (information being collected / will be laid on the Table) is in 3.9% of LS13 answers and 0.02% in LS18, M2 labels DEFERRAL for 99.8 to 100% of formula questions in every LS, and M2 DEFERRAL parts fall from 4.7 to 0.03 per 100 parts. Gold agrees (4.6 -> 0.7 per 100 parts, same in M2 on those parts; only 25 gold deferral parts, CI wide).
6. Excluding formula questions (which score about 0.11) the decline is larger: LS18 minus LS13 -0.112 [-0.129, -0.095] raw, -0.087 [-0.099, -0.075] controlled. The vanishing of deferral partly masks the rise of other shortfalls.
7. Rise of GENERIC and INCOMPLETE (M2): per 100 parts GENERIC 6.6 -> 15.2 (LS13 to LS18, +8.6 [7.8, 9.4]), INCOMPLETE 4.8 -> 13.3 (+8.5 [7.2, 9.9]); together about 111% of the +15.3 [13.1, 17.7] pp rise in non-FULL parts, offset by DEFERRAL -4.7 [-5.2, -4.3]. Within request-type change is +11.1 pp of +10.7 (composition -0.4 pp). INCOMPLETE rise is confirmed in gold (+6.6 [3.6, 9.8] early to late).
8. No break at changes of government. Jumps at first sitting of the new LS (+-5y windows, HC3): 2004 +0.021 [-0.007, 0.048], 2014 +0.005 [-0.010, 0.019], 2024 +0.007 [-0.032, 0.045]; placebo re-elections 2009 -0.015 [-0.036, 0.005], 2019 +0.006 [-0.013, 0.024]. Break models at placebo dates fit as well as at government changes (LOO RMSE 0.0134 vs 0.0137; cubic 0.0138). No level jump at the LS17/LS18 boundary, the extraction source change (API to docx) therefore shows no discontinuity.
9. The trend is flat then accelerating, not government specific. Within LS13-16 slopes are -0.004 to +0.001 per year (CIs include 0 except LS16, -0.003 [-0.006, -0.0001]); the decline is concentrated after about 2018: estimated kink 2018.3 [2016.9, 2019.7], slope -0.011 per year afterwards; within LS17 -0.009 [-0.019, 0.001]; within LS18 -0.026 [-0.041, -0.012] per year (7 sessions).

**Does not hold / caveats**
- The raw tail (LS18 0.771) is not all "worse behaviour": about 37% of the LS18 gap and half the trend is associated with longer questions (median 356 -> 644 characters, 59 -> 101 words, same pattern in every extraction source) and a shift to ACTION requests; the length gradient is also present in human gold (-0.160 [-0.268, -0.058] per log unit) and M2 does not over penalise long questions (M2 minus gold -0.007 [-0.048, 0.037]). Whether longer questions are a cause or just a marker (e.g. bundled asks) is not identified. Request type shares are M2 labels (S2 onward), so treat those controls with care.
- Controlling for answer side variables (length, annexure; mediators, S7) makes the decline larger (120%), so answers are longer and more complete in form but not in content: a sensitivity only.
- GENERIC's rise is not confirmed by gold (+0.3 [-4.9, 5.4] early to late vs M2 +5.4 [0.5, 10.2] on the same parts): humans put more late shortfalls into DIFFERENT_QUESTION, JURISDICTION and OTHER. Type agreement among non-FULL in both is 82% (92% early, 79 to 81% later), so type shares from M2 are less reliable than the non-FULL rate itself.
- The sup-F scan prefers a break near 2020.9 (inside LS17), the hinge near 2018.3: the exact timing is imprecise and neither is a government change. Joint HC3 Wald tests add little over a cubic trend (3 government dates p=0.02; placebo dates p=0.003 to 0.013), session means are autocorrelated and n=78, so treat as descriptive. Power at 2024 is low (7 post sessions, jump CI +-0.04); 2024 pre-trend extrapolation gap -0.010 [-0.040, 0.021].
- Design caveats: LS13-16 are a 1/6 sample (design weighted), LS17-18 a census, LS18 comes from different extraction (docx / pdf vs API) and is still incomplete (to Aug 2026); gold CIs use a Rao-Wu bootstrap on 400 questions with collapsed singleton strata; PPI++ CIs use two-way clusters with few sessions per LS (t with sessions-1 df).

Exploratory; everything run is reported. Outcome y_m2 = mean over parts of FULL=1 / PARTIAL=.5 / NONE=0 (Gemini 3.8 flash). Design weights m2_weight (LS13-16 1/6 sample, LS17-18 census). Statistics only; no answer text is quoted. Script: `analysis/explore_rq2.py`.

## 1. Real or LLM drift?

Gold sample: 400 questions (1,636 parts) from the 400-question `in_estimate` set; 0 gold parts lack an M2 label (dropped). Weights 1/gold_pi (Hajek means). CIs: Rao-Wu rescaled bootstrap, B=2000, strata = main_sample strata (singletons collapsed within LS), 95% percentile. Estimates refer to the M3 universe (answered, 1-6 parts).

### 1a. Design-weighted human gold score vs M2 on the same questions (question score, mean over parts)

| group | n_q | n_parts | gold_mean | m2_mean_on_gold_q | bias_q (M2-gold) |
|---|---|---|---|---|---|
| LS13 | 54 | 215 | 0.845 [0.750, 0.923] | 0.865 [0.775, 0.942] | 0.020 [-0.004, 0.042] |
| LS14 | 51 | 198 | 0.811 [0.713, 0.890] | 0.835 [0.744, 0.907] | 0.024 [-0.015, 0.068] |
| LS15 | 57 | 235 | 0.778 [0.682, 0.868] | 0.816 [0.722, 0.902] | 0.038 [-0.002, 0.086] |
| LS16 | 58 | 242 | 0.819 [0.751, 0.885] | 0.832 [0.765, 0.895] | 0.013 [-0.026, 0.051] |
| LS17 | 90 | 377 | 0.809 [0.756, 0.858] | 0.832 [0.784, 0.879] | 0.023 [-0.003, 0.049] |
| LS18 | 90 | 369 | 0.730 [0.674, 0.784] | 0.761 [0.702, 0.816] | 0.032 [0.001, 0.064] |
| early (LS13-14, 1999-2009) | 105 | 413 | 0.829 [0.762, 0.887] | 0.851 [0.789, 0.907] | 0.022 [-0.000, 0.045] |
| middle (LS15-16, 2009-2019) | 115 | 477 | 0.799 [0.740, 0.853] | 0.824 [0.766, 0.877] | 0.026 [-0.003, 0.059] |
| late (LS17-18, 2019-2026) | 180 | 746 | 0.780 [0.744, 0.819] | 0.806 [0.770, 0.843] | 0.026 [0.006, 0.047] |
| All LS (pop.-weighted) | 400 | 1,636 | 0.805 [0.770, 0.836] | 0.829 [0.797, 0.859] | 0.024 [0.009, 0.041] |


### 1b. M2 (gemini-3.8-flash) vs human gold at the part level (design-weighted; unweighted point estimates in last two columns)

| group | n_parts | accuracy | kappa | kappa_linear | bias_part (M2-gold) | gold_nonFULL_rate | m2_nonFULL_rate | sens_nonFULL | spec_FULL | accuracy_unweighted | kappa_unweighted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LS13 | 215 | 0.924 [0.881, 0.966] | 0.774 [0.593, 0.894] | 0.827 [0.620, 0.932] | 0.023 [0.003, 0.045] | 0.207 [0.116, 0.321] | 0.185 [0.090, 0.297] | 0.823 [0.609, 0.963] | 0.982 [0.959, 1.000] | 0.926 | 0.787 |
| LS14 | 198 | 0.859 [0.796, 0.923] | 0.643 [0.455, 0.799] | 0.682 [0.476, 0.847] | 0.039 [-0.007, 0.092] | 0.246 [0.163, 0.340] | 0.233 [0.157, 0.320] | 0.767 [0.571, 0.944] | 0.941 [0.887, 0.986] | 0.884 | 0.684 |
| LS15 | 235 | 0.870 [0.789, 0.944] | 0.703 [0.546, 0.855] | 0.751 [0.579, 0.892] | 0.038 [-0.001, 0.081] | 0.300 [0.181, 0.419] | 0.253 [0.142, 0.371] | 0.761 [0.589, 0.908] | 0.965 [0.925, 0.995] | 0.864 | 0.662 |
| LS16 | 242 | 0.859 [0.806, 0.910] | 0.681 [0.548, 0.782] | 0.729 [0.594, 0.828] | 0.012 [-0.026, 0.051] | 0.283 [0.192, 0.372] | 0.285 [0.179, 0.393] | 0.825 [0.692, 0.919] | 0.928 [0.856, 0.981] | 0.864 | 0.698 |
| LS17 | 377 | 0.887 [0.853, 0.922] | 0.737 [0.641, 0.821] | 0.769 [0.671, 0.845] | 0.023 [-0.005, 0.052] | 0.279 [0.212, 0.348] | 0.261 [0.195, 0.326] | 0.813 [0.711, 0.901] | 0.953 [0.916, 0.985] | 0.875 | 0.729 |
| LS18 | 369 | 0.838 [0.789, 0.887] | 0.702 [0.622, 0.785] | 0.748 [0.672, 0.821] | 0.035 [0.005, 0.069] | 0.429 [0.358, 0.501] | 0.367 [0.295, 0.439] | 0.803 [0.718, 0.884] | 0.961 [0.925, 0.991] | 0.824 | 0.684 |
| early (LS13-14, 1999-2009) | 413 | 0.894 [0.856, 0.930] | 0.709 [0.587, 0.809] | 0.756 [0.623, 0.856] | 0.030 [0.006, 0.058] | 0.225 [0.158, 0.298] | 0.207 [0.144, 0.278] | 0.795 [0.663, 0.909] | 0.964 [0.936, 0.987] | 0.906 | 0.736 |
| middle (LS15-16, 2009-2019) | 477 | 0.864 [0.816, 0.910] | 0.692 [0.595, 0.782] | 0.740 [0.635, 0.829] | 0.025 [-0.004, 0.055] | 0.291 [0.216, 0.367] | 0.270 [0.193, 0.350] | 0.793 [0.694, 0.883] | 0.946 [0.906, 0.977] | 0.864 | 0.683 |
| late (LS17-18, 2019-2026) | 746 | 0.869 [0.841, 0.897] | 0.727 [0.665, 0.786] | 0.763 [0.701, 0.819] | 0.028 [0.006, 0.050] | 0.333 [0.283, 0.384] | 0.299 [0.249, 0.349] | 0.809 [0.740, 0.871] | 0.956 [0.927, 0.981] | 0.850 | 0.709 |
| All LS (pop.-weighted) | 1,636 | 0.876 [0.852, 0.898] | 0.709 [0.655, 0.760] | 0.753 [0.693, 0.804] | 0.027 [0.012, 0.044] | 0.280 [0.240, 0.320] | 0.256 [0.216, 0.299] | 0.798 [0.741, 0.856] | 0.955 [0.935, 0.971] | 0.868 | 0.711 |


bias_part = mean M2 part score minus mean gold part score (positive = M2 more lenient than humans). sens_nonFULL = P(M2 says not FULL | gold not FULL); spec_FULL = P(M2 says FULL | gold FULL).

### 1c. Is Gemini's bias time-varying?

- Homogeneity of question-level bias across LS13-18 (bootstrap Wald, 5 df): chi2=1.06, p=0.957; part-level bias: chi2=1.56, p=0.906.
- Slopes per decade on the gold questions (weighted OLS on year):

| stat | 95% CI |
|---|---|
| gold slope /decade | -0.024 [-0.061, 0.017] |
| M2 slope /decade (gold q) | -0.024 [-0.060, 0.016] |
| bias slope /decade | 0.000 [-0.014, 0.015] |


Contrasts (95% bootstrap CI):

| contrast | stat | diff |
|---|---|---|
| LS18 - LS13 | gold_mean | -0.115 [-0.212, -0.007] |
| LS18 - LS13 | m2_mean_on_gold_q | -0.103 [-0.204, 0.006] |
| LS18 - LS13 | bias_q (M2-gold) | 0.012 [-0.029, 0.053] |
| LS18 - LS13 | bias_part (M2-gold) | 0.012 [-0.027, 0.052] |
| LS18 - LS13 | accuracy | -0.086 [-0.151, -0.020] |
| LS18 - LS13 | kappa | -0.071 [-0.216, 0.128] |
| LS17 - LS13 | gold_mean | -0.036 [-0.129, 0.069] |
| LS17 - LS13 | m2_mean_on_gold_q | -0.033 [-0.126, 0.065] |
| LS17 - LS13 | bias_q (M2-gold) | 0.003 [-0.031, 0.039] |
| LS17 - LS13 | bias_part (M2-gold) | 0.000 [-0.037, 0.036] |
| LS17 - LS13 | accuracy | -0.037 [-0.093, 0.021] |
| LS17 - LS13 | kappa | -0.037 [-0.191, 0.165] |
| late - early | gold_mean | -0.048 [-0.117, 0.028] |
| late - early | m2_mean_on_gold_q | -0.044 [-0.109, 0.025] |
| late - early | bias_q (M2-gold) | 0.004 [-0.027, 0.034] |
| late - early | bias_part (M2-gold) | -0.003 [-0.037, 0.031] |
| late - early | accuracy | -0.025 [-0.070, 0.025] |
| late - early | kappa | 0.017 [-0.096, 0.150] |


### 1d. PPI / PPI++ corrected means (M3 universe)

`bias_correct.ppi_fit`, intercept-only per group, design weights m2_weight for the prediction term, 1/gold_pi for the rectifier; two-way (MP x session) cluster SEs, t(sessions-1) CIs. n rows used=139,654, gold=400; sampled-but-unlabelled=0, gold-without-M2=0. 'simple PPI' = design-weighted M2 mean over the universe + Hajek mean of (gold - M2) on gold (lambda=1), SE = sqrt(session-cluster SE^2 + bootstrap SE^2).

gold only (HT, lambda=0) divides the HT total by the M2-sample estimate of N and is very noisy; the Hajek gold mean is the usable gold-only benchmark.

| group | naive (on predictions) | gold only (Hajek, bootstrap) | gold only (HT, lambda=0) | PPI (lambda=1.0) | PPI++ (lambda tuned) | simple PPI (lambda=1, bootstrap rectifier) |
|---|---|---|---|---|---|---|
| LS13 | 0.851 [0.845, 0.858] | 0.845 [0.750, 0.923] | 0.847 [0.522, 1.173] | 0.831 [0.803, 0.860] | 0.832 [0.803, 0.860] | 0.831 [0.805, 0.858] |
| LS14 | 0.866 [0.858, 0.874] | 0.811 [0.713, 0.890] | 0.810 [0.551, 1.069] | 0.842 [0.786, 0.897] | 0.839 [0.789, 0.889] | 0.842 [0.795, 0.888] |
| LS15 | 0.850 [0.846, 0.855] | 0.778 [0.682, 0.868] | 0.783 [0.520, 1.047] | 0.812 [0.768, 0.856] | 0.810 [0.769, 0.851] | 0.812 [0.762, 0.862] |
| LS16 | 0.843 [0.838, 0.848] | 0.819 [0.751, 0.885] | 0.822 [0.532, 1.112] | 0.830 [0.785, 0.874] | 0.829 [0.786, 0.873] | 0.830 [0.788, 0.872] |
| LS17 | 0.820 [0.811, 0.828] | 0.809 [0.756, 0.858] | 0.811 [0.525, 1.096] | 0.797 [0.764, 0.830] | 0.797 [0.766, 0.829] | 0.797 [0.767, 0.827] |
| LS18 | 0.771 [0.750, 0.793] | 0.730 [0.674, 0.784] | 0.729 [0.507, 0.951] | 0.740 [0.687, 0.792] | 0.738 [0.697, 0.780] | 0.740 [0.694, 0.786] |
| early (LS13-14, 1999-2009) | 0.858 [0.853, 0.864] | 0.829 [0.762, 0.887] | 0.830 [0.637, 1.023] | 0.836 [0.808, 0.865] | 0.836 [0.809, 0.863] | 0.836 [0.811, 0.861] |
| middle (LS15-16, 2009-2019) | 0.847 [0.843, 0.850] | 0.799 [0.740, 0.853] | 0.803 [0.621, 0.985] | 0.821 [0.792, 0.850] | 0.820 [0.792, 0.848] | 0.821 [0.789, 0.852] |
| late (LS17-18, 2019-2026) | 0.802 [0.788, 0.817] | 0.780 [0.744, 0.819] | 0.781 [0.591, 0.971] | 0.776 [0.748, 0.804] | 0.776 [0.749, 0.804] | 0.776 [0.750, 0.802] |
| All LS (pop.-weighted) | 0.840 [0.833, 0.847] | 0.805 [0.770, 0.836] | 0.807 [0.703, 0.911] | 0.815 [0.798, 0.833] | 0.815 [0.798, 0.832] | 0.815 [0.798, 0.833] |


Differences between groups (independent groups; SE = sqrt(se_a^2 + se_b^2); normal 95% CI):

| method | contrast | diff | z |
|---|---|---|---|
| naive (on predictions) | LS18 - LS13 | -0.080 [-0.099, -0.062] | -8.532 |
| naive (on predictions) | LS17 - LS13 | -0.032 [-0.042, -0.022] | -6.298 |
| naive (on predictions) | late - early | -0.056 [-0.071, -0.041] | -7.385 |
| gold only (Hajek, bootstrap) | LS18 - LS13 | -0.115 [-0.220, -0.010] | -2.153 |
| gold only (Hajek, bootstrap) | LS17 - LS13 | -0.036 [-0.138, 0.067] | -0.679 |
| gold only (Hajek, bootstrap) | late - early | -0.048 [-0.121, 0.025] | -1.291 |
| PPI++ (lambda tuned) | LS18 - LS13 | -0.093 [-0.135, -0.051] | -4.334 |
| PPI++ (lambda tuned) | LS17 - LS13 | -0.034 [-0.073, 0.004] | -1.746 |
| PPI++ (lambda tuned) | late - early | -0.060 [-0.096, -0.023] | -3.219 |
| simple PPI (lambda=1, bootstrap rectifier) | LS18 - LS13 | -0.092 [-0.136, -0.048] | -4.088 |
| simple PPI (lambda=1, bootstrap rectifier) | LS17 - LS13 | -0.034 [-0.071, 0.002] | -1.859 |
| simple PPI (lambda=1, bootstrap rectifier) | late - early | -0.060 [-0.094, -0.026] | -3.439 |


Linear time trend in the mean (per decade), design-based:

| method | slope per decade [95% CI] |
|---|---|
| naive (on predictions) | -0.0266 [-0.0347, -0.0186] |
| gold only (HT, lambda=0) | -0.0108 [-0.1416, 0.1200] |
| PPI++ (lambda tuned) | -0.0268 [-0.0442, -0.0094] |
| PPI (lambda=1.0) | -0.0273 [-0.0451, -0.0094] |


Design-weighted M2 means by LS (no correction), two-way clustered CIs:

| ls | population | n | mean_y_m2 [95% CI] |
|---|---|---|---|
| 13 | all analysed questions | 12,170 | 0.851 [0.844, 0.858] |
| 13 | M3 universe (1-6 parts) | 11,566 | 0.851 [0.845, 0.858] |
| 14 | all analysed questions | 10,996 | 0.864 [0.856, 0.873] |
| 14 | M3 universe (1-6 parts) | 10,446 | 0.866 [0.858, 0.874] |
| 15 | all analysed questions | 13,136 | 0.850 [0.845, 0.854] |
| 15 | M3 universe (1-6 parts) | 12,683 | 0.850 [0.846, 0.855] |
| 16 | all analysed questions | 13,066 | 0.843 [0.837, 0.848] |
| 16 | M3 universe (1-6 parts) | 12,951 | 0.843 [0.838, 0.848] |
| 17 | all analysed questions | 60,307 | 0.820 [0.811, 0.828] |
| 17 | M3 universe (1-6 parts) | 58,508 | 0.820 [0.811, 0.828] |
| 18 | all analysed questions | 34,674 | 0.771 [0.749, 0.792] |
| 18 | M3 universe (1-6 parts) | 33,306 | 0.771 [0.750, 0.793] |


Figure: `rq2_fig1_gold_vs_m2.png`

## 2. Is it question complexity?

### 2a. Did questions get longer / more multi-part? (all analysed questions; request-type shares are design-weighted M2 labels)

| LS | n questions (all) | mean n_parts | % >=5 parts | median question chars | mean log_qlen | % starred | % security | % >0 opposition askers | % rt_information | % rt_reason | % rt_action | % rt_opinion | median question words | median words per part | % questions flagged Hindi (text stripped) | mean answer chars | median answer words | % has annexure | % OCR answers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 72,987 | 3.95 | 29.78 | 356.00 | 5.87 | 8.62 | 9.26 | 43.30 | 52.34 | 7.38 | 36.88 | 3.40 | 59.00 | 15.67 | 0.00 | 902.49 | 105.00 | 12.33 | 0.00 |
| 14 | 65,969 | 4.04 | 32.56 | 358.00 | 5.87 | 9.17 | 9.07 | 38.14 | 49.64 | 6.35 | 40.79 | 3.21 | 60.00 | 15.25 | 0.00 | 953.10 | 115.00 | 13.78 | 0.00 |
| 15 | 78,749 | 4.15 | 39.50 | 406.00 | 5.99 | 8.11 | 9.90 | 41.83 | 47.30 | 6.95 | 42.11 | 3.63 | 67.00 | 16.50 | 0.00 | 1,139.72 | 143.00 | 19.43 | 0.00 |
| 16 | 78,454 | 3.96 | 32.64 | 454.00 | 6.11 | 8.10 | 9.30 | 33.73 | 42.83 | 6.77 | 47.06 | 3.34 | 75.00 | 19.33 | 0.15 | 1,412.41 | 172.00 | 25.36 | 0.07 |
| 17 | 60,307 | 4.09 | 35.75 | 477.00 | 6.16 | 7.71 | 5.68 | 43.13 | 45.62 | 7.02 | 44.53 | 2.84 | 78.00 | 19.75 | 0.34 | 1,824.65 | 218.00 | 30.79 | 0.07 |
| 18 | 34,674 | 4.24 | 40.35 | 644.00 | 6.42 | 7.96 | 3.52 | 52.61 | 48.80 | 5.30 | 43.68 | 2.22 | 101.00 | 23.80 | 17.63 | 2,653.19 | 306.00 | 33.49 | 0.10 |


Linear trend per decade (two-way clustered CI):

| variable | slope per decade [95% CI] | mean |
|---|---|---|
| n_parts | 0.0707 [0.0304, 0.1110] | 4.054 |
| log_qlen | 0.2028 [0.1703, 0.2354] | 6.034 |
| starred | -0.0047 [-0.0075, -0.0019] | 0.083 |
| rt_action | 0.0343 [0.0252, 0.0433] | 0.434 |
| rt_information | -0.0274 [-0.0379, -0.0168] | 0.472 |
| rt_reason | -0.0036 [-0.0064, -0.0007] | 0.066 |
| rt_opinion | -0.0033 [-0.0049, -0.0017] | 0.029 |


### 2b. Regressions of the question score (y_m2: FULL=1, PARTIAL=.5, NONE=0; y_full: share of parts FULL) on LS dummies (ref LS13)

WLS with design weights m2_weight; two-way clustered SEs (first-asker MP x session); t(min G - 1) CIs. Coefficients are in score units (1.0 = full scale). S2 onward uses request-type shares that are themselves M2 labels (so they may absorb part of the measured decline if M2 labels differ over time); S7 adds answer-side variables that are possible mediators and is shown only as a sensitivity.

| y | spec | n | LS14 vs LS13 | LS15 vs LS13 | LS16 vs LS13 | LS17 vs LS13 | LS18 vs LS13 | % of LS18 gap surviving | % of LS17 gap surviving |
|---|---|---|---|---|---|---|---|---|---|
| y_m2 | S0 none | 144,349 | 0.014 [0.004, 0.023] | -0.001 [-0.009, 0.006] | -0.008 [-0.016, -0.000] | -0.031 [-0.041, -0.021] | -0.080 [-0.097, -0.063] | 100.0 | 100.0 |
| y_m2 | S1a n_parts | 144,349 | 0.014 [0.005, 0.023] | 0.000 [-0.007, 0.008] | -0.008 [-0.016, 0.000] | -0.030 [-0.040, -0.020] | -0.078 [-0.095, -0.061] | 97.2 | 96.6 |
| y_m2 | S1b log_qlen | 144,349 | 0.013 [0.005, 0.022] | 0.006 [-0.002, 0.013] | 0.006 [-0.002, 0.014] | -0.015 [-0.024, -0.005] | -0.049 [-0.062, -0.035] | 60.6 | 46.9 |
| y_m2 | S1c starred | 144,349 | 0.013 [0.004, 0.023] | -0.001 [-0.008, 0.007] | -0.008 [-0.016, 0.000] | -0.031 [-0.041, -0.021] | -0.080 [-0.097, -0.063] | 99.7 | 98.9 |
| y_m2 | S1 n_parts+log_qlen+starred | 144,349 | 0.012 [0.004, 0.021] | 0.007 [-0.000, 0.015] | 0.013 [0.005, 0.021] | -0.007 [-0.016, 0.002] | -0.035 [-0.048, -0.023] | 44.0 | 23.2 |
| y_m2 | S2 S1 + request-type shares | 144,349 | 0.010 [0.002, 0.018] | 0.004 [-0.003, 0.012] | 0.008 [0.000, 0.015] | -0.011 [-0.020, -0.002] | -0.040 [-0.052, -0.028] | 50.0 | 36.3 |
| y_m2 | S3 S1 + ministry_group FE | 144,349 | 0.012 [0.004, 0.021] | 0.008 [0.000, 0.015] | 0.014 [0.006, 0.021] | -0.007 [-0.016, 0.002] | -0.036 [-0.049, -0.024] | 45.4 | 22.9 |
| y_m2 | S3b S1 + ministry_canonical FE (77) | 144,349 | 0.011 [0.003, 0.020] | 0.009 [0.001, 0.016] | 0.012 [0.004, 0.020] | -0.012 [-0.021, -0.002] | -0.043 [-0.056, -0.031] | 54.2 | 37.2 |
| y_m2 | S4 S1 + topic FE (K=50) | 144,349 | 0.012 [0.003, 0.020] | 0.008 [0.001, 0.015] | 0.013 [0.006, 0.020] | -0.009 [-0.018, -0.000] | -0.041 [-0.053, -0.028] | 50.9 | 29.5 |
| y_m2 | S5 S2 + ministry_canonical + topic FE (full) | 144,349 | 0.009 [0.000, 0.017] | 0.006 [-0.002, 0.014] | 0.007 [-0.000, 0.015] | -0.016 [-0.026, -0.007] | -0.051 [-0.063, -0.038] | 63.4 | 52.5 |
| y_m2 | S6 S5 + coasker + asker bloc | 144,349 | 0.008 [-0.001, 0.016] | 0.006 [-0.002, 0.014] | 0.008 [0.000, 0.016] | -0.015 [-0.025, -0.006] | -0.049 [-0.062, -0.037] | 61.6 | 49.9 |
| y_m2 | S7 S6 + answer-side (log answer len, annexure; mediators) | 144,349 | 0.004 [-0.005, 0.012] | -0.007 [-0.015, -0.000] | -0.016 [-0.023, -0.008] | -0.052 [-0.063, -0.041] | -0.096 [-0.109, -0.083] | 119.8 | 168.1 |
| y_m2 | S8 S5 with flexible length: question-length decile FE + n_parts FE (1..8+) | 144,349 | 0.008 [-0.000, 0.017] | 0.006 [-0.002, 0.014] | 0.007 [-0.001, 0.014] | -0.017 [-0.026, -0.007] | -0.053 [-0.065, -0.040] | 65.8 | 53.4 |
| y_m2 | S9 S5 with log words per part instead of log_qlen | 144,349 | 0.008 [-0.000, 0.017] | 0.004 [-0.003, 0.012] | 0.005 [-0.003, 0.013] | -0.018 [-0.028, -0.008] | -0.056 [-0.069, -0.043] | 69.9 | 58.4 |
| y_full | S0 none | 144,349 | 0.012 [0.001, 0.023] | -0.015 [-0.024, -0.007] | -0.036 [-0.045, -0.027] | -0.072 [-0.084, -0.059] | -0.151 [-0.177, -0.125] | 100.0 | 100.0 |
| y_full | S1 n_parts+log_qlen+starred | 144,349 | 0.010 [0.000, 0.019] | -0.003 [-0.012, 0.006] | -0.002 [-0.011, 0.007] | -0.034 [-0.045, -0.024] | -0.081 [-0.098, -0.064] | 53.6 | 47.9 |
| y_full | S5 S2 + ministry_canonical + topic FE (full) | 144,349 | 0.007 [-0.003, 0.017] | -0.003 [-0.013, 0.006] | -0.007 [-0.016, 0.003] | -0.043 [-0.055, -0.031] | -0.097 [-0.114, -0.080] | 64.0 | 59.7 |
| y_full | S8 S5 with flexible length: question-length decile FE + n_parts FE (1..8+) | 144,349 | 0.006 [-0.004, 0.016] | -0.005 [-0.014, 0.005] | -0.009 [-0.019, 0.000] | -0.044 [-0.056, -0.032] | -0.100 [-0.117, -0.082] | 65.9 | 61.2 |


### 2c. Same, linear trend per decade (decade = (year - 2000) / 10)

| y | spec | n | slope per decade | g_mp | g_session | % of slope surviving |
|---|---|---|---|---|---|---|
| y_m2 | S0 none | 144,349 | -0.0263 [-0.0343, -0.0183] | 1,918 | 78 | 100.0 |
| y_m2 | S1a n_parts | 144,349 | -0.0257 [-0.0336, -0.0178] | 1,918 | 78 | 97.9 |
| y_m2 | S1b log_qlen | 144,349 | -0.0141 [-0.0206, -0.0076] | 1,918 | 78 | 53.6 |
| y_m2 | S1c starred | 144,349 | -0.0261 [-0.0341, -0.0181] | 1,918 | 78 | 99.3 |
| y_m2 | S1 n_parts+log_qlen+starred | 144,349 | -0.0085 [-0.0146, -0.0023] | 1,918 | 78 | 32.2 |
| y_m2 | S2 S1 + request-type shares | 144,349 | -0.0106 [-0.0165, -0.0047] | 1,918 | 78 | 40.2 |
| y_m2 | S3 S1 + ministry_group FE | 144,349 | -0.0085 [-0.0147, -0.0024] | 1,918 | 78 | 32.4 |
| y_m2 | S3b S1 + ministry_canonical FE (77) | 144,349 | -0.0108 [-0.0179, -0.0038] | 1,918 | 78 | 41.2 |
| y_m2 | S4 S1 + topic FE (K=50) | 144,349 | -0.0096 [-0.0158, -0.0034] | 1,918 | 78 | 36.5 |
| y_m2 | S5 S2 + ministry_canonical + topic FE (full) | 144,349 | -0.0133 [-0.0204, -0.0062] | 1,918 | 78 | 50.6 |
| y_m2 | S6 S5 + coasker + asker bloc | 144,349 | -0.0125 [-0.0195, -0.0054] | 1,918 | 78 | 47.5 |
| y_m2 | S7 S6 + answer-side (log answer len, annexure; mediators) | 144,349 | -0.0315 [-0.0386, -0.0244] | 1,918 | 78 | 119.8 |
| y_m2 | S8 S5 with flexible length: question-length decile FE + n_parts FE (1..8+) | 144,349 | -0.0137 [-0.0205, -0.0069] | 1,918 | 78 | 52.1 |
| y_m2 | S9 S5 with log words per part instead of log_qlen | 144,349 | -0.0151 [-0.0223, -0.0078] | 1,918 | 78 | 57.3 |
| y_full | S0 none | 144,349 | -0.0522 [-0.0638, -0.0406] | 1,918 | 78 | 100.0 |
| y_full | S1 n_parts+log_qlen+starred | 144,349 | -0.0243 [-0.0320, -0.0166] | 1,918 | 78 | 46.5 |
| y_full | S5 S2 + ministry_canonical + topic FE (full) | 144,349 | -0.0291 [-0.0380, -0.0201] | 1,918 | 78 | 55.7 |
| y_full | S8 S5 with flexible length: question-length decile FE + n_parts FE (1..8+) | 144,349 | -0.0299 [-0.0385, -0.0214] | 1,918 | 78 | 57.3 |


Coefficients on controls in S5 (y_m2, LS dummies; per unit of each control):

| term | 95% CI | p |
|---|---|---|
| n_parts | 0.010 [0.008, 0.012] | 0.0000 |
| log_qlen | -0.079 [-0.086, -0.072] | 0.0000 |
| starred | 0.049 [0.043, 0.056] | 0.0000 |
| rt_reason | -0.034 [-0.053, -0.016] | 0.0004 |
| rt_action | 0.046 [0.039, 0.053] | 0.0000 |
| rt_opinion | 0.004 [-0.015, 0.023] | 0.6509 |
| ls[14] | 0.009 [0.000, 0.017] | 0.0417 |
| ls[15] | 0.006 [-0.002, 0.014] | 0.1232 |
| ls[16] | 0.007 [-0.000, 0.015] | 0.0633 |
| ls[17] | -0.016 [-0.026, -0.007] | 0.0012 |
| ls[18] | -0.051 [-0.063, -0.038] | 0.0000 |


### 2d. Data-source diagnostics (extraction source differs by LS; is the LS18 length increase an extraction artefact?)

LS13-17 questions come from the API ('api', a few fallback docx), LS18 from docx / pdf / ocr documents. Word counts (whitespace-split) rise in every LS18 source, so the length increase is not a whitespace / encoding artefact of one extractor. Devanagari text is already stripped from question_text in the data (the Hindi flag only records its presence).

| LS | source_q | n (M2 sample) | weighted share of LS % | mean y_m2 | mean log_qlen | median question words | % question has Hindi |
|---|---|---|---|---|---|---|---|
| 13 | api | 12,170 | 100.000 | 0.851 | 5.872 | 60.000 | 0.000 |
| 14 | api | 10,996 | 100.000 | 0.864 | 5.868 | 60.000 | 0.000 |
| 15 | api | 13,136 | 100.000 | 0.850 | 5.987 | 67.000 | 0.000 |
| 16 | api | 13,060 | 99.954 | 0.843 | 6.111 | 75.000 | 0.153 |
| 16 | fallback_docx | 6 | 0.046 | 0.854 | 5.753 | 57.000 | 0.000 |
| 17 | api | 60,303 | 99.993 | 0.820 | 6.158 | 78.000 | 0.337 |
| 17 | fallback_docx | 1 | 0.002 | 1.000 | 5.838 | 58.000 | 0.000 |
| 18 | docx | 33,117 | 95.510 | 0.771 | 6.420 | 101.000 | 17.749 |
| 18 | fallback_pdf | 1,516 | 4.372 | 0.756 | 6.406 | 101.000 | 15.369 |
| 18 | ocr | 35 | 0.101 | 0.752 | 6.166 | 70.000 | 0.000 |
| 18 | pdf | 5 | 0.014 | 0.900 | 6.336 | 104.000 | 20.000 |


Within LS18 only (n=34,674; S5 controls + Hindi flag + source_q dummies, reference = most frequent source): does the extraction source predict y_m2?

| term | 95% CI | p |
|---|---|---|
| n_parts | 0.011 [0.007, 0.016] | 0.0010 |
| log_qlen | -0.091 [-0.108, -0.074] | 0.0000 |
| starred | 0.045 [0.032, 0.058] | 0.0001 |
| q_hindi_f | 0.016 [-0.025, 0.057] | 0.3777 |
| source_q[fallback_pdf] | -0.017 [-0.045, 0.010] | 0.1778 |
| source_q[ocr] | -0.037 [-0.089, 0.014] | 0.1276 |
| source_q[pdf] | 0.143 [0.033, 0.253] | 0.0190 |


Does the question-length gradient show up in the human gold? Gold sample (400 questions), HT-weighted WLS of the question score on log question length, n_parts, starred and LS dummies; Rao-Wu bootstrap 95% CI. 'M2 minus gold' tests whether M2 over-penalises long questions.

| variable | outcome | coef [95% CI] |
|---|---|---|
| log_qlen | human gold | -0.160 [-0.271, -0.053] |
| log_qlen | M2 (same questions) | -0.166 [-0.281, -0.049] |
| log_qlen | M2 minus gold | -0.007 [-0.046, 0.037] |
| n_parts | human gold | 0.016 [-0.031, 0.061] |
| n_parts | M2 (same questions) | 0.026 [-0.021, 0.072] |
| n_parts | M2 minus gold | 0.011 [-0.010, 0.030] |
| starred | human gold | 0.056 [-0.009, 0.121] |
| starred | M2 (same questions) | 0.049 [-0.017, 0.115] |
| starred | M2 minus gold | -0.007 [-0.032, 0.019] |


## 3. Composition of evasion

### 3a. Rate per 100 labelled parts (design-weighted M2; session-cluster bootstrap 95% CI, B=2000, stratified by LS)

The evasion-type columns add up to 'ALL non-FULL' (= PARTIAL + NONE). OTHER pools OTHER, WITHHELD, DOES_NOT_ARISE, ANNEXURE_ONLY.

| group | GENERIC | INCOMPLETE | DIFFERENT_QUESTION | NOT_MAINTAINED | JURISDICTION | DEFERRAL | OTHER | PARTIAL | NONE | ALL non-FULL |
|---|---|---|---|---|---|---|---|---|---|---|
| LS13 | 6.55 [6.27, 6.86] | 4.77 [4.57, 4.96] | 1.37 [1.20, 1.56] | 1.16 [1.00, 1.32] | 0.71 [0.61, 0.81] | 4.69 [4.27, 5.18] | 1.08 [0.93, 1.21] | 9.85 [9.45, 10.22] | 10.49 [10.06, 11.03] | 20.34 [20.01, 20.65] |
| LS14 | 6.89 [6.53, 7.29] | 4.91 [4.62, 5.28] | 1.48 [1.35, 1.59] | 1.23 [1.09, 1.40] | 0.68 [0.54, 0.82] | 2.93 [2.40, 3.41] | 0.99 [0.88, 1.11] | 10.36 [9.90, 10.94] | 8.76 [8.00, 9.43] | 19.12 [18.12, 20.01] |
| LS15 | 7.58 [7.28, 7.91] | 6.37 [6.01, 6.77] | 1.58 [1.42, 1.74] | 2.08 [1.92, 2.25] | 0.97 [0.86, 1.08] | 2.25 [2.01, 2.54] | 0.89 [0.81, 0.96] | 12.96 [12.49, 13.46] | 8.77 [8.47, 9.08] | 21.73 [21.21, 22.27] |
| LS16 | 8.16 [7.84, 8.47] | 8.30 [8.09, 8.51] | 1.80 [1.59, 1.99] | 2.54 [2.38, 2.72] | 1.34 [1.19, 1.48] | 0.99 [0.76, 1.22] | 0.85 [0.77, 0.93] | 15.86 [15.49, 16.22] | 8.10 [7.80, 8.41] | 23.97 [23.49, 24.48] |
| LS17 | 11.13 [10.48, 11.81] | 8.74 [8.47, 8.97] | 2.85 [2.58, 3.33] | 2.16 [2.03, 2.29] | 1.61 [1.50, 1.73] | 0.17 [0.10, 0.25] | 0.66 [0.59, 0.72] | 18.09 [17.57, 18.64] | 9.23 [8.71, 9.88] | 27.32 [26.36, 28.32] |
| LS18 | 15.17 [14.45, 15.93] | 13.27 [11.91, 14.68] | 2.98 [2.90, 3.09] | 1.95 [1.84, 2.07] | 1.83 [1.64, 2.01] | 0.03 [0.01, 0.04] | 0.42 [0.40, 0.44] | 24.52 [22.72, 26.39] | 11.14 [10.58, 11.73] | 35.65 [33.44, 38.00] |
| early (LS13-14) | 6.72 [6.50, 6.96] | 4.84 [4.66, 5.04] | 1.42 [1.31, 1.54] | 1.19 [1.08, 1.31] | 0.70 [0.61, 0.79] | 3.85 [3.54, 4.17] | 1.04 [0.95, 1.13] | 10.09 [9.79, 10.42] | 9.66 [9.26, 10.03] | 19.76 [19.26, 20.19] |
| middle (LS15-16) | 7.86 [7.64, 8.08] | 7.31 [7.07, 7.56] | 1.69 [1.56, 1.82] | 2.30 [2.19, 2.42] | 1.15 [1.06, 1.25] | 1.63 [1.47, 1.82] | 0.87 [0.82, 0.93] | 14.38 [14.00, 14.72] | 8.44 [8.23, 8.67] | 22.82 [22.44, 23.23] |
| late (LS17-18) | 12.63 [11.99, 13.23] | 10.43 [9.80, 11.09] | 2.90 [2.72, 3.20] | 2.08 [1.99, 2.17] | 1.69 [1.59, 1.79] | 0.12 [0.07, 0.17] | 0.57 [0.52, 0.62] | 20.49 [19.57, 21.42] | 9.94 [9.50, 10.42] | 30.43 [29.16, 31.72] |


### 3b. Evasion-type share (%) among non-FULL parts, design-weighted M2

| group | GENERIC | INCOMPLETE | DIFFERENT_QUESTION | NOT_MAINTAINED | JURISDICTION | DEFERRAL | OTHER |
|---|---|---|---|---|---|---|---|
| LS13 | 32.2 [30.8, 33.7] | 23.5 [22.4, 24.5] | 6.7 [5.9, 7.7] | 5.7 [4.9, 6.5] | 3.5 [3.0, 3.9] | 23.1 [21.1, 25.2] | 5.3 [4.6, 6.0] |
| LS14 | 36.0 [35.0, 37.4] | 25.7 [24.2, 27.4] | 7.7 [7.1, 8.3] | 6.4 [5.8, 7.2] | 3.6 [2.9, 4.2] | 15.3 [12.8, 17.5] | 5.2 [4.6, 5.9] |
| LS15 | 34.9 [33.4, 36.6] | 29.3 [28.0, 30.8] | 7.3 [6.6, 7.9] | 9.6 [9.0, 10.2] | 4.5 [4.0, 5.0] | 10.4 [9.3, 11.7] | 4.1 [3.7, 4.4] |
| LS16 | 34.0 [33.1, 35.0] | 34.6 [33.9, 35.5] | 7.5 [6.7, 8.2] | 10.6 [9.9, 11.3] | 5.6 [4.9, 6.2] | 4.1 [3.1, 5.1] | 3.5 [3.2, 3.9] |
| LS17 | 40.7 [39.6, 41.9] | 32.0 [31.2, 32.7] | 10.4 [9.6, 11.9] | 7.9 [7.4, 8.4] | 5.9 [5.4, 6.4] | 0.6 [0.3, 0.9] | 2.4 [2.1, 2.7] |
| LS18 | 42.5 [41.7, 43.4] | 37.2 [35.6, 38.7] | 8.4 [7.8, 9.1] | 5.5 [5.2, 5.8] | 5.1 [4.9, 5.3] | 0.1 [0.0, 0.1] | 1.2 [1.1, 1.3] |
| early (LS13-14) | 34.0 [33.1, 35.0] | 24.5 [23.6, 25.4] | 7.2 [6.6, 7.8] | 6.0 [5.5, 6.6] | 3.5 [3.1, 4.0] | 19.5 [18.0, 20.9] | 5.3 [4.8, 5.7] |
| middle (LS15-16) | 34.5 [33.5, 35.4] | 32.0 [31.2, 32.9] | 7.4 [6.9, 7.9] | 10.1 [9.6, 10.5] | 5.0 [4.6, 5.4] | 7.2 [6.4, 8.0] | 3.8 [3.6, 4.1] |
| late (LS17-18) | 41.5 [40.7, 42.3] | 34.3 [33.3, 35.2] | 9.5 [9.0, 10.4] | 6.8 [6.4, 7.3] | 5.6 [5.3, 5.9] | 0.4 [0.2, 0.6] | 1.9 [1.7, 2.1] |


### 3c. Which types drive the rise? Change in rate per 100 parts (percentage points), with CI

| contrast | type | change (pp of parts) | % of total rise |
|---|---|---|---|
| LS18 - LS13 | GENERIC | 8.61 [7.81, 9.44] | 56.2 |
| LS18 - LS13 | INCOMPLETE | 8.50 [7.17, 9.92] | 55.5 |
| LS18 - LS13 | DIFFERENT_QUESTION | 1.61 [1.40, 1.81] | 10.5 |
| LS18 - LS13 | NOT_MAINTAINED | 0.79 [0.60, 0.99] | 5.2 |
| LS18 - LS13 | JURISDICTION | 1.12 [0.91, 1.32] | 7.3 |
| LS18 - LS13 | DEFERRAL | -4.66 [-5.15, -4.25] | -30.5 |
| LS18 - LS13 | OTHER | -0.66 [-0.79, -0.51] | -4.3 |
| LS18 - LS13 | PARTIAL | 14.67 [12.82, 16.53] |  |
| LS18 - LS13 | NONE | 0.65 [-0.16, 1.38] |  |
| LS18 - LS13 | ALL non-FULL | 15.31 [13.07, 17.70] |  |
| LS17 - LS13 | GENERIC | 4.57 [3.86, 5.34] | 65.5 |
| LS17 - LS13 | INCOMPLETE | 3.97 [3.64, 4.28] | 56.8 |
| LS17 - LS13 | DIFFERENT_QUESTION | 1.48 [1.13, 1.99] | 21.2 |
| LS17 - LS13 | NOT_MAINTAINED | 1.00 [0.80, 1.20] | 14.4 |
| LS17 - LS13 | JURISDICTION | 0.90 [0.75, 1.05] | 12.9 |
| LS17 - LS13 | DEFERRAL | -4.52 [-5.02, -4.09] | -64.7 |
| LS17 - LS13 | OTHER | -0.42 [-0.57, -0.27] | -6.0 |
| LS17 - LS13 | PARTIAL | 8.24 [7.60, 8.93] |  |
| LS17 - LS13 | NONE | -1.26 [-2.02, -0.51] |  |
| LS17 - LS13 | ALL non-FULL | 6.98 [5.94, 8.02] |  |
| late (LS17-18) - early (LS13-14) | GENERIC | 5.92 [5.24, 6.56] | 55.4 |
| late (LS17-18) - early (LS13-14) | INCOMPLETE | 5.59 [4.93, 6.30] | 52.4 |
| late (LS17-18) - early (LS13-14) | DIFFERENT_QUESTION | 1.48 [1.26, 1.79] | 13.8 |
| late (LS17-18) - early (LS13-14) | NOT_MAINTAINED | 0.89 [0.74, 1.03] | 8.3 |
| late (LS17-18) - early (LS13-14) | JURISDICTION | 1.00 [0.86, 1.13] | 9.3 |
| late (LS17-18) - early (LS13-14) | DEFERRAL | -3.73 [-4.05, -3.42] | -34.9 |
| late (LS17-18) - early (LS13-14) | OTHER | -0.47 [-0.57, -0.37] | -4.4 |
| late (LS17-18) - early (LS13-14) | PARTIAL | 10.39 [9.43, 11.37] |  |
| late (LS17-18) - early (LS13-14) | NONE | 0.28 [-0.31, 0.92] |  |
| late (LS17-18) - early (LS13-14) | ALL non-FULL | 10.67 [9.33, 12.09] |  |
| LS18 - LS17 | GENERIC | 4.04 [3.01, 5.03] | 48.5 |
| LS18 - LS17 | INCOMPLETE | 4.54 [3.15, 5.95] | 54.4 |
| LS18 - LS17 | DIFFERENT_QUESTION | 0.14 [-0.35, 0.43] | 1.6 |
| LS18 - LS17 | NOT_MAINTAINED | -0.21 [-0.38, -0.03] | -2.5 |
| LS18 - LS17 | JURISDICTION | 0.21 [-0.01, 0.45] | 2.6 |
| LS18 - LS17 | DEFERRAL | -0.15 [-0.23, -0.07] | -1.8 |
| LS18 - LS17 | OTHER | -0.24 [-0.31, -0.17] | -2.9 |
| LS18 - LS17 | PARTIAL | 6.43 [4.55, 8.34] |  |
| LS18 - LS17 | NONE | 1.91 [1.07, 2.71] |  |
| LS18 - LS17 | ALL non-FULL | 8.33 [5.82, 10.84] |  |


### 3d. Composition vs within-type change at the part level (request type, M2-labelled)

| request_type | share early % | nonFULL rate early % | share late % | nonFULL rate late % |
|---|---|---|---|---|
| INFORMATION | 51.4 | 21.3 | 47.4 | 33.6 |
| ACTION | 37.9 | 17.3 | 43.6 | 25.4 |
| REASON | 7.2 | 20.4 | 6.4 | 38.3 |
| OPINION | 3.4 | 21.8 | 2.7 | 37.3 |


Kitagawa decomposition of the early -> late change in the non-FULL part rate (19.8% -> 30.4%, +10.7 pp): composition across request types -0.38 pp, within-type +11.05 pp.

### 3e. Gold-based evasion-type shares (% of non-FULL parts) by period, with M2 labels on the same gold parts for comparison

HT-weighted (1/gold_pi), Rao-Wu bootstrap 95% CI. 'nonFULL rate' = % of parts that are not FULL. Small n: treat as indicative.

| period | source | nonFULL rate % | GENERIC | INCOMPLETE | DIFFERENT_QUESTION | NOT_MAINTAINED | JURISDICTION | DEFERRAL | OTHER |
|---|---|---|---|---|---|---|---|---|---|
| early (LS13-14, 1999-2009) | human gold | 22.5 [16.1, 29.8] | 39.8 [23.3, 59.0] | 15.3 [8.1, 24.9] | 17.2 [6.6, 30.2] | 4.9 [0.0, 11.4] | 0.0 [0.0, 0.0] | 20.5 [0.0, 39.5] | 2.3 [0.0, 5.2] |
| early (LS13-14, 1999-2009) | M2 on the same parts | 20.7 [14.4, 28.1] | 34.2 [19.1, 52.9] | 28.3 [16.4, 43.7] | 9.8 [2.0, 20.7] | 5.4 [0.0, 11.8] | 0.0 [0.0, 0.0] | 22.3 [0.0, 42.7] | 0.0 [0.0, 0.0] |
| middle (LS15-16, 2009-2019) | human gold | 29.1 [21.6, 36.8] | 25.1 [13.8, 36.5] | 29.8 [18.9, 43.3] | 8.9 [3.5, 15.7] | 11.4 [5.2, 19.4] | 7.0 [0.1, 15.3] | 7.0 [0.0, 19.2] | 10.8 [2.7, 19.3] |
| middle (LS15-16, 2009-2019) | M2 on the same parts | 27.0 [18.9, 35.0] | 26.8 [13.7, 40.1] | 32.6 [19.6, 47.9] | 6.4 [0.1, 17.0] | 12.3 [5.9, 20.5] | 7.1 [0.0, 15.7] | 7.5 [0.0, 20.8] | 7.4 [0.0, 15.0] |
| late (LS17-18, 2019-2026) | human gold | 33.3 [28.2, 38.5] | 27.9 [20.3, 35.7] | 30.2 [23.7, 37.2] | 17.5 [12.1, 23.7] | 5.7 [2.5, 9.5] | 7.8 [3.2, 13.7] | 2.0 [0.0, 6.0] | 8.8 [3.9, 14.7] |
| late (LS17-18, 2019-2026) | M2 on the same parts | 29.9 [25.2, 34.9] | 41.5 [32.5, 50.1] | 33.7 [26.6, 41.6] | 9.8 [5.0, 15.3] | 6.3 [2.7, 10.5] | 2.2 [0.4, 4.5] | 2.2 [0.0, 6.6] | 4.3 [0.8, 8.8] |
| all | human gold | 28.0 [24.0, 32.1] | 29.9 [22.9, 36.9] | 26.0 [20.1, 32.3] | 13.7 [9.3, 18.1] | 7.9 [4.7, 11.7] | 5.4 [2.1, 9.2] | 9.1 [2.1, 17.1] | 7.9 [3.9, 12.5] |
| all | M2 on the same parts | 25.6 [21.6, 29.9] | 33.1 [25.4, 41.2] | 31.7 [24.4, 39.3] | 8.4 [4.1, 13.8] | 8.6 [5.2, 12.5] | 3.7 [0.8, 7.4] | 10.0 [2.3, 18.6] | 4.5 [1.1, 8.4] |


Gold vs M2 on the same parts: rate per 100 parts, early (LS13-14) vs late (LS17-18):

| source | type | rate early | rate late | change late - early (pp of parts) [95% CI] |
|---|---|---|---|---|
| human gold | GENERIC | 9.0 | 9.3 | 0.3 [-4.9, 5.4] |
| human gold | INCOMPLETE | 3.5 | 10.1 | 6.6 [3.6, 9.8] |
| human gold | DIFFERENT_QUESTION | 3.9 | 5.8 | 2.0 [-1.2, 5.1] |
| human gold | NOT_MAINTAINED | 1.1 | 1.9 | 0.8 [-1.0, 2.5] |
| human gold | JURISDICTION | 0.0 | 2.6 | 2.6 [1.1, 4.5] |
| human gold | DEFERRAL | 4.6 | 0.7 | -4.0 [-10.5, 0.8] |
| human gold | OTHER | 0.5 | 2.9 | 2.4 [0.6, 4.6] |
| human gold | nonFULL | 22.5 | 33.3 | 10.8 [2.3, 19.2] |
| M2 on same parts | GENERIC | 7.1 | 12.4 | 5.4 [0.5, 10.2] |
| M2 on same parts | INCOMPLETE | 5.9 | 10.1 | 4.2 [0.4, 7.9] |
| M2 on same parts | DIFFERENT_QUESTION | 2.0 | 2.9 | 0.9 [-1.5, 3.1] |
| M2 on same parts | NOT_MAINTAINED | 1.1 | 1.9 | 0.8 [-1.1, 2.5] |
| M2 on same parts | JURISDICTION | 0.0 | 0.7 | 0.7 [0.1, 1.3] |
| M2 on same parts | DEFERRAL | 4.6 | 0.7 | -4.0 [-10.5, 0.8] |
| M2 on same parts | OTHER | 0.0 | 1.3 | 1.3 [0.2, 2.7] |
| M2 on same parts | nonFULL | 20.7 | 29.9 | 9.2 [0.6, 17.2] |


Evasion-type agreement among parts that are non-FULL in both (types as defined above):

| period | parts non-FULL in both | evasion-type agreement (unweighted) | gold DEFERRAL parts | of which M2 DEFERRAL |
|---|---|---|---|---|
| early (LS13-14, 1999-2009) | 74 | 0.919 | 13 | 13 |
| middle (LS15-16, 2009-2019) | 106 | 0.792 | 8 | 8 |
| late (LS17-18, 2019-2026) | 227 | 0.806 | 4 | 4 |
| all | 407 | 0.823 | 25 | 25 |


### 3f. Is the decline of DEFERRAL real? Independent text-marker check

Regex flags computed on the answer text of every analysed question (no text is stored or quoted). Marker A: wording of information 'being collected / compiled / gathered / collated'. Marker B: wording that the information 'will be laid on the Table'. 'Both' is the classic deferral formula. Rates are % of questions (census frame, unweighted; LS13-16 are the full population here, not the 1/6 sample).

| LS | n questions | % marker A | % marker B | % both (formula) | M2 sample: % questions with any DEFERRAL part (weighted) | M2 sample: % formula questions (weighted) | among formula Qs: % with any M2 DEFERRAL (weighted) |
|---|---|---|---|---|---|---|---|
| 13 | 72,987 | 4.77 | 4.03 | 3.92 | 5.11 | 4.17 | 99.80 |
| 14 | 65,969 | 3.34 | 2.70 | 2.62 | 3.43 | 2.65 | 99.66 |
| 15 | 78,749 | 2.74 | 2.05 | 2.02 | 2.77 | 1.94 | 100.00 |
| 16 | 78,454 | 1.26 | 0.90 | 0.88 | 1.26 | 0.93 | 100.00 |
| 17 | 60,307 | 0.33 | 0.15 | 0.15 | 0.25 | 0.15 | 100.00 |
| 18 | 34,674 | 0.10 | 0.02 | 0.02 | 0.03 | 0.02 | 100.00 |


Mean y_m2 of questions carrying the formula vs the rest:

| LS | formula questions | other questions |
|---|---|---|
| 13 | 0.113 | 0.883 |
| 14 | 0.104 | 0.885 |
| 15 | 0.134 | 0.864 |
| 16 | 0.173 | 0.849 |
| 17 | 0.193 | 0.821 |
| 18 | 0.354 | 0.771 |


Agreement between the text marker and M2 DEFERRAL (design-weighted, all LS): among questions with any M2 DEFERRAL part, 76.4% carry the formula; among formula questions, 99.8% have an M2 DEFERRAL part.

### 3g. How much does the vanishing deferral formula mask? y_m2 regressions with and without formula questions

| sample | spec | n | LS17 vs LS13 | LS18 vs LS13 | slope per decade |
|---|---|---|---|---|---|
| all questions | S0 none | 144,349 | -0.031 [-0.041, -0.021] | -0.080 [-0.097, -0.063] | -0.0263 [-0.0343, -0.0183] |
| all questions | S5 full controls | 144,349 | -0.016 [-0.026, -0.007] | -0.051 [-0.063, -0.038] | -0.0133 [-0.0204, -0.0062] |
| excluding deferral-formula questions | S0 none | 143,078 | -0.062 [-0.071, -0.053] | -0.112 [-0.129, -0.095] | -0.0405 [-0.0476, -0.0333] |
| excluding deferral-formula questions | S5 full controls | 143,078 | -0.049 [-0.058, -0.040] | -0.087 [-0.099, -0.075] | -0.0294 [-0.0355, -0.0233] |


Figure: `rq2_fig2_evasion_composition.png`

## 4. Within-government trends

78 sessions (LS13: 13, LS14: 14, LS15: 14, LS16: 16, LS17: 14, LS18: 7); design-weighted mean y_m2 per session, SE (linearised, iid within session) median 0.0074, range 0.0028-0.0479. 'Adjusted' = session fixed effects from a WLS with n_parts, log_qlen, starred, request-type shares, ministry_canonical and topic FE (S5 controls) evaluated at the overall design-weighted covariate mean. Break dates = first sitting of the new Lok Sabha (the government changes at the election in between, there are no sittings between February and July).

### 4a. Linear trend within each government period (session-level, equal-weighted OLS, HC3 CIs)

| govt_period | series | LS | sessions | first sitting | last | mean level | slope/yr [95% CI, HC3] |
|---|---|---|---|---|---|---|---|
| NDA-V3 | unadjusted | 13 | 13 | 1999.9 | 2004.0 | 0.853 | -0.0042 [-0.0153, 0.0070] |
| NDA-V3 | adjusted | 13 | 13 | 1999.9 | 2004.0 | 0.843 | -0.0020 [-0.0087, 0.0047] |
| UPA-1 | unadjusted | 14 | 14 | 2004.6 | 2009.1 | 0.866 | 0.0012 [-0.0043, 0.0066] |
| UPA-1 | adjusted | 14 | 14 | 2004.6 | 2009.1 | 0.852 | 0.0005 [-0.0043, 0.0053] |
| UPA-2 | unadjusted | 15 | 14 | 2009.6 | 2014.0 | 0.849 | -0.0019 [-0.0062, 0.0025] |
| UPA-2 | adjusted | 15 | 14 | 2009.6 | 2014.0 | 0.848 | 0.0026 [-0.0016, 0.0068] |
| NDA-M1 | unadjusted | 16 | 16 | 2014.6 | 2019.1 | 0.842 | -0.0028 [-0.0055, -0.0001] |
| NDA-M1 | adjusted | 16 | 16 | 2014.6 | 2019.1 | 0.849 | -0.0017 [-0.0050, 0.0015] |
| NDA-M2 | unadjusted | 17 | 14 | 2019.5 | 2024.1 | 0.817 | -0.0093 [-0.0190, 0.0005] |
| NDA-M2 | adjusted | 17 | 14 | 2019.5 | 2024.1 | 0.823 | -0.0086 [-0.0177, 0.0005] |
| NDA-M3 | unadjusted | 18 | 7 | 2024.6 | 2026.6 | 0.772 | -0.0262 [-0.0409, -0.0115] |
| NDA-M3 | adjusted | 18 | 7 | 2024.6 | 2026.6 | 0.791 | -0.0124 [-0.0244, -0.0005] |


### 4b. Segmented check at each election: y = a + b1 (t-t0) + g POST + d POST (t-t0), sessions within +-5 years, HC3, t(n-4)

g = level jump at t0 (extrapolating each side's line); d = change in slope per year. The extrapolation gap is the mean of (actual - pre-trend prediction) over the post sessions, pre-trend fitted on the preceding 5 years only.

| series | break | type | sessions (pre/post) | pre-slope/yr | jump [95% CI] | p(jump) | slope change/yr [95% CI] | p(slope chg) | pre-trend extrapolation gap [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| unadjusted | 2004 (NDA-V3 -> UPA-1) | government change | 13/14 | -0.0042 [-0.0147, 0.0063] | 0.0206 [-0.0068, 0.0479] | 0.134 | 0.0053 [-0.0063, 0.0170] | 0.354 | 0.0328 [-0.0017, 0.0673] |
| unadjusted | 2014 (UPA-2 -> NDA-M1) | government change | 14/16 | -0.0019 [-0.0060, 0.0022] | 0.0045 [-0.0100, 0.0191] | 0.528 | -0.0010 [-0.0058, 0.0039] | 0.687 | 0.0023 [-0.0169, 0.0214] |
| unadjusted | 2024 (NDA-M2 -> NDA-M3) | government change | 13/7 | -0.0101 [-0.0215, 0.0014] | 0.0067 [-0.0315, 0.0448] | 0.716 | -0.0161 [-0.0328, 0.0005] | 0.057 | -0.0097 [-0.0404, 0.0211] |
| unadjusted | 2009 (UPA-1 -> UPA-2, same govt) | placebo (no change of govt) | 14/14 | 0.0012 [-0.0040, 0.0063] | -0.0154 [-0.0355, 0.0046] | 0.125 | -0.0030 [-0.0096, 0.0036] | 0.355 | -0.0223 [-0.0561, 0.0115] |
| unadjusted | 2019 (NDA-M1 -> NDA-M2, same govt) | placebo (no change of govt) | 16/14 | -0.0028 [-0.0054, -0.0002] | 0.0055 [-0.0132, 0.0241] | 0.553 | -0.0064 [-0.0160, 0.0031] | 0.177 | -0.0106 [-0.0282, 0.0070] |
| adjusted | 2004 (NDA-V3 -> UPA-1) | government change | 13/14 | -0.0020 [-0.0083, 0.0043] | 0.0127 [-0.0076, 0.0331] | 0.209 | 0.0025 [-0.0052, 0.0103] | 0.505 | 0.0185 [-0.0039, 0.0410] |
| adjusted | 2014 (UPA-2 -> NDA-M1) | government change | 14/16 | 0.0026 [-0.0013, 0.0065] | -0.0022 [-0.0179, 0.0134] | 0.772 | -0.0043 [-0.0094, 0.0007] | 0.088 | -0.0125 [-0.0311, 0.0061] |
| adjusted | 2024 (NDA-M2 -> NDA-M3) | government change | 13/7 | -0.0096 [-0.0203, 0.0010] | 0.0048 [-0.0291, 0.0387] | 0.768 | -0.0028 [-0.0173, 0.0117] | 0.686 | 0.0019 [-0.0264, 0.0303] |
| adjusted | 2009 (UPA-1 -> UPA-2, same govt) | placebo (no change of govt) | 14/14 | 0.0005 [-0.0040, 0.0051] | -0.0116 [-0.0298, 0.0067] | 0.203 | 0.0021 [-0.0040, 0.0081] | 0.485 | -0.0069 [-0.0371, 0.0234] |
| adjusted | 2019 (NDA-M1 -> NDA-M2, same govt) | placebo (no change of govt) | 16/14 | -0.0017 [-0.0049, 0.0014] | 0.0002 [-0.0184, 0.0189] | 0.979 | -0.0069 [-0.0160, 0.0023] | 0.136 | -0.0169 [-0.0342, 0.0005] |


Sensitivity: precision-weighted (weights 1/(se^2 + 0.02^2)), unadjusted series:

| break | jump [95% CI] (precision-weighted) | slope change/yr |
|---|---|---|
| 2004 (NDA-V3 -> UPA-1) | 0.0150 [-0.0048, 0.0347] | 0.0018 [-0.0054, 0.0090] |
| 2014 (UPA-2 -> NDA-M1) | 0.0044 [-0.0101, 0.0188] | -0.0009 [-0.0057, 0.0039] |
| 2024 (NDA-M2 -> NDA-M3) | 0.0055 [-0.0302, 0.0412] | -0.0165 [-0.0328, -0.0003] |
| 2009 (UPA-1 -> UPA-2, same govt) | -0.0148 [-0.0358, 0.0062] | -0.0028 [-0.0096, 0.0040] |
| 2019 (NDA-M1 -> NDA-M2, same govt) | 0.0046 [-0.0131, 0.0223] | -0.0062 [-0.0151, 0.0027] |


### 4c. Does a smooth trend fit as well as a trend with breaks? (78 session means, unadjusted)

| model | k | R2 | AIC | LOO-CV RMSE | resid SD |
|---|---|---|---|---|---|
| linear | 2 | 0.5718 | -611.3935 | 0.0201 | 0.0196 |
| quadratic | 3 | 0.7849 | -663.0885 | 0.0145 | 0.0140 |
| cubic | 4 | 0.8108 | -671.0848 | 0.0138 | 0.0132 |
| linear + jumps & slope changes at 3 govt changes | 8 | 0.8363 | -674.3892 | 0.0137 | 0.0126 |
| linear + jumps only at 3 govt changes | 5 | 0.8031 | -665.9833 | 0.0142 | 0.0136 |
| linear + jumps & slope changes at 2 placebo dates | 6 | 0.8282 | -674.6128 | 0.0134 | 0.0128 |
| linear + jumps only at 2 placebo dates | 4 | 0.6504 | -623.1946 | 0.0185 | 0.0180 |
| linear + jumps at all 5 LS boundaries (jumps only) | 7 | 0.8065 | -663.3325 | 0.0144 | 0.0136 |


Joint HC3 Wald tests of break terms added to a smooth cubic time trend (null = smooth nonlinear trend):

| joint test (all break terms = 0) | Wald chi2 | df | p (F approx) |
|---|---|---|---|
| 3 govt changes: jumps only (on top of cubic) | 7.270 | 3 | 0.073 |
| 3 govt changes: jumps + slope changes (on top of cubic) | 16.862 | 6 | 0.017 |
| 2 placebo dates: jumps only (on top of cubic) | 12.387 | 2 | 0.003 |
| 2 placebo dates: jumps + slope changes (on top of cubic) | 13.616 | 4 | 0.013 |


Leave-one-session-out RMSE of cubic + break terms (cubic alone: 0.0138):

| model | k | LOO-CV RMSE |
|---|---|---|
| cubic + 3 govt changes: jumps only | 7 | 0.0136 |
| cubic + 3 govt changes: jumps + slope changes | 10 | 0.0138 |
| cubic + 2 placebo dates: jumps only | 6 | 0.0130 |
| cubic + 2 placebo dates: jumps + slope changes | 8 | 0.0131 |


### 4c2. A single estimated kink (hinge) instead of election breaks

y = a + c * max(t - kappa, 0) (flat before kappa), kappa searched on 2008-2025: kappa_hat = 2018.3 (residual-bootstrap 95% interval 2016.9 to 2019.7, B=500); slope after kappa = -0.0108 per year; R2 = 0.770; LOO RMSE = 0.0146 (vs cubic 0.0138, 3 govt-change breaks 0.0137). Linear trend + kink: pre-kink slope -0.0008/yr, extra slope after kappa -0.0087/yr, LOO RMSE 0.0142.

### 4d. Unknown-break scan (sup-F for a level + slope break in a linear trend)

sup-F=47.42 at t=2020.9 (candidate breaks between sessions, 15% trimming); residual-bootstrap p=0.000 (B=500, iid residuals, sessions treated as exchangeable so p is approximate). Top-5 candidate dates: 2020.9 (F=47.4), 2021.4 (F=47.4), 2021.8 (F=47.2), 2022.1 (F=46.8), 2020.4 (F=46.5). Government changes in decimal years: 2004=2004.5, 2014=2014.5, 2024=2024.6.

Figure: `rq2_fig3_sessions_breaks.png`
