# Within-MP test of the decline (exploratory)

Question: does the fall in the LLM question score reflect a change in WHO asks (composition of askers) rather than in how ministers answer?

Method: outcome y_m2 (0 to 1), design weights m2_weight, WLS, two-way clustered SEs (first-asker MP x session), t(min G - 1) CIs, exactly as explore_rq2.py (task 2). LS contrasts use LS dummies (reference LS13); the trend uses (year - 2000) / 10. Controls = the S5 specification of explore_rq2.py (n_parts, log question length, starred, request-type shares, ministry and topic fixed effects). MP fixed effects = dummies for mpsno of the first asker (multi-asker questions are assigned to the first asker, the project convention). Estimates are in score units, 95% CIs in brackets.

Sample: 144,349 labelled questions; 143,547 (99.4%) have a resolved mpsno for the first asker (1,894 MPs). Across the whole corpus (all askers) 629 of 1,992 MPs with a resolved mpsno appear in two or more Lok Sabhas. In the labelled sample, 587 MPs (first asker) ask in 2+ Lok Sabhas and 251 ask in both an early (LS13 to LS16) and a late (LS17 or LS18) Lok Sabha.

Identification note: with MP fixed effects the Lok Sabha contrasts and the trend are identified only from MPs observed in several Lok Sabhas (an MP seen in one Lok Sabha only absorbs his or her own level and contributes nothing to the contrasts). In the raw model the point estimates on S_res and S_multi are therefore identical in principle; the controlled model differs slightly because the control coefficients also use within MP variation of single Lok Sabha MPs. The informative comparison is always the no FE estimate on the SAME sample (rows 'no MP FE').

## 1 to 3. Baseline and MP fixed effects on the resolved mpsno sample

FACTS reference (all labelled questions): raw LS18 minus LS13 -0.080 [-0.097, -0.063], trend -0.026 [-0.034, -0.018]; controlled -0.051 [-0.063, -0.038], trend -0.013 [-0.020, -0.006].

### S_all (all labelled questions, no MP FE; replication of FACTS)

| spec | N questions | N MPs | LS18 minus LS13 | trend per decade |
|---|---|---|---|---|
| no MP FE raw | 144,349 | 1,918 | -0.080 [-0.097, -0.063] | -0.026 [-0.034, -0.018] |
| no MP FE + controls | 144,349 | 1,918 | -0.051 [-0.063, -0.038] | -0.013 [-0.020, -0.006] |

### S_res (resolved mpsno), key contrasts

| spec | N questions | N MPs | LS18 minus LS13 | trend per decade |
|---|---|---|---|---|
| no MP FE raw | 143,547 | 1,894 | -0.080 [-0.097, -0.063] | -0.026 [-0.034, -0.018] |
| MP FE raw | 143,547 | 1,894 | -0.081 [-0.102, -0.060] | -0.032 [-0.042, -0.023] |
| no MP FE + controls | 143,547 | 1,894 | -0.051 [-0.063, -0.038] | -0.013 [-0.021, -0.006] |
| MP FE + controls | 143,547 | 1,894 | -0.048 [-0.066, -0.031] | -0.018 [-0.027, -0.009] |

### S_res, every Lok Sabha against LS13 and trend

| spec | N questions | N MPs | LS14 minus LS13 | LS15 minus LS13 | LS16 minus LS13 | LS17 minus LS13 | LS18 minus LS13 | trend per decade |
|---|---|---|---|---|---|---|---|---|
| no MP FE raw | 143,547 | 1,894 | 0.014 [0.005, 0.023] | -0.001 [-0.008, 0.007] | -0.008 [-0.016, -0.000] | -0.031 [-0.041, -0.021] | -0.080 [-0.097, -0.063] | -0.026 [-0.034, -0.018] |
| MP FE raw | 143,547 | 1,894 | 0.012 [0.000, 0.023] | -0.007 [-0.019, 0.006] | -0.020 [-0.035, -0.005] | -0.040 [-0.055, -0.024] | -0.081 [-0.102, -0.060] | -0.032 [-0.042, -0.023] |
| no MP FE + controls | 143,547 | 1,894 | 0.009 [0.000, 0.017] | 0.007 [-0.001, 0.015] | 0.007 [-0.001, 0.015] | -0.016 [-0.026, -0.007] | -0.051 [-0.063, -0.038] | -0.013 [-0.021, -0.006] |
| MP FE + controls | 143,547 | 1,894 | 0.009 [-0.002, 0.020] | 0.004 [-0.009, 0.017] | 0.003 [-0.012, 0.017] | -0.020 [-0.035, -0.004] | -0.048 [-0.066, -0.031] | -0.018 [-0.027, -0.009] |

## 4. Restricted to MPs observed in several Lok Sabhas

### S_multi (MPs asking in 2+ Lok Sabhas)

| spec | N questions | N MPs | LS14 minus LS13 | LS15 minus LS13 | LS16 minus LS13 | LS17 minus LS13 | LS18 minus LS13 | trend per decade |
|---|---|---|---|---|---|---|---|---|
| no MP FE raw | 83,769 | 587 | 0.017 [0.006, 0.028] | 0.000 [-0.011, 0.011] | -0.009 [-0.020, 0.002] | -0.031 [-0.044, -0.019] | -0.076 [-0.094, -0.057] | -0.027 [-0.035, -0.019] |
| MP FE raw | 83,769 | 587 | 0.012 [-0.001, 0.024] | -0.007 [-0.020, 0.006] | -0.020 [-0.035, -0.004] | -0.040 [-0.056, -0.024] | -0.081 [-0.103, -0.059] | -0.032 [-0.042, -0.022] |
| no MP FE + controls | 83,769 | 587 | 0.011 [0.000, 0.021] | 0.007 [-0.004, 0.018] | 0.006 [-0.005, 0.018] | -0.019 [-0.031, -0.006] | -0.049 [-0.064, -0.034] | -0.015 [-0.023, -0.007] |
| MP FE + controls | 83,769 | 587 | 0.008 [-0.005, 0.022] | 0.003 [-0.011, 0.017] | 0.001 [-0.015, 0.017] | -0.021 [-0.038, -0.005] | -0.050 [-0.070, -0.031] | -0.018 [-0.028, -0.008] |

### S_both (MPs asking in both early LS13 to LS16 and late LS17 or LS18)

| spec | N questions | N MPs | LS14 minus LS13 | LS15 minus LS13 | LS16 minus LS13 | LS17 minus LS13 | LS18 minus LS13 | trend per decade |
|---|---|---|---|---|---|---|---|---|
| no MP FE raw | 49,248 | 251 | 0.027 [-0.002, 0.055] | 0.010 [-0.014, 0.034] | -0.001 [-0.025, 0.022] | -0.021 [-0.045, 0.003] | -0.067 [-0.097, -0.037] | -0.030 [-0.040, -0.020] |
| MP FE raw | 49,248 | 251 | 0.035 [0.002, 0.068] | 0.013 [-0.014, 0.040] | -0.007 [-0.035, 0.021] | -0.024 [-0.051, 0.004] | -0.070 [-0.104, -0.036] | -0.038 [-0.050, -0.026] |
| no MP FE + controls | 49,248 | 251 | 0.015 [-0.013, 0.042] | 0.016 [-0.008, 0.041] | 0.014 [-0.011, 0.038] | -0.010 [-0.035, 0.014] | -0.042 [-0.071, -0.014] | -0.019 [-0.029, -0.008] |
| MP FE + controls | 49,248 | 251 | 0.026 [-0.008, 0.059] | 0.021 [-0.007, 0.049] | 0.013 [-0.016, 0.042] | -0.008 [-0.038, 0.021] | -0.041 [-0.074, -0.008] | -0.026 [-0.038, -0.013] |

## Share of the decline that remains within MP (ratio of point estimates)

| sample | contrast | MP FE raw / no FE raw | MP FE controlled / no FE controlled | MP FE controlled / no FE raw |
|---|---|---|---|---|
| S_res | LS18 minus LS13 | 1.013 | 0.954 | 0.606 |
| S_res | trend per decade | 1.231 | 1.320 | 0.672 |
| S_multi | LS18 minus LS13 | 1.073 | 1.021 | 0.667 |
| S_multi | trend per decade | 1.192 | 1.228 | 0.683 |
| S_both | LS18 minus LS13 | 1.043 | 0.981 | 0.618 |
| S_both | trend per decade | 1.260 | 1.374 | 0.850 |

The first ratio is the share of the raw decline that survives holding the identity of the asker fixed. The second is the share of the controlled decline that survives; the third compares the fully adjusted within MP estimate with the raw no FE estimate (all on the same sample).

## Composition of askers by Lok Sabha (all analysed questions, resolved first asker)

| LS | questions | % resolved MP id | MPs asking | % questions by MPs who asked in an earlier LS | % questions by MPs who ask again in a later LS |
|---|---|---|---|---|---|
| 13 | 72,987 | 99.8 | 471 | 0.0 | 47.7 |
| 14 | 65,969 | 99.9 | 470 | 35.2 | 38.9 |
| 15 | 78,749 | 98.8 | 448 | 34.5 | 39.8 |
| 16 | 78,454 | 99.2 | 495 | 34.0 | 43.6 |
| 17 | 60,307 | 99.2 | 499 | 49.1 | 37.6 |
| 18 | 34,674 | 100.0 | 466 | 47.3 | 0.0 |

## LS18 session coverage

LS18 sessions present in the data: 2, 3, 4, 5, 6, 7, 8 (7 sessions with questions; session 1 in June 2024 was the oath and Speaker election session without a question hour). Date range of questions: 2024-07-22 to 2026-08-12. 34,720 questions in the corpus, 34,674 labelled.

| session | first | last | questions | labelled (M2) |
|---|---|---|---|---|
| 2 | 2024-07-22 | 2024-08-09 | 3,499 | 3,498 |
| 3 | 2024-11-25 | 2024-12-20 | 4,749 | 4,745 |
| 4 | 2025-02-03 | 2025-04-04 | 6,250 | 6,233 |
| 5 | 2025-07-21 | 2025-08-21 | 5,248 | 5,237 |
| 6 | 2025-12-01 | 2025-12-18 | 3,499 | 3,497 |
| 7 | 2026-01-29 | 2026-04-02 | 6,975 | 6,969 |
| 8 | 2026-07-20 | 2026-08-12 | 4,500 | 4,495 |

Full estimate table: results/explore/within_mp.csv.
