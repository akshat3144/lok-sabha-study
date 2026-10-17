# RQ3 (exploratory): who asks, and does the opposition gap depend on the question?

Statistics only. Outcome y_m2 (0-1) unless stated; OLS with the models.py H1/H3 controls (n_parts, log_qlen, co-askers, request-type shares), ministry_group + topic (K=50) + session FE; design weights m2_weight (pooled LS13-18; LS17/18 are a census, weight 1); SEs two-way clustered by first-asker MP and session, t(min G-1) CIs. Reference = government askers. Every model run is in `rq3_all_models.csv` (245 coefficient rows logged); p-values are raw and unadjusted for the many comparisons here.

Sample: 144349 questions with a design weight (LS13 12170, LS14 10996, LS15 13136, LS16 13066, LS17 60307, LS18 34674); weighted shares in the tables below.


## 1. Opposition gap by Lok Sabha

| LS | estimate | CI two-way | p | CI MP-only | p_mp | n | n_opp | share_opp_w | sessions | mean_y |
|---|---|---|---|---|---|---|---|---|---|---|
| LS13 | 0.0212 | [+0.0051, +0.0374] | 0.0144 | [+0.0081, +0.0344] | 0.0017 | 12,170 | 5,286 | 0.4344 | 13 | 0.8506 |
| LS14 | -0.0100 | [-0.0258, +0.0057] | 0.1925 | [-0.0209, +0.0008] | 0.0691 | 10,996 | 4,177 | 0.3799 | 14 | 0.8643 |
| LS15 | -0.0035 | [-0.0176, +0.0105] | 0.5988 | [-0.0135, +0.0065] | 0.4901 | 13,136 | 5,436 | 0.4138 | 14 | 0.8496 |
| LS16 | 0.0091 | [-0.0026, +0.0208] | 0.1182 | [-0.0005, +0.0187] | 0.0644 | 13,066 | 4,382 | 0.3354 | 16 | 0.8427 |
| LS17 | -0.0110 | [-0.0178, -0.0042] | 0.0040 | [-0.0165, -0.0055] | 0.0001 | 60,307 | 26,008 | 0.4313 | 14 | 0.8196 |
| LS18 | -0.0384 | [-0.0501, -0.0266] | 0.0002 | [-0.0462, -0.0305] | 0.0000 | 34,674 | 18,243 | 0.5261 | 7 | 0.7705 |


(`estimate` = coefficient on opposition askers, controls + ministry_group, topic and within-LS session FE; `p` two-way t(sessions-1), `p_mp` MP-clustered; `share_opp_w` weighted share of all-opposition questions; `mean_y` weighted LS mean.)

Raw weighted means of y_m2 by LS and asker bloc (no controls):

| ls | government | opposition | mixed | outside_support | all | raw_opp_minus_gov |
|---|---|---|---|---|---|---|
| 13 | 0.840 | 0.855 | 0.842 | 0.887 | 0.851 | 0.015 |
| 14 | 0.871 | 0.860 | 0.857 | 0.872 | 0.864 | -0.011 |
| 15 | 0.858 | 0.850 | 0.836 | 0.847 | 0.850 | -0.008 |
| 16 | 0.841 | 0.851 | 0.830 |  | 0.843 | 0.010 |
| 17 | 0.832 | 0.807 | 0.818 |  | 0.820 | -0.025 |
| 18 | 0.799 | 0.746 | 0.785 | 0.769 | 0.771 | -0.053 |


Sensitivity of the LS-level opposition coefficient to the fixed effects:

| LS | ministry_canonical FE (est) | no topic/ministry FE (est) | ministry_canonical FE (p) | no topic/ministry FE (p) |
|---|---|---|---|---|
| LS13 | 0.0214 | 0.0200 | 0.0194 | 0.0100 |
| LS14 | -0.0103 | -0.0098 | 0.2075 | 0.1017 |
| LS15 | -0.0036 | -0.0044 | 0.6311 | 0.4610 |
| LS16 | 0.0098 | 0.0081 | 0.1169 | 0.1055 |
| LS17 | -0.0087 | -0.0126 | 0.0160 | 0.0012 |
| LS18 | -0.0353 | -0.0427 | 0.0004 | 0.0001 |


Pooled model with LS-specific opposition coefficients (one model, session FE, weights):

| group | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|
| LS13 | 0.0146 | 0.0036 | 0.0257 | 0.0102 | 144,349 |
| LS14 | -0.0042 | -0.0155 | 0.0071 | 0.4614 | 144,349 |
| LS15 | 0.0021 | -0.0087 | 0.0129 | 0.6969 | 144,349 |
| LS16 | 0.0065 | -0.0018 | 0.0148 | 0.1215 | 144,349 |
| LS17 | -0.0143 | -0.0204 | -0.0083 | 0.0000 | 144,349 |
| LS18 | -0.0431 | -0.0524 | -0.0339 | 0.0000 | 144,349 |


Joint Wald test that the six LS gaps are equal: chi2(5) = 90.34, p = 5.71e-18 (p = 8.12e-12 with F, t-df).

Trend and pre/post tests:

| spec | term | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|
| opp + opp x (LS-13) trend: gap at LS13 | opp | 0.0152 | 0.0058 | 0.0246 | 0.0019 | 144,349 |
| opp x (LS-13): change in gap per Lok Sabha | opp_trend | -0.0078 | -0.0110 | -0.0047 | 0.0000 | 144,349 |
| pre/post: gap LS13-16 | opp | 0.0053 | -0.0014 | 0.0120 | 0.1196 | 144,349 |
| pre/post: change in gap | opp_post | -0.0302 | -0.0408 | -0.0195 | 0.0000 | 144,349 |
| pre/post | opp + opp_post (gap LS17-18) | -0.0249 | -0.0339 | -0.0159 | 0.0000 | 144,349 |


Reference estimates (replication of the known numbers):

| group | spec | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|
| pooled LS13-18 | H1 (weighted, session FE) | -0.0031 | -0.0098 | 0.0036 | 0.3596 | 144,349 |
| pooled LS13-18 | H1 unweighted | -0.0142 | -0.0214 | -0.0070 | 0.0002 | 144,349 |
| LS17-18 | session FE | -0.0217 | -0.0306 | -0.0128 | 0.0001 | 94,981 |
| LS17-18 | no session FE (H4key-style) | -0.0244 | -0.0337 | -0.0152 | 0.0000 | 94,981 |


## 2. Finer asker roles

Categories (derived from coalitions.csv, see script docstring): gov_main = INC (UPA periods) / BJP (NDA periods); gov_partner = other government-role parties; opp_main = BJP under UPA / INC under NDA; opp_other = all other opposition; outside = outside_support; `*_both` = multi-asker questions mixing sub-categories inside one bloc; mixed = askers from different blocs.

Questions per category (M2 sample, unweighted counts):

| ls | gov_both | gov_main | gov_partner | mixed | opp_both | opp_main | opp_other | outside |
|---|---|---|---|---|---|---|---|---|
| 13 | 169 | 2,841 | 1,514 | 1,342 | 260 | 2,616 | 2,410 | 1,018 |
| 14 | 131 | 2,152 | 737 | 2,056 | 209 | 2,503 | 1,465 | 1,743 |
| 15 | 97 | 3,348 | 531 | 2,838 | 482 | 2,638 | 2,316 | 886 |
| 16 | 342 | 5,290 | 879 | 2,173 | 219 | 882 | 3,281 | 0 |
| 17 | 1,026 | 25,512 | 2,308 | 5,453 | 1,146 | 7,041 | 17,821 | 0 |
| 18 | 793 | 10,511 | 3,428 | 1,607 | 664 | 7,506 | 10,073 | 92 |


Most frequent coalition-partner askers (single and multi-asker questions in the M2 sample, by asker_party = first asker of single-asker rows; top 6 per LS):

| ls | party | n |
|---|---|---|
| 13 | SS | 504 |
| 13 | JD(U) | 395 |
| 13 | BJD | 188 |
| 13 | PMK | 130 |
| 13 | DMK | 85 |
| 13 | ABLTC | 76 |
| 14 | RJD | 256 |
| 14 | DMK | 139 |
| 14 | NCP | 106 |
| 14 | PMK | 99 |
| 14 | JMM | 93 |
| 14 | TRS | 31 |
| 15 | DMK | 235 |
| 15 | NCP | 157 |
| 15 | KC(M) | 53 |
| 15 | AITC | 33 |
| 15 | BPF | 16 |
| 15 | IUML | 14 |
| 16 | SS | 432 |
| 16 | TDP | 268 |
| 16 | SDF | 39 |
| 16 | LJP | 37 |
| 16 | SAD | 37 |
| 16 | SWP | 25 |
| 17 | JD(U) | 1,090 |
| 17 | SS | 516 |
| 17 | AIADMK | 150 |
| 17 | LJP | 146 |
| 17 | AJSU | 115 |
| 17 | RLP | 63 |
| 18 | TDP | 1,683 |
| 18 | JD(U) | 768 |
| 18 | JSP | 230 |
| 18 | LJP(RV) | 195 |
| 18 | SS | 152 |
| 18 | RLD | 106 |


Coefficients relative to main-ruling-party askers (gov_main), by Lok Sabha (full spec, weights, within-LS session FE):

| term | LS13 | LS14 | LS15 | LS16 | LS17 | LS18 |
|---|---|---|---|---|---|---|
| c_gov_partner | -0.0047 [-0.0280, +0.0186] (p=0.67) | +0.0074 [-0.0086, +0.0235] (p=0.33) | +0.0009 [-0.0151, +0.0170] (p=0.9) | -0.0065 [-0.0244, +0.0115] (p=0.46) | +0.0028 [-0.0139, +0.0195] (p=0.72) | -0.0215 [-0.0416, -0.0013] (p=0.04) |
| c_gov_both | -0.0173 [-0.0739, +0.0392] (p=0.52) | -0.0277 [-0.0798, +0.0245] (p=0.27) | +0.0097 [-0.0387, +0.0580] (p=0.67) | +0.0166 [-0.0082, +0.0415] (p=0.17) | -0.0028 [-0.0171, +0.0116] (p=0.69) | +0.0057 [-0.0255, +0.0370] (p=0.67) |
| c_opp_main | +0.0225 [+0.0015, +0.0435] (p=0.038) | -0.0074 [-0.0233, +0.0084] (p=0.33) | -0.0095 [-0.0256, +0.0066] (p=0.23) | +0.0131 [-0.0075, +0.0337] (p=0.2) | -0.0125 [-0.0227, -0.0024] (p=0.019) | -0.0410 [-0.0528, -0.0293] (p=0.00014) |
| c_opp_other | +0.0141 [-0.0081, +0.0364] (p=0.19) | -0.0128 [-0.0365, +0.0109] (p=0.26) | +0.0021 [-0.0143, +0.0185] (p=0.79) | +0.0072 [-0.0055, +0.0199] (p=0.24) | -0.0086 [-0.0164, -0.0008] (p=0.033) | -0.0443 [-0.0586, -0.0300] (p=0.00027) |
| c_opp_both | +0.0300 [-0.0111, +0.0711] (p=0.14) | -0.0115 [-0.0640, +0.0409] (p=0.64) | +0.0071 [-0.0210, +0.0353] (p=0.59) | +0.0196 [-0.0121, +0.0513] (p=0.21) | -0.0351 [-0.0601, -0.0101] (p=0.0096) | -0.0454 [-0.0602, -0.0307] (p=0.00028) |
| c_outside | +0.0357 [+0.0083, +0.0630] (p=0.015) | +0.0035 [-0.0154, +0.0225] (p=0.69) | -0.0047 [-0.0237, +0.0143] (p=0.6) | n/a | n/a | -0.0200 [-0.0370, -0.0030] (p=0.028) |
| c_mixed | +0.0164 [-0.0150, +0.0478] (p=0.28) | +0.0052 [-0.0155, +0.0260] (p=0.6) | -0.0075 [-0.0257, +0.0106] (p=0.39) | +0.0094 [-0.0077, +0.0265] (p=0.26) | +0.0009 [-0.0097, +0.0116] (p=0.85) | -0.0075 [-0.0305, +0.0155] (p=0.46) |


Contrasts (same models):

| term | group | estimate | ci_lo | ci_hi | p |
|---|---|---|---|---|---|
| opp_other - opp_main | LS13 | -0.0084 | -0.0295 | 0.0128 | 0.4058 |
| opp_main - gov_partner | LS13 | 0.0272 | -0.0003 | 0.0548 | 0.0525 |
| opp_other - opp_main | LS14 | -0.0054 | -0.0268 | 0.0160 | 0.5969 |
| opp_main - gov_partner | LS14 | -0.0149 | -0.0362 | 0.0064 | 0.1546 |
| opp_other - opp_main | LS15 | 0.0116 | -0.0059 | 0.0290 | 0.1753 |
| opp_main - gov_partner | LS15 | -0.0104 | -0.0302 | 0.0093 | 0.2743 |
| opp_other - opp_main | LS16 | -0.0059 | -0.0265 | 0.0147 | 0.5511 |
| opp_main - gov_partner | LS16 | 0.0196 | -0.0067 | 0.0458 | 0.1329 |
| opp_other - opp_main | LS17 | 0.0040 | -0.0062 | 0.0141 | 0.4142 |
| opp_main - gov_partner | LS17 | -0.0153 | -0.0320 | 0.0013 | 0.0680 |
| opp_other - opp_main | LS18 | -0.0033 | -0.0162 | 0.0097 | 0.5565 |
| opp_main - gov_partner | LS18 | -0.0196 | -0.0394 | 0.0002 | 0.0518 |


Pooled / LS17-18 / LS18 (ref gov_main):

| group | term | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted) | c_gov_partner | -0.0065 | -0.0144 | 0.0015 | 0.1098 | 144,349 |
| pooled LS13-18 (weighted) | c_gov_both | -0.0009 | -0.0152 | 0.0134 | 0.9049 | 144,349 |
| pooled LS13-18 (weighted) | c_opp_main | -0.0044 | -0.0123 | 0.0034 | 0.2662 | 144,349 |
| pooled LS13-18 (weighted) | c_opp_other | -0.0046 | -0.0119 | 0.0027 | 0.2105 | 144,349 |
| pooled LS13-18 (weighted) | c_opp_both | -0.0013 | -0.0149 | 0.0124 | 0.8529 | 144,349 |
| pooled LS13-18 (weighted) | c_outside | 0.0100 | -0.0011 | 0.0211 | 0.0772 | 144,349 |
| pooled LS13-18 (weighted) | c_mixed | -0.0012 | -0.0090 | 0.0066 | 0.7681 | 144,349 |
| pooled LS13-18 (weighted) | opp_other - opp_main | -0.0002 | -0.0071 | 0.0067 | 0.9579 | 144,349 |
| LS17-18 | c_gov_partner | -0.0085 | -0.0219 | 0.0048 | 0.1970 | 94,981 |
| LS17-18 | c_gov_both | 0.0035 | -0.0113 | 0.0183 | 0.6272 | 94,981 |
| LS17-18 | c_opp_main | -0.0238 | -0.0335 | -0.0141 | 0.0001 | 94,981 |
| LS17-18 | c_opp_other | -0.0211 | -0.0320 | -0.0102 | 0.0007 | 94,981 |
| LS17-18 | c_opp_both | -0.0405 | -0.0572 | -0.0238 | 0.0001 | 94,981 |
| LS17-18 | c_outside | -0.0050 | -0.0154 | 0.0054 | 0.3269 | 94,981 |
| LS17-18 | c_mixed | -0.0004 | -0.0090 | 0.0082 | 0.9304 | 94,981 |
| LS17-18 | opp_other - opp_main | 0.0027 | -0.0050 | 0.0104 | 0.4725 | 94,981 |
| LS18 | c_gov_partner | -0.0215 | -0.0416 | -0.0013 | 0.0403 | 34,674 |
| LS18 | c_gov_both | 0.0057 | -0.0255 | 0.0370 | 0.6696 | 34,674 |
| LS18 | c_opp_main | -0.0410 | -0.0528 | -0.0293 | 0.0001 | 34,674 |
| LS18 | c_opp_other | -0.0443 | -0.0586 | -0.0300 | 0.0003 | 34,674 |
| LS18 | c_opp_both | -0.0454 | -0.0602 | -0.0307 | 0.0003 | 34,674 |
| LS18 | c_outside | -0.0200 | -0.0370 | -0.0030 | 0.0279 | 34,674 |
| LS18 | c_mixed | -0.0075 | -0.0305 | 0.0155 | 0.4562 | 34,674 |
| LS18 | opp_other - opp_main | -0.0033 | -0.0162 | 0.0097 | 0.5565 | 34,674 |


Single-asker questions only (clean coding; ref gov_main; same controls and FE):

| group | term | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted) | c_gov_partner | -0.0056 | -0.0145 | 0.0032 | 0.2077 | 105,261 |
| pooled LS13-18 (weighted) | c_opp_main | -0.0032 | -0.0116 | 0.0052 | 0.4474 | 105,261 |
| pooled LS13-18 (weighted) | c_opp_other | -0.0035 | -0.0113 | 0.0043 | 0.3736 | 105,261 |
| pooled LS13-18 (weighted) | c_outside | 0.0113 | -0.0007 | 0.0233 | 0.0637 | 105,261 |
| LS17-18 | c_gov_partner | -0.0077 | -0.0221 | 0.0068 | 0.2827 | 70,819 |
| LS17-18 | c_opp_main | -0.0197 | -0.0289 | -0.0106 | 0.0002 | 70,819 |
| LS17-18 | c_opp_other | -0.0195 | -0.0299 | -0.0091 | 0.0009 | 70,819 |
| LS17-18 | c_outside | 0.0014 | -0.0101 | 0.0128 | 0.8072 | 70,819 |
| LS18 | c_gov_partner | -0.0230 | -0.0462 | 0.0002 | 0.0518 | 27,443 |
| LS18 | c_opp_main | -0.0350 | -0.0472 | -0.0228 | 0.0004 | 27,443 |
| LS18 | c_opp_other | -0.0392 | -0.0559 | -0.0224 | 0.0012 | 27,443 |
| LS18 | c_outside | -0.0128 | -0.0318 | 0.0063 | 0.1525 | 27,443 |
| LS17 | c_gov_partner | 0.0050 | -0.0102 | 0.0201 | 0.4936 | 43,376 |
| LS17 | c_opp_main | -0.0097 | -0.0191 | -0.0003 | 0.0436 | 43,376 |
| LS17 | c_opp_other | -0.0079 | -0.0151 | -0.0007 | 0.0335 | 43,376 |
| LS17 | c_outside |  |  |  |  | 43,376 |


LS17 and LS18 single-asker questions: ministry_group FE vs ministry_canonical FE (do partners simply ask different ministries?):

| group | spec | term | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|---|
| LS17 | single-asker, ministry_group FE | c_gov_partner | 0.0050 | -0.0102 | 0.0201 | 0.4936 | 43,376 |
| LS17 | single-asker, ministry_group FE | c_opp_main | -0.0097 | -0.0191 | -0.0003 | 0.0436 | 43,376 |
| LS17 | single-asker, ministry_group FE | c_opp_other | -0.0079 | -0.0151 | -0.0007 | 0.0335 | 43,376 |
| LS17 | single-asker, ministry_canonical FE | c_gov_partner | 0.0063 | -0.0100 | 0.0226 | 0.4176 | 43,376 |
| LS17 | single-asker, ministry_canonical FE | c_opp_main | -0.0057 | -0.0149 | 0.0035 | 0.2028 | 43,376 |
| LS17 | single-asker, ministry_canonical FE | c_opp_other | -0.0063 | -0.0138 | 0.0012 | 0.0913 | 43,376 |
| LS17 | single-asker, ministry_group FE, MP-cluster only | c_gov_partner | 0.0050 | -0.0085 | 0.0184 | 0.4683 | 43,376 |
| LS17 | single-asker, ministry_group FE, MP-cluster only | c_opp_main | -0.0097 | -0.0186 | -0.0008 | 0.0322 | 43,376 |
| LS17 | single-asker, ministry_group FE, MP-cluster only | c_opp_other | -0.0079 | -0.0147 | -0.0011 | 0.0226 | 43,376 |
| LS18 | single-asker, ministry_group FE | c_gov_partner | -0.0230 | -0.0462 | 0.0002 | 0.0518 | 27,443 |
| LS18 | single-asker, ministry_group FE | c_opp_main | -0.0350 | -0.0472 | -0.0228 | 0.0004 | 27,443 |
| LS18 | single-asker, ministry_group FE | c_opp_other | -0.0392 | -0.0559 | -0.0224 | 0.0012 | 27,443 |
| LS18 | single-asker, ministry_group FE | c_outside | -0.0128 | -0.0318 | 0.0063 | 0.1525 | 27,443 |
| LS18 | single-asker, ministry_canonical FE | c_gov_partner | -0.0233 | -0.0462 | -0.0004 | 0.0473 | 27,443 |
| LS18 | single-asker, ministry_canonical FE | c_opp_main | -0.0332 | -0.0453 | -0.0210 | 0.0005 | 27,443 |
| LS18 | single-asker, ministry_canonical FE | c_opp_other | -0.0367 | -0.0532 | -0.0201 | 0.0017 | 27,443 |
| LS18 | single-asker, ministry_canonical FE | c_outside | 0.0014 | -0.0228 | 0.0257 | 0.8914 | 27,443 |
| LS18 | single-asker, ministry_group FE, MP-cluster only | c_gov_partner | -0.0230 | -0.0407 | -0.0053 | 0.0111 | 27,443 |
| LS18 | single-asker, ministry_group FE, MP-cluster only | c_opp_main | -0.0350 | -0.0455 | -0.0245 | 0.0000 | 27,443 |
| LS18 | single-asker, ministry_group FE, MP-cluster only | c_opp_other | -0.0392 | -0.0496 | -0.0287 | 0.0000 | 27,443 |
| LS18 | single-asker, ministry_group FE, MP-cluster only | c_outside | -0.0128 | -0.0542 | 0.0287 | 0.5454 | 27,443 |


Within government single-asker questions, party vs BJP (parties with at least 150 questions; the rest pooled as `other partners`):

| group | term | n_party | mean_y_party | estimate | ci_lo | ci_hi | p | p_mp |
|---|---|---|---|---|---|---|---|---|
| LS18 | pp_TDP | 1,295 | 0.7409 | -0.0421 | -0.0823 | -0.0018 | 0.0430 | 0.0002 |
| LS18 | pp_JD(U) | 646 | 0.7600 | -0.0315 | -0.0671 | 0.0041 | 0.0735 | 0.0330 |
| LS18 | pp_JSP | 211 | 0.7996 | 0.0164 | -0.0865 | 0.1193 | 0.7100 | 0.7232 |
| LS18 | pp_LJP(RV) | 156 | 0.8100 | 0.0048 | -0.0284 | 0.0380 | 0.7349 | 0.6924 |
| LS18 | p_other_partner | 441 | 0.7951 | -0.0043 | -0.0336 | 0.0249 | 0.7293 | 0.7054 |
| LS18 | BJP (reference) | 8,630 | 0.8036 | 0.0000 |  |  |  |  |
| LS17 | pp_JD(U) | 1,020 | 0.8348 | 0.0060 | -0.0177 | 0.0296 | 0.5952 | 0.5933 |
| LS17 | pp_SS | 366 | 0.8305 | 0.0072 | -0.0235 | 0.0379 | 0.6201 | 0.6043 |
| LS17 | pp_AIADMK | 150 | 0.8580 | 0.0236 | -0.0030 | 0.0501 | 0.0774 | 0.0004 |
| LS17 | p_other_partner | 547 | 0.8227 | -0.0034 | -0.0281 | 0.0213 | 0.7714 | 0.7679 |
| LS17 | BJP (reference) | 19,650 | 0.8315 | 0.0000 |  |  |  |  |


Raw mean y_m2 by question category, LS17-18 (unweighted census):

| ls | qcat | mean | size |
|---|---|---|---|
| 17 | gov_both | 0.819 | 1,026 |
| 17 | gov_main | 0.832 | 25,512 |
| 17 | gov_partner | 0.830 | 2,308 |
| 17 | mixed | 0.818 | 5,453 |
| 17 | opp_both | 0.770 | 1,146 |
| 17 | opp_main | 0.801 | 7,041 |
| 17 | opp_other | 0.811 | 17,821 |
| 18 | gov_both | 0.801 | 793 |
| 18 | gov_main | 0.810 | 10,511 |
| 18 | gov_partner | 0.767 | 3,428 |
| 18 | mixed | 0.785 | 1,607 |
| 18 | opp_both | 0.728 | 664 |
| 18 | opp_main | 0.747 | 7,506 |
| 18 | opp_other | 0.746 | 10,073 |
| 18 | outside | 0.769 | 92 |


## 3. Does the gap depend on the question?

### Starred vs unstarred

| group | term | spec | estimate | ci_lo | ci_hi | p | n | n_opp | mean_y |
|---|---|---|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted): unstarred | opp | split sample | -0.0027 | -0.0096 | 0.0041 | 0.4278 | 132,769 | 58,674 | 0.8358 |
| pooled LS13-18 (weighted): starred | opp | split sample | -0.0060 | -0.0186 | 0.0066 | 0.3447 | 11,580 | 4,858 | 0.8773 |
| pooled LS13-18 (weighted): interaction | starred_x_opp | opp x starred (starred minus unstarred gap) | -0.0052 | -0.0156 | 0.0053 | 0.3286 | 144,349 |  |  |
| pooled LS13-18 (weighted) | gap among starred (interaction model) | interaction | -0.0078 | -0.0188 | 0.0033 | 0.1670 | 144,349 |  |  |
| LS17-18: unstarred | opp | split sample | -0.0209 | -0.0296 | -0.0122 | 0.0001 | 87,572 | 40,898 | 0.7988 |
| LS17-18: starred | opp | split sample | -0.0257 | -0.0409 | -0.0105 | 0.0022 | 7,409 | 3,353 | 0.8356 |
| LS17-18: interaction | starred_x_opp | opp x starred (starred minus unstarred gap) | -0.0026 | -0.0131 | 0.0080 | 0.6167 | 94,981 |  |  |
| LS17-18 | gap among starred (interaction model) | interaction | -0.0237 | -0.0377 | -0.0096 | 0.0022 | 94,981 |  |  |


### Security vs other ministries

| group | spec | estimate | ci_lo | ci_hi | p | n | n_opp | mean_y |
|---|---|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted): other ministries | split sample | -0.0028 | -0.0099 | 0.0043 | 0.4304 | 134,966 | 59,133 | 0.8411 |
| pooled LS13-18 (weighted): security | split sample | -0.0070 | -0.0224 | 0.0084 | 0.3712 | 9,383 | 4,399 | 0.8195 |
| pooled LS13-18 (weighted): interaction | opp x security (security minus other gap) | 0.0005 | -0.0132 | 0.0141 | 0.9445 | 144,349 |  |  |
| LS17-18: other ministries | split sample | -0.0205 | -0.0297 | -0.0113 | 0.0002 | 90,336 | 41,786 | 0.8031 |
| LS17-18: security | split sample | -0.0423 | -0.0670 | -0.0175 | 0.0020 | 4,645 | 2,465 | 0.7736 |
| LS17-18: interaction | opp x security (security minus other gap) | -0.0148 | -0.0366 | 0.0069 | 0.1698 | 94,981 |  |  |


By ministry group (opposition coefficient within each group; topic and session FE):

| group | estimate | ci_lo | ci_hi | p | n | n_opp |
|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted): agriculture_rural | -0.0090 | -0.0209 | 0.0029 | 0.1371 | 17,847 | 7,807 |
| pooled LS13-18 (weighted): economy_finance | -0.0056 | -0.0230 | 0.0117 | 0.5196 | 10,020 | 4,717 |
| pooled LS13-18 (weighted): energy_environment_resources | -0.0010 | -0.0117 | 0.0097 | 0.8520 | 20,989 | 9,002 |
| pooled LS13-18 (weighted): governance_law_other | -0.0068 | -0.0282 | 0.0146 | 0.5310 | 8,817 | 3,770 |
| pooled LS13-18 (weighted): industry_commerce_labour | -0.0009 | -0.0144 | 0.0127 | 0.9004 | 20,151 | 8,657 |
| pooled LS13-18 (weighted): infrastructure_transport | -0.0022 | -0.0126 | 0.0081 | 0.6673 | 27,746 | 12,341 |
| pooled LS13-18 (weighted): security | -0.0070 | -0.0224 | 0.0084 | 0.3712 | 9,383 | 4,399 |
| pooled LS13-18 (weighted): social_welfare_health_education | -0.0000 | -0.0123 | 0.0122 | 0.9963 | 29,396 | 12,839 |
| LS17-18: agriculture_rural | -0.0295 | -0.0406 | -0.0183 | 0.0000 | 12,473 | 5,795 |
| LS17-18: economy_finance | -0.0170 | -0.0386 | 0.0047 | 0.1186 | 5,571 | 2,972 |
| LS17-18: energy_environment_resources | -0.0108 | -0.0246 | 0.0031 | 0.1220 | 13,283 | 5,912 |
| LS17-18: governance_law_other | -0.0275 | -0.0440 | -0.0110 | 0.0024 | 5,663 | 2,507 |
| LS17-18: industry_commerce_labour | -0.0264 | -0.0441 | -0.0087 | 0.0056 | 13,610 | 6,052 |
| LS17-18: infrastructure_transport | -0.0095 | -0.0214 | 0.0023 | 0.1074 | 18,078 | 8,574 |
| LS17-18: security | -0.0423 | -0.0670 | -0.0175 | 0.0020 | 4,645 | 2,465 |
| LS17-18: social_welfare_health_education | -0.0222 | -0.0359 | -0.0085 | 0.0030 | 21,658 | 9,974 |


### By request type (part level)

Each row is a part-level regression (score of each question part: FULL=1, PARTIAL=.5, NONE=0) on the question's asker bloc; parts of a question share the question's regressors, so inference is clustered by MP and session; part-level means weight questions by their number of parts, so the 'all parts' row is not identical to the question-level estimate. Controls: n_parts, log_qlen, coasker, ministry_group, topic, session FE.

| request_type | n_parts | mean_score |
|---|---|---|
| ACTION | 253,324 | 0.851 |
| INFORMATION | 282,467 | 0.795 |
| OPINION | 17,551 | 0.766 |
| REASON | 39,208 | 0.748 |


| group | spec | estimate | ci_lo | ci_hi | p | n | n_opp | mean_y |
|---|---|---|---|---|---|---|---|---|
| pooled LS13-18 (weighted): INFORMATION | part-level, by request type | -0.0062 | -0.0143 | 0.0019 | 0.1329 | 282,467 | 123,641 | 0.8201 |
| pooled LS13-18 (weighted): REASON | part-level, by request type | -0.0031 | -0.0157 | 0.0096 | 0.6291 | 39,208 | 20,239 | 0.7830 |
| pooled LS13-18 (weighted): ACTION | part-level, by request type | -0.0004 | -0.0077 | 0.0069 | 0.9130 | 253,324 | 105,548 | 0.8650 |
| pooled LS13-18 (weighted): OPINION | part-level, by request type | -0.0112 | -0.0314 | 0.0090 | 0.2718 | 17,551 | 7,491 | 0.8002 |
| pooled LS13-18 (weighted): all parts | part-level, all parts | -0.0055 | -0.0127 | 0.0018 | 0.1378 | 592,550 |  |  |
| pooled LS13-18 (weighted): opp x REASON | part-level interaction (minus INFORMATION gap) | 0.0011 | -0.0088 | 0.0111 | 0.8244 | 592,550 |  |  |
| pooled LS13-18 (weighted): opp x ACTION | part-level interaction (minus INFORMATION gap) | 0.0039 | -0.0014 | 0.0092 | 0.1466 | 592,550 |  |  |
| pooled LS13-18 (weighted): opp x OPINION | part-level interaction (minus INFORMATION gap) | -0.0073 | -0.0226 | 0.0079 | 0.3404 | 592,550 |  |  |
| pooled LS13-18 (weighted) | Wald, df=3 | 3.6253 |  |  | 0.3049 | 592,550 |  |  |
| LS17-18: INFORMATION | part-level, by request type | -0.0266 | -0.0382 | -0.0151 | 0.0001 | 186,478 | 86,301 | 0.7747 |
| LS17-18: REASON | part-level, by request type | -0.0183 | -0.0364 | -0.0001 | 0.0490 | 25,089 | 14,754 | 0.7164 |
| LS17-18: ACTION | part-level, by request type | -0.0196 | -0.0276 | -0.0115 | 0.0001 | 171,641 | 75,490 | 0.8396 |
| LS17-18: OPINION | part-level, by request type | -0.0462 | -0.0802 | -0.0122 | 0.0102 | 10,565 | 4,851 | 0.7315 |
| LS17-18: all parts | part-level, all parts | -0.0286 | -0.0382 | -0.0190 | 0.0000 | 393,773 |  |  |
| LS17-18: opp x REASON | part-level interaction (minus INFORMATION gap) | 0.0070 | -0.0120 | 0.0259 | 0.4520 | 393,773 |  |  |
| LS17-18: opp x ACTION | part-level interaction (minus INFORMATION gap) | 0.0084 | 0.0010 | 0.0157 | 0.0276 | 393,773 |  |  |
| LS17-18: opp x OPINION | part-level interaction (minus INFORMATION gap) | -0.0222 | -0.0467 | 0.0023 | 0.0738 | 393,773 |  |  |
| LS17-18 | Wald, df=3 | 21.4471 |  |  | 0.0001 | 393,773 |  |  |


## 4. Robustness

### Single-asker questions only

| group | spec | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|---|
| LS13 | single-asker only | 0.0191 | -0.0007 | 0.0390 | 0.0575 | 9,344 |
| LS14 | single-asker only | -0.0142 | -0.0302 | 0.0018 | 0.0775 | 7,796 |
| LS15 | single-asker only | -0.0044 | -0.0215 | 0.0127 | 0.5911 | 8,321 |
| LS16 | single-asker only | 0.0121 | -0.0027 | 0.0269 | 0.1018 | 8,981 |
| LS17 | single-asker only | -0.0089 | -0.0155 | -0.0024 | 0.0116 | 43,376 |
| LS18 | single-asker only | -0.0320 | -0.0449 | -0.0191 | 0.0009 | 27,443 |
| pooled LS13-18 (weighted) | single-asker only | -0.0022 | -0.0093 | 0.0048 | 0.5330 | 105,261 |
| LS17-18 | single-asker only | -0.0184 | -0.0266 | -0.0102 | 0.0001 | 70,819 |


### Strict bloc coding (asker_bloc_strict, opp_strict vs strict government)

| group | estimate | ci_lo | ci_hi | p | n |
|---|---|---|---|---|---|
| LS13 | 0.0208 | 0.0032 | 0.0384 | 0.0241 | 12,170 |
| LS14 | -0.0073 | -0.0230 | 0.0084 | 0.3351 | 10,996 |
| LS15 | -0.0036 | -0.0178 | 0.0107 | 0.5981 | 13,136 |
| LS16 | 0.0095 | -0.0021 | 0.0212 | 0.1003 | 13,066 |
| LS17 | -0.0110 | -0.0178 | -0.0042 | 0.0038 | 60,307 |
| LS18 | -0.0386 | -0.0503 | -0.0268 | 0.0002 | 34,674 |
| pooled LS13-18 (weighted) | -0.0027 | -0.0094 | 0.0041 | 0.4300 | 144,349 |
| LS17-18 | -0.0218 | -0.0307 | -0.0129 | 0.0001 | 94,981 |


### Size of the mean gap relative to the overall decline

| quantity | value |
|---|---|
| mean_LS13 | 0.8506 |
| mean_LS16 | 0.8427 |
| mean_LS17 | 0.8196 |
| mean_LS18 | 0.7705 |
| mean_LS13-16 (weighted) | 0.8513 |
| mean_LS17-18 | 0.8017 |
| decline LS13-16 -> LS17-18 | -0.0496 |
| decline LS13 -> LS18 | -0.0801 |
| decline LS16 -> LS18 | -0.0721 |
| opp share LS17-18 | 0.4659 |
| opp share LS18 | 0.5261 |
| gap LS17-18 (beta) | -0.0217 |
| mean gap LS17-18 = beta x opp share | -0.0101 |
| mean-gap / decline (LS13-16 -> LS17-18) | 0.2041 |
| gap LS13-16 (beta, pre/post model) | 0.0053 |
| opp share LS13-16 (weighted) | 0.3906 |
| change in mean gap = post gap x share_post - pre gap x share_pre | -0.0137 |
|   as share of decline LS13-16 -> LS17-18 | 0.2753 |
| gap LS18 (beta) | -0.0384 |
| mean gap LS18 = beta x opp share | -0.0202 |
| mean-gap LS18 / decline (LS16 -> LS18) | 0.2798 |
| mean-gap LS18 / decline (LS13 -> LS18) | 0.2520 |
| government: LS13-16 -> LS17-18 | -0.0290 |
| opposition: LS13-16 -> LS17-18 | -0.0722 |


Conditional decline (controls, ministry_group + topic FE, no session FE): LS17-18 vs LS13-16 = -0.0291 [-0.0384, -0.0199], p=2.1e-08.

### Gold (human-coded) questions, y_gold

Gold-complete questions (all parts human-coded), weights 1/gold_pi (inclusion probability from the main-sample draw); lean spec (n_parts, log_qlen, coasker, LS FE; no topic or ministry FE because n is small). Compare with the same lean spec on y_m2.

| ls | n | n_opp | mean_gold |
|---|---|---|---|
| 13 | 54 | 29 | 0.845 |
| 14 | 51 | 19 | 0.838 |
| 15 | 57 | 27 | 0.799 |
| 16 | 58 | 20 | 0.804 |
| 17 | 90 | 39 | 0.780 |
| 18 | 90 | 55 | 0.714 |


| group | spec | estimate | ci_lo | ci_hi | p | n | n_opp |
|---|---|---|---|---|---|---|---|
| gold pooled LS13-18 | y_gold, 1/pi weights, LS FE + core ctrl | 0.0318 | -0.0513 | 0.1148 | 0.4486 | 400 | 189 |
| gold pooled LS13-18 | y_gold, unweighted, LS FE + core ctrl | -0.0231 | -0.0794 | 0.0332 | 0.4161 | 400 | 189 |
| gold LS17-18 | y_gold, 1/pi weights, LS FE + core ctrl | -0.0655 | -0.1286 | -0.0023 | 0.0430 | 180 | 94 |
| gold LS17-18 | y_gold, unweighted, LS FE + core ctrl | -0.1025 | -0.1666 | -0.0385 | 0.0033 | 180 | 94 |
| gold LS13-16 | y_gold, 1/pi weights, LS FE + core ctrl | 0.0644 | -0.0471 | 0.1759 | 0.2520 | 220 | 95 |
| gold LS13-16 | y_gold, unweighted, LS FE + core ctrl | 0.0542 | -0.0322 | 0.1405 | 0.2137 | 220 | 95 |
| gold pooled LS13-18 | y_m2 on the gold questions, unweighted, LS FE + core ctrl | -0.0371 | -0.0922 | 0.0181 | 0.1843 | 400 | 189 |
| gold LS17-18 | y_m2 on the gold questions, unweighted, LS FE + core ctrl | -0.1071 | -0.1727 | -0.0414 | 0.0028 | 180 | 94 |
| M2 full pooled LS13-18 | y_m2 full sample, LS FE + core ctrl (lean spec) | -0.0058 | -0.0127 | 0.0010 | 0.0922 | 144,349 | 63,532 |
| M2 full LS17-18 | y_m2 full sample, LS FE + core ctrl (lean spec) | -0.0310 | -0.0408 | -0.0212 | 0.0000 | 94,981 | 44,251 |


Raw weighted opposition-minus-government difference in y_gold (bootstrap over questions, 1000 draws, percentile CI):

| group | n | raw opp - gov (1/pi weighted) | boot_lo | boot_hi |
|---|---|---|---|---|
| gold LS13-16 | 170 | 0.0565 | -0.0413 | 0.1618 |
| gold LS17-18 | 170 | -0.0785 | -0.1512 | -0.0019 |


## Figures

- `rq3_fig1_gap_by_ls.png`: opposition coefficient by Lok Sabha (two-way and MP-only CIs, strict-coding point) and raw means by bloc.
- `rq3_fig2_roles.png`: coalition-partner / main-opposition / other-opposition coefficients by Lok Sabha; LS18 coalition partners by party vs BJP.


## What holds (read with the caveats below)

1. **A gap emerges late and abruptly, not gradually.** Within-LS opposition coefficients: LS13 +0.021, LS14 -0.010, LS15 -0.004, LS16 +0.009 (none of LS14-16 distinguishable from 0), LS17 -0.011 [-0.018, -0.004], LS18 -0.038 [-0.050, -0.027]. Equal-gaps Wald test rejects (chi2(5)=90); pre/post change in the gap -0.030 [-0.041, -0.020]; gap in LS13-16 +0.005 [-0.001, +0.012]. The pooled H1 of -0.003 is the average of a small early positive and a clearly negative late gap.
2. **The "-0.013" quoted for LS17-18 is the LS17 gap from the H4-key interacted model** (opp | single-party majority, n=94,981, [-0.019, -0.007]). The LS17-18 pooled gap with the H1/H3 spec is -0.022 [-0.031, -0.013] (-0.024 without session FE); LS17 alone -0.011, LS18 alone -0.038.
3. **Roles.** No difference between the main opposition party and other opposition in any LS (opp_other minus opp_main between -0.008 and +0.012, all p>0.17; LS18 -0.003). Coalition partners are not answered like the ruling party in LS18 only: -0.022 [-0.042, -0.001] (single-asker -0.023; MP-only CI [-0.041, -0.005]); LS17 and earlier partners are at about 0 (LS17 +0.003). By party in LS18: TDP -0.042, JD(U) -0.032, JSP/LJP(RV)/other about 0. Partners sit about midway between BJP and opposition (-0.041 / -0.044; opp_main minus partner -0.020, p=0.052). Outside-support askers are too few (n=92 in LS18) to say anything.
4. **No clear moderator.** Starred vs unstarred: gaps -0.021 vs -0.026 in LS17-18, interaction -0.003 (p=0.62). Security vs other: -0.042 vs -0.021 in LS17-18, interaction -0.015 [-0.037, +0.007]. Request type (part level, LS17-18): INFORMATION -0.027, REASON -0.018, ACTION -0.020, OPINION -0.046; joint equality test p=0.0001 in LS17-18 (ACTION gap smaller than INFORMATION by 0.008, p=0.03; OPINION larger by 0.022, p=0.07), not rejected pooled (p=0.30). The LS17-18 gap is negative in all 8 ministry groups (5 of 8 individually significant).
5. **Robust to coding and sample.** Single-asker only: LS17 -0.009, LS18 -0.032, LS17-18 -0.018. Strict bloc coding: LS17 -0.011, LS18 -0.039, LS17-18 -0.022. Ministry_canonical FE or no topic/ministry FE: LS18 -0.035 / -0.043, LS17 -0.009 / -0.013. MP-only clustering narrows every CI (LS18 only 7 sessions, t(6)).
6. **Size.** Mean gap = gap x opposition share = -0.010 in LS17-18, about 20% of the -0.050 fall in mean y_m2 from LS13-16 to LS17-18 (the change in the mean gap, -0.014, is about 28%); in LS18 -0.020, about 25-28% of the fall from LS13 or LS16. Most of the decline is common to both blocs (raw: government askers -0.029, opposition -0.072; conditional decline net of controls -0.029).
7. **Gold direction agrees.** LS17-18 gold-complete questions (n=180): -0.066 [-0.129, -0.002] with 1/pi weights, -0.103 unweighted; LS13-16 (n=220) +0.064 [-0.047, +0.176] weighted (n.s.); pooled gold not distinguishable from 0 (n=400). The gold subset's LS17-18 gap is larger than the full-sample lean-spec estimate (-0.031), and y_m2 on the same gold questions gives -0.107, so this is sampling variation of a small subset, not M2 vs human disagreement.

Caveats: exploratory, raw p-values, about 245 coefficients logged in rq3_all_models.csv; per-LS two-way CIs rest on 7-16 session clusters; the late gap coincides with a census (LS17/18) vs a 1/6 sample (LS13-16) design, with weights applied; asker role is coded from party at the question date, and the main-opposition label is the other alliance's anchor party, not the formal Leader of Opposition.

