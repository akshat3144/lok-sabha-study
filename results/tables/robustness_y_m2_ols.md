### Robustness (primary estimates H1-H4 under each variant)

| variant | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| baseline | -0.0031 | -0.0318 | -0.0281 | -0.0263 |
| fractional logit (AME for H1,H2; logit scale H3,H4) | -0.0026 | -0.0311 | -0.2093 | -0.1286 |
| exclude grouped answers | -0.0042 | -0.0130 | -0.0089 | -0.0087 |
| exclude annexure-only (M2 label-based) | -0.0031 | -0.0318 | -0.0281 | -0.0263 |
| exclude all answers with an annexure | 0.0004 | -0.0340 | -0.0250 | -0.0313 |
| exclude 'statement laid' answers | -0.0024 | -0.0302 | -0.0291 | -0.0243 |
| [12] exclude unseen annexures (has_annexure, no annexure text) | -0.0019 | -0.0228 | -0.0272 | -0.0291 |
| starred only | -0.0060 | -0.0357 | -0.0170 | -0.0329 |
| unstarred only | -0.0027 | -0.0311 | -0.0296 | -0.0248 |
| strict party coding (asker_bloc_strict) | -0.0027 | -0.0318 | -0.0281 | -0.0265 |
| drop split-party windows | -0.0030 | -0.0320 | -0.0281 | -0.0258 |
| pure blocs only (gov vs opp askers) | -0.0029 | -0.0319 | -0.0279 | -0.0275 |
| single-asker questions only | -0.0022 | -0.0343 | -0.0242 | -0.0235 |
| no topic FE | -0.0035 | -0.0299 | -0.0313 | -0.0295 |
| full ministry FE (H3 not identified) | -0.0025 | -0.0341 |  | -0.0254 |
| [16] exclude OCR-transcribed answers | -0.0031 | -0.0318 | -0.0281 | -0.0262 |
| + starred, minister rank | -0.0030 | -0.0292 | -0.0263 | -0.0255 |
| outcome: share of FULL parts | -0.0025 | -0.0476 | -0.0368 | -0.0337 |
| outcome: M1 cosine | -0.0024 | 0.0248 | 0.0071 | -0.0110 |
| outcome: M1 part coverage | -0.0030 | 0.0449 | 0.0119 | -0.0101 |
| outcome: second embedding model | -0.0029 | 0.0253 | 0.0072 | -0.0119 |
| [14] M2 unweighted (ignores design weights; H1/H3) | -0.0142 |  | -0.0279 |  |
| [15a] M1 outcome, full corpus (unweighted; H1/H3) | -0.0024 |  | 0.0071 |  |
| [15b] M1 outcome, M2 sample with design weights (H1/H3) | -0.0031 |  | 0.0041 |  |

### Drop-one-ministry summary

| hypothesis | n_drops | min_estimate | min_when | max_estimate | max_when | share_same_sign_as_median | share_p_lt_0.05 |
|---|---|---|---|---|---|---|---|
| H1 | 45 | -0.0040 | drop ministry: Health and Family Welfare | -0.0020 | drop ministry: Agriculture and Farmers Welfare | 1.0000 | 0.0000 |
| H2 | 45 | -0.0335 | drop ministry: Petroleum and Natural Gas | -0.0228 | drop ministry: Railways | 1.0000 | 1.0000 |
| H3 | 45 | -0.0338 | drop ministry: External Affairs | -0.0180 | drop ministry: Home Affairs | 1.0000 | 1.0000 |
| H4 | 45 | -0.0275 | drop ministry: Tourism | -0.0246 | drop ministry: Education | 1.0000 | 1.0000 |

### Placebo contrasts and within-LS trends

| contrast | majority_a | majority_b | same_majority_type | term | estimate | se | ci_lo | ci_hi | p | n | g_session |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LS14 vs LS13 | coalition-dependent | coalition-dependent | 1 | late | 0.0084 | 0.0047 | -0.0012 | 0.0181 | 0.0833 | 23,166 | 27 |
| LS15 vs LS14 | coalition-dependent | coalition-dependent | 1 | late | -0.0042 | 0.0050 | -0.0144 | 0.0060 | 0.4037 | 24,132 | 28 |
| LS17 vs LS16 | single-party majority | single-party majority | 1 | late | -0.0222 | 0.0042 | -0.0308 | -0.0135 | 0.0000 | 73,373 | 30 |
| LS16 vs LS15 | coalition-dependent | single-party majority | 0 | late | 0.0026 | 0.0034 | -0.0043 | 0.0095 | 0.4478 | 26,202 | 30 |
| LS18 vs LS17 | single-party majority | coalition-dependent | 0 | late | -0.0318 | 0.0062 | -0.0448 | -0.0189 | 0.0001 | 94,981 | 21 |
| within-LS17 trend per year | single-party majority | single-party majority | 1 | yrs | -0.0064 | 0.0018 | -0.0104 | -0.0025 | 0.0038 | 60,307 | 14 |
| within-LS18 trend per year | coalition-dependent | coalition-dependent | 1 | yrs | -0.0141 | 0.0043 | -0.0247 | -0.0036 | 0.0170 | 34,674 | 7 |

### Summary over variants (baseline excluded; drop-one-ministry reported separately)

| hypothesis | baseline | baseline_p | n_variants | n_same_sign | min | min_variant | max | max_variant | sign_flips | n_p_lt_05 |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | -0.0031 | 0.3596 | 23 | 22 | -0.0142 | [14] M2 unweighted (ignores design weights; H1/H3) | 0.0004 | exclude all answers with an annexure | exclude all answers with an annexure | 1 |
| H2 | -0.0318 | 0.0001 | 20 | 17 | -0.0476 | outcome: share of FULL parts | 0.0449 | outcome: M1 part coverage | outcome: M1 cosine; outcome: M1 part coverage; outcome: second embedding model | 20 |
| H3 | -0.0281 | 0.0000 | 22 | 17 | -0.2093 | fractional logit (AME for H1,H2; logit scale H3,H4) | 0.0119 | outcome: M1 part coverage | outcome: M1 cosine; outcome: M1 part coverage; outcome: second embedding model; [15a] M1 outcome, full corpus (unweighted; H1/H3); [15b] M1 outcome, M2 sample with design weights (H1/H3) | 20 |
| H4 | -0.0263 | 0.0001 | 20 | 20 | -0.1286 | fractional logit (AME for H1,H2; logit scale H3,H4) | -0.0087 | exclude grouped answers | none | 18 |

### PPI++ variants (item 13; gold sample only)

| variant | hypothesis | estimate | se | ci_lo | ci_hi | p | lambda | n | n_gold | n_flagged_gold_dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| PPI++ baseline (all gold questions) | H1 | 0.0063 | 0.0211 | -0.0357 | 0.0483 | 0.7654 | 0.9678 | 139,654 | 400 | 0 |
| PPI++ baseline (all gold questions) | H2 | -0.0566 | 0.0265 | -0.1120 | -0.0012 | 0.0456 | 0.9302 | 91,814 | 180 | 0 |
| PPI++ baseline (all gold questions) | H3 | -0.0313 | 0.0171 | -0.0655 | 0.0028 | 0.0718 | 0.9588 | 139,654 | 400 | 0 |
| PPI++ baseline (all gold questions) | H4 | -0.0285 | 0.0398 | -0.1114 | 0.0545 | 0.4825 | 0.9588 | 91,814 | 180 | 0 |
| [13] PPI++ excluding gold questions with 'split'/'garbled' notes | H1 | 0.0090 | 0.0199 | -0.0307 | 0.0487 | 0.6518 | 0.9796 | 139,619 | 365 | 35 |
| [13] PPI++ excluding gold questions with 'split'/'garbled' notes | H2 | -0.0479 | 0.0241 | -0.0982 | 0.0025 | 0.0611 | 0.9448 | 91,797 | 163 | 35 |
| [13] PPI++ excluding gold questions with 'split'/'garbled' notes | H3 | -0.0312 | 0.0170 | -0.0651 | 0.0028 | 0.0715 | 0.9580 | 139,619 | 365 | 35 |
| [13] PPI++ excluding gold questions with 'split'/'garbled' notes | H4 | -0.0343 | 0.0434 | -0.1249 | 0.0563 | 0.4385 | 0.9594 | 91,797 | 163 | 35 |

### Plan 4.6 coverage

all items 1-16 present
