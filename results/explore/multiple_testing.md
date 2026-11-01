# Multiple comparison adjustment for the headline tests (exploratory)

Family: 19 headline tests quoted in FACTS.md (RQ2: 9, RQ3: 8, RQ4: 2). Raw p-values are two sided. Exact p-values are taken from the result files where they exist; otherwise they are recovered from the 95% CI by the normal approximation (column 'p basis' says which, row by row). Holm controls the family wise error rate; Benjamini and Hochberg (BH q) controls the false discovery rate. Each is applied to the whole family and, separately, within each RQ.

Notes. (1) The three government change jumps and the DEFERRAL and INCOMPLETE rows are separate tests of separate claims, but the jump tests are claims of NO effect, so for them a large adjusted p-value is the expected, consistent outcome (they were never expected to be significant); adjusting them mostly lengthens the family and makes the Holm threshold for the other tests slightly stricter. (2) The PPI++ and controlled LS18 minus LS13 and trend tests are four views of one decline and are strongly dependent; Holm is conservative under dependence. (3) The ministry heterogeneity row is a Cochran Q test computed here from the reported ministry scores and design based standard errors (it ignores any covariance between ministries, which are disjoint sets of parts, so this is reasonable). (4) The jurisdiction row uses rounded CI inputs.

## Whole family (Holm and BH over all tests)

| RQ | test | estimate | 95% CI | raw p | p basis | Holm p | BH q | survives at 0.05 (Holm) | source |
|---|---|---|---|---|---|---|---|---|---|
| RQ2 | PPI++ LS18 minus LS13 | -0.093 | [-0.135, -0.051] | 1.47e-05 | exact: from reported z = -4.334 (normal) | 2.05e-04 | 4.64e-05 | yes | results/explore/rq2_t1_ppi_contrasts.csv |
| RQ2 | PPI++ trend per decade | -0.027 | [-0.044, -0.009] | 0.0025 | recovered from the 95% CI by the normal approximation | 0.0279 | 0.0054 | yes | results/explore/rq2_t1_ppi_slope.csv |
| RQ2 | Controlled LS18 minus LS13 (S5) | -0.051 | [-0.063, -0.038] | 6.84e-16 | recovered from the 95% CI by the normal approximation (the model CI is t with 77 df; the p-value differs negligibly) | 1.03e-14 | 2.60e-15 | yes | results/explore/rq2_t2_reg_ls.csv |
| RQ2 | Controlled trend per decade (S5) | -0.013 | [-0.020, -0.006] | 2.25e-04 | recovered from the 95% CI by the normal approximation (model CI is t with 77 df) | 0.0027 | 5.34e-04 | yes | results/explore/rq2_t2_reg_dec.csv |
| RQ2 | Jump at change of government 2004 | 0.021 | [-0.007, 0.048] | 0.1340 | exact: p(jump) from file | 0.7093 | 0.1698 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | Jump at change of government 2014 | 0.005 | [-0.010, 0.019] | 0.5277 | exact: p(jump) from file | 1.0000 | 0.5898 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | Jump at change of government 2024 | 0.007 | [-0.031, 0.045] | 0.7161 | exact: p(jump) from file | 1.0000 | 0.7161 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | INCOMPLETE rise LS18 minus LS13 (pp of parts) | 8.500 | [7.170, 9.920] | 8.67e-34 | recovered from the 95% CI by the normal approximation (CI is a session cluster bootstrap percentile CI) | 1.39e-32 | 4.12e-33 | yes | results/explore/rq2_t3_contributions.csv |
| RQ2 | DEFERRAL fall LS18 minus LS13 (pp of parts) | -4.660 | [-5.150, -4.250] | 1.38e-91 | recovered from the 95% CI by the normal approximation (CI is a session cluster bootstrap percentile CI) | 2.48e-90 | 1.31e-90 | yes | results/explore/rq2_t3_contributions.csv |
| RQ3 | Opposition minus ruling gap, LS13 | 0.021 | [0.005, 0.037] | 0.0144 | exact: t(sessions-1) p from file | 0.1293 | 0.0248 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS14 | -0.010 | [-0.026, 0.006] | 0.1925 | exact: t(sessions-1) p from file | 0.7700 | 0.2286 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS15 | -0.004 | [-0.018, 0.011] | 0.5988 | exact: t(sessions-1) p from file | 1.0000 | 0.6320 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS16 | 0.009 | [-0.003, 0.021] | 0.1182 | exact: t(sessions-1) p from file | 0.7093 | 0.1604 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS17 | -0.011 | [-0.018, -0.004] | 0.0040 | exact: t(sessions-1) p from file | 0.0401 | 0.0076 | yes | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS18 | -0.038 | [-0.050, -0.027] | 2.06e-04 | exact: t(sessions-1) p from file | 0.0027 | 5.34e-04 | yes | results/explore/rq3_all_models.csv |
| RQ3 | LS18 coalition partner minus BJP gap | -0.021 | [-0.042, -0.001] | 0.0403 | exact: t(sessions-1) p from file | 0.3220 | 0.0628 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Gold LS17+LS18 opposition gap (n = 180) | -0.065 | [-0.129, -0.002] | 0.0430 | exact: p from file | 0.3220 | 0.0628 | NO | results/explore/rq3_all_models.csv |
| RQ4 | Ministry heterogeneity (Cochran Q across 55 ministries, df 54, I2 0.994) | Q = 9207.6 | n/a | <1e-300 | computed here from the table (score, score_se): Q = sum w (x - xbar)^2, chi2(54); no CI | <1e-300 | <1e-300 | yes | results/explore/a1_ministry_table_full.csv |
| RQ4 | JURISDICTION share rise LS13 to LS18 (pp of parts) | 1.120 | [1.010, 1.230] | 3.83e-89 | recovered from the two reported CIs (independent, normal approximation; inputs rounded to 2 decimals) | 6.51e-88 | 2.42e-88 | yes | results/explore/rq4_rq1.md A2 (by LS table) |

## Within each RQ (Holm and BH over the tests of that RQ)

| RQ | test | estimate | 95% CI | raw p | p basis | Holm p | BH q | survives at 0.05 (Holm) | source |
|---|---|---|---|---|---|---|---|---|---|
| RQ2 | PPI++ LS18 minus LS13 | -0.093 | [-0.135, -0.051] | 1.47e-05 | exact: from reported z = -4.334 (normal) | 8.79e-05 | 3.30e-05 | yes | results/explore/rq2_t1_ppi_contrasts.csv |
| RQ2 | PPI++ trend per decade | -0.027 | [-0.044, -0.009] | 0.0025 | recovered from the 95% CI by the normal approximation | 0.0102 | 0.0038 | yes | results/explore/rq2_t1_ppi_slope.csv |
| RQ2 | Controlled LS18 minus LS13 (S5) | -0.051 | [-0.063, -0.038] | 6.84e-16 | recovered from the 95% CI by the normal approximation (the model CI is t with 77 df; the p-value differs negligibly) | 4.79e-15 | 2.05e-15 | yes | results/explore/rq2_t2_reg_ls.csv |
| RQ2 | Controlled trend per decade (S5) | -0.013 | [-0.020, -0.006] | 2.25e-04 | recovered from the 95% CI by the normal approximation (model CI is t with 77 df) | 0.0011 | 4.04e-04 | yes | results/explore/rq2_t2_reg_dec.csv |
| RQ2 | Jump at change of government 2004 | 0.021 | [-0.007, 0.048] | 0.1340 | exact: p(jump) from file | 0.4021 | 0.1723 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | Jump at change of government 2014 | 0.005 | [-0.010, 0.019] | 0.5277 | exact: p(jump) from file | 1.0000 | 0.5937 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | Jump at change of government 2024 | 0.007 | [-0.031, 0.045] | 0.7161 | exact: p(jump) from file | 1.0000 | 0.7161 | NO | results/explore/rq2_t4_segmented.csv (unadjusted series) |
| RQ2 | INCOMPLETE rise LS18 minus LS13 (pp of parts) | 8.500 | [7.170, 9.920] | 8.67e-34 | recovered from the 95% CI by the normal approximation (CI is a session cluster bootstrap percentile CI) | 6.94e-33 | 3.90e-33 | yes | results/explore/rq2_t3_contributions.csv |
| RQ2 | DEFERRAL fall LS18 minus LS13 (pp of parts) | -4.660 | [-5.150, -4.250] | 1.38e-91 | recovered from the 95% CI by the normal approximation (CI is a session cluster bootstrap percentile CI) | 1.24e-90 | 1.24e-90 | yes | results/explore/rq2_t3_contributions.csv |
| RQ3 | Opposition minus ruling gap, LS13 | 0.021 | [0.005, 0.037] | 0.0144 | exact: t(sessions-1) p from file | 0.0862 | 0.0383 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS14 | -0.010 | [-0.026, 0.006] | 0.1925 | exact: t(sessions-1) p from file | 0.3850 | 0.2200 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS15 | -0.004 | [-0.018, 0.011] | 0.5988 | exact: t(sessions-1) p from file | 0.5988 | 0.5988 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS16 | 0.009 | [-0.003, 0.021] | 0.1182 | exact: t(sessions-1) p from file | 0.3547 | 0.1576 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS17 | -0.011 | [-0.018, -0.004] | 0.0040 | exact: t(sessions-1) p from file | 0.0281 | 0.0160 | yes | results/explore/rq3_all_models.csv |
| RQ3 | Opposition minus ruling gap, LS18 | -0.038 | [-0.050, -0.027] | 2.06e-04 | exact: t(sessions-1) p from file | 0.0016 | 0.0016 | yes | results/explore/rq3_all_models.csv |
| RQ3 | LS18 coalition partner minus BJP gap | -0.021 | [-0.042, -0.001] | 0.0403 | exact: t(sessions-1) p from file | 0.2013 | 0.0687 | NO | results/explore/rq3_all_models.csv |
| RQ3 | Gold LS17+LS18 opposition gap (n = 180) | -0.065 | [-0.129, -0.002] | 0.0430 | exact: p from file | 0.2013 | 0.0687 | NO | results/explore/rq3_all_models.csv |
| RQ4 | Ministry heterogeneity (Cochran Q across 55 ministries, df 54, I2 0.994) | Q = 9207.6 | n/a | <1e-300 | computed here from the table (score, score_se): Q = sum w (x - xbar)^2, chi2(54); no CI | <1e-300 | <1e-300 | yes | results/explore/a1_ministry_table_full.csv |
| RQ4 | JURISDICTION share rise LS13 to LS18 (pp of parts) | 1.120 | [1.010, 1.230] | 3.83e-89 | recovered from the two reported CIs (independent, normal approximation; inputs rounded to 2 decimals) | 3.83e-89 | 3.83e-89 | yes | results/explore/rq4_rq1.md A2 (by LS table) |

## Summary

- Tests that do NOT survive Holm at 0.05 over the whole family (9 of 19): Jump at change of government 2004; Jump at change of government 2014; Jump at change of government 2024; Opposition minus ruling gap, LS13; Opposition minus ruling gap, LS14; Opposition minus ruling gap, LS15; Opposition minus ruling gap, LS16; LS18 coalition partner minus BJP gap; Gold LS17+LS18 opposition gap (n = 180).
- Tests that do NOT survive Holm within their RQ (9 of 19): Jump at change of government 2004; Jump at change of government 2014; Jump at change of government 2024; Opposition minus ruling gap, LS13; Opposition minus ruling gap, LS14; Opposition minus ruling gap, LS15; Opposition minus ruling gap, LS16; LS18 coalition partner minus BJP gap; Gold LS17+LS18 opposition gap (n = 180).
- Tests with BH q at or above 0.05 over the whole family (8 of 19): Jump at change of government 2004; Jump at change of government 2014; Jump at change of government 2024; Opposition minus ruling gap, LS14; Opposition minus ruling gap, LS15; Opposition minus ruling gap, LS16; LS18 coalition partner minus BJP gap; Gold LS17+LS18 opposition gap (n = 180).

Full table: results/explore/multiple_testing.csv.
