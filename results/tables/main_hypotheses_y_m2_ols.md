Outcome y_m2, ols; two-way clustered (MP, session); Holm over H1-H4; pooled all-LS models design-weighted (m2_weight, §4.3a), LS17-vs-LS18 key models unweighted (census)

| hypothesis | spec | term | estimate | se | ci_lo | ci_hi | p | p_holm | p_one_sided | p_holm_one_sided | direction_ok | n | g_mp | g_session | family | weights |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | H1/H3 (session FE) | opp | -0.0031 | 0.0033 | -0.0098 | 0.0036 | 0.3596 | 0.3596 | 0.1798 | 0.5394 | 1 | 144,349 | 1,918 | 78 | ols | m2_weight |
| H2 | H2 key LS17 vs LS18 | coalition_dep | -0.0318 | 0.0062 | -0.0448 | -0.0189 | 0.0001 | 0.0002 | 1.0000 | 1.0000 | 0 | 94,981 | 808 | 21 | ols | none |
| H3 | H1/H3 (session FE) | security_vs_other | -0.0281 | 0.0048 | -0.0376 | -0.0186 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 144,349 | 1,918 | 78 | ols | m2_weight |
| H4 | H4 key LS17 vs LS18 | opp_x_coal | -0.0263 | 0.0052 | -0.0371 | -0.0155 | 0.0001 | 0.0002 | 1.0000 | 1.0000 | 0 | 94,981 | 808 | 21 | ols | none |

Secondary / variant specifications (no multiplicity adjustment)

| term | estimate | se | ci_lo | ci_hi | t | p | hypothesis | spec | n | g_mp | g_session | family | weights | expected_sign | direction_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coalition_dep | -0.0058 | 0.0041 | -0.0140 | 0.0024 | -1.3992 | 0.1658 | H2 | H2 secondary: all LS | 144,349 | 1,918 | 78 | ols | m2_weight | 1 | 0 |
| opp_x_coal | 0.0023 | 0.0055 | -0.0087 | 0.0133 | 0.4240 | 0.6727 | H4 | H4: all LS | 144,349 | 1,918 | 78 | ols | m2_weight | 1 | 1 |
| opp_x_coal | 0.0000 | 0.0054 | -0.0107 | 0.0108 | 0.0082 | 0.9935 | H4 | H4: all LS + session FE | 144,349 | 1,918 | 78 | ols | m2_weight | 1 | 1 |
| opp | -0.0131 | 0.0030 | -0.0193 | -0.0068 | -4.3648 | 0.0003 | H1 | H1 within key contrast (opp gap | majority, LS17) | 94,981 | 808 | 21 | ols | none | -1 | 1 |
