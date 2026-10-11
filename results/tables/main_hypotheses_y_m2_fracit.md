Outcome y_m2, fracit; two-way clustered (MP, session); Holm over H1-H4; pooled all-LS models design-weighted (m2_weight, §4.3a), LS17-vs-LS18 key models unweighted (census)

| hypothesis | spec | term | estimate | se | ci_lo | ci_hi | p | p_holm | p_one_sided | p_holm_one_sided | direction_ok | n | g_mp | g_session | family | weights |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | H1/H3 (session FE) | opp (AME) | -0.0026 | 0.0033 | -0.0091 | 0.0040 | 0.4403 | 0.4403 | 0.2201 | 0.6604 | 1 | 144,349 | 1,918 | 78 | fracit | m2_weight |
| H2 | H2 key LS17 vs LS18 | coalition_dep (AME) | -0.0311 | 0.0060 | -0.0435 | -0.0186 | 0.0000 | 0.0001 | 1.0000 | 1.0000 | 0 | 94,981 | 808 | 21 | fracit | none |
| H3 | H1/H3 (session FE); logit scale | security_vs_other | -0.2093 | 0.0331 | -0.2752 | -0.1434 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 144,349 | 1,918 | 78 | fracit | m2_weight |
| H4 | H4 key LS17 vs LS18 | opp_x_coal | -0.1286 | 0.0290 | -0.1892 | -0.0680 | 0.0003 | 0.0005 | 0.9999 | 1.0000 | 0 | 94,981 | 808 | 21 | fracit | none |

Secondary / variant specifications (no multiplicity adjustment)

| term | estimate | se | ci_lo | ci_hi | t | p | hypothesis | spec | n | g_mp | g_session | family | weights | expected_sign | direction_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coalition_dep | -0.0382 | 0.0283 | -0.0946 | 0.0183 | -1.3468 | 0.1820 | H2 | H2 secondary: all LS | 144,349 | 1,918 | 78 | fracit | m2_weight | 1 | 0 |
| opp_x_coal | 0.0165 | 0.0401 | -0.0634 | 0.0963 | 0.4101 | 0.6829 | H4 | H4: all LS | 144,349 | 1,918 | 78 | fracit | m2_weight | 1 | 1 |
| opp_x_coal | -0.0008 | 0.0389 | -0.0783 | 0.0767 | -0.0210 | 0.9833 | H4 | H4: all LS + session FE | 144,349 | 1,918 | 78 | fracit | m2_weight | 1 | 0 |
| opp | -0.0934 | 0.0204 | -0.1359 | -0.0509 | -4.5844 | 0.0002 | H1 | H1 within key contrast (opp gap | majority, LS17) | 94,981 | 808 | 21 | fracit | none | -1 | 1 |
