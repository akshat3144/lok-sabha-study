Bias-corrected (PPI++/DSL) estimates; gold universe only; two-way clustered; Holm within method; prediction term design-weighted (m2_weight)

| hypothesis | term | estimate | se | ci_lo | ci_hi | t | p | method | lambda | n | n_gold | spec | family | weights | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | H1 | -0.0213 | 0.0238 | -0.0687 | 0.0261 | -0.8940 | 0.3741 | naive (on predictions) |  | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 0.3741 |
| H1 | H1 | 6405582825.2498 | 144512188.8480 | 6117822294.0191 | 6693343356.4805 | 44.3256 | 0.0000 | gold only (HT, lambda=0) | 0.0000 | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 0.0000 |
| H1 | H1 | 0.0463 | 0.1101 | -0.1729 | 0.2655 | 0.4203 | 0.6755 | PPI++ (lambda tuned) | 0.9669 | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 1.0000 |
| H2 | H2 | -0.1899 | 0.0370 | -0.2669 | -0.1128 | -5.1374 | 0.0001 | naive (on predictions) |  | 91,814 | 180 | H2 key (LS17 vs LS18) | fracit | m2_weight | 0.0002 |
| H2 | H2 | -2947980631.8463 | 165032264.9568 | -3292231904.1657 | -2603729359.5269 | -17.8631 | 0.0000 | gold only (HT, lambda=0) | 0.0000 | 91,814 | 180 | H2 key (LS17 vs LS18) | fracit | m2_weight | 0.0000 |
| H2 | H2 | -0.3266 | 0.1540 | -0.6479 | -0.0053 | -2.1207 | 0.0466 | PPI++ (lambda tuned) | 0.9360 | 91,814 | 180 | H2 key (LS17 vs LS18) | fracit | m2_weight | 0.1866 |
| H3 | H3 | -0.2114 | 0.0323 | -0.2757 | -0.1471 | -6.5454 | 0.0000 | naive (on predictions) |  | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 0.0000 |
| H3 | H3 | 454583318.7835 | 144857389.5617 | 166135405.1143 | 743031232.4527 | 3.1381 | 0.0024 | gold only (HT, lambda=0) | 0.0000 | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 0.0024 |
| H3 | H3 | -0.2294 | 0.1657 | -0.5593 | 0.1004 | -1.3850 | 0.1700 | PPI++ (lambda tuned) | 0.9625 | 139,654 | 400 | H1/H3 (session FE) | fracit | m2_weight | 0.5101 |
| H4 | H4 | -0.1381 | 0.0298 | -0.2002 | -0.0759 | -4.6352 | 0.0002 | naive (on predictions) |  | 91,814 | 180 | H4 key (LS17 vs LS18) | fracit | m2_weight | 0.0003 |
| H4 | H4 | 6049496844.9371 | 260958286.3509 | 5505147398.3480 | 6593846291.5263 | 23.1819 | 0.0000 | gold only (HT, lambda=0) | 0.0000 | 91,814 | 180 | H4 key (LS17 vs LS18) | fracit | m2_weight | 0.0000 |
| H4 | H4 | -0.1343 | 0.2262 | -0.6061 | 0.3375 | -0.5939 | 0.5592 | PPI++ (lambda tuned) | 0.9638 | 91,814 | 180 | H4 key (LS17 vs LS18) | fracit | m2_weight | 1.0000 |
