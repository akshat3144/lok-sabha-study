Bias-corrected (PPI++/DSL) estimates; gold universe only; two-way clustered; Holm within method; prediction term design-weighted (m2_weight)

| hypothesis | term | estimate | se | ci_lo | ci_hi | t | p | method | lambda | n | n_gold | spec | family | weights | p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | H1 | -0.0034 | 0.0032 | -0.0098 | 0.0030 | -1.0493 | 0.2973 | naive (on predictions) |  | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 0.2973 |
| H1 | H1 | 0.1652 | 0.1454 | -0.1244 | 0.4548 | 1.1361 | 0.2594 | gold only (HT, lambda=0) | 0.0000 | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 1.0000 |
| H1 | H1 | 0.0063 | 0.0211 | -0.0357 | 0.0483 | 0.2994 | 0.7654 | PPI++ (lambda tuned) | 0.9678 | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 0.9651 |
| H2 | H2 | -0.0305 | 0.0061 | -0.0432 | -0.0178 | -4.9937 | 0.0001 | naive (on predictions) |  | 91,814 | 180 | H2 key (LS17 vs LS18) | ols | m2_weight | 0.0001 |
| H2 | H2 | -0.1343 | 0.1685 | -0.4857 | 0.2172 | -0.7968 | 0.4349 | gold only (HT, lambda=0) | 0.0000 | 91,814 | 180 | H2 key (LS17 vs LS18) | ols | m2_weight | 1.0000 |
| H2 | H2 | -0.0566 | 0.0265 | -0.1120 | -0.0012 | -2.1320 | 0.0456 | PPI++ (lambda tuned) | 0.9302 | 91,814 | 180 | H2 key (LS17 vs LS18) | ols | m2_weight | 0.1824 |
| H3 | H3 | -0.0283 | 0.0046 | -0.0375 | -0.0190 | -6.0923 | 0.0000 | naive (on predictions) |  | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 0.0000 |
| H3 | H3 | 0.0155 | 0.1448 | -0.2728 | 0.3037 | 0.1068 | 0.9152 | gold only (HT, lambda=0) | 0.0000 | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 1.0000 |
| H3 | H3 | -0.0313 | 0.0171 | -0.0655 | 0.0028 | -1.8255 | 0.0718 | PPI++ (lambda tuned) | 0.9588 | 139,654 | 400 | H1/H3 (session FE) | ols | m2_weight | 0.2154 |
| H4 | H4 | -0.0277 | 0.0053 | -0.0388 | -0.0165 | -5.1829 | 0.0000 | naive (on predictions) |  | 91,814 | 180 | H4 key (LS17 vs LS18) | ols | m2_weight | 0.0001 |
| H4 | H4 | 0.2525 | 0.2678 | -0.3061 | 0.8111 | 0.9429 | 0.3570 | gold only (HT, lambda=0) | 0.0000 | 91,814 | 180 | H4 key (LS17 vs LS18) | ols | m2_weight | 1.0000 |
| H4 | H4 | -0.0285 | 0.0398 | -0.1114 | 0.0545 | -0.7156 | 0.4825 | PPI++ (lambda tuned) | 0.9588 | 91,814 | 180 | H4 key (LS17 vs LS18) | ols | m2_weight | 0.9651 |
