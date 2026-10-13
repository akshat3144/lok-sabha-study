### Table 2. Measurement validity

| Panel | Comparison | n parts | Metric | Estimate | 95% CI |
|---|---|---|---|---|---|
| A. Human inter-coder reliability (double-coded estimate parts) | Responsiveness label (FULL/PARTIAL/NONE) | 413 | % agreement | 86.9 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Responsiveness label (FULL/PARTIAL/NONE) | 413 | Cohen's kappa | 0.708 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Responsiveness label (FULL/PARTIAL/NONE) | 413 | Linear-weighted kappa | 0.728 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Responsiveness label (FULL/PARTIAL/NONE) | 413 | Krippendorff alpha (nominal) | 0.707 | [0.624, 0.784] |
| A. Human inter-coder reliability (double-coded estimate parts) | Responsiveness label (FULL/PARTIAL/NONE) | 413 | Krippendorff alpha (ordinal) | 0.752 | [0.653, 0.839] |
| A. Human inter-coder reliability (double-coded estimate parts) | Evasion type | 413 | % agreement | 86.0 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Evasion type | 413 | Cohen's kappa | 0.704 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Evasion type | 413 | Krippendorff alpha (nominal) | 0.704 | [0.625, 0.782] |
| A. Human inter-coder reliability (double-coded estimate parts) | Request type | 413 | % agreement | 87.2 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Request type | 413 | Cohen's kappa | 0.774 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Request type | 413 | Krippendorff alpha (nominal) | 0.772 | [0.695, 0.840] |
| A. Human inter-coder reliability (double-coded estimate parts) | Evasion type (both coders non-FULL) | 97 | Cohen's kappa | 0.803 |  |
| A. Human inter-coder reliability (double-coded estimate parts) | Evasion type (both coders non-FULL) | 97 | Krippendorff alpha (nominal) | 0.804 | [0.681, 0.907] |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Accuracy | 0.868 | [0.849, 0.885] |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Cohen's kappa | 0.711 | [0.668, 0.749] |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Linear-weighted kappa | 0.756 |  |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Macro-F1 | 0.787 |  |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Design-weighted accuracy | 0.876 |  |
| B. M2 (Gemini) vs gold labels | M2 label vs gold label | 1,636 | Design-weighted kappa | 0.709 |  |
| B. M2 (Gemini) vs gold labels | M2 evasion type vs gold (gold non-FULL parts) | 507 | Accuracy | 0.661 |  |
| B. M2 (Gemini) vs gold labels | M2 vs gold, gold = NONE | 206 | Share labelled correctly by M2 | 0.650 |  |
| B. M2 (Gemini) vs gold labels | M2 vs gold, gold = PARTIAL | 301 | Share labelled correctly by M2 | 0.684 |  |
| B. M2 (Gemini) vs gold labels | M2 vs gold, gold = FULL | 1,129 | Share labelled correctly by M2 | 0.957 |  |
| C. M1 (embedding similarity) vs gold labels | M1 cosine, gold FULL vs NONE | 1,307 | AUC | 0.642 | [0.600, 0.682] |
| C. M1 (embedding similarity) vs gold labels | M1 cosine, gold FULL vs rest | 1,606 | AUC | 0.543 |  |
| C. M1 (embedding similarity) vs gold labels | M1 cosine vs gold label score | 1,606 | Spearman rho | 0.097 | [0.048, 0.158] |

_Recomputed with validate.py (bootstrap B=1000 for reliability, 500 for M2/M1; cluster = question). Reliability: parts coded by both coders in the estimation strata. Gold: adjudicated/agreed/single-coder labels, sanity stratum (no answer text) excluded. Evasion-type agreement is not weighted. Class recall uses gold-label rows of the M2 confusion matrix._
