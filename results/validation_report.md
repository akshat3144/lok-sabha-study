# M3 validation report (stage: main)

Codebook v1.2; kappa threshold for M2 model choice = 0.6; bootstrap B = 1000 (cluster = question).

## 1. Inter-coder reliability (human labels)

Coders: coder1, coder2. Only parts coded by both coders are used (primary set = double-coded estimate questions: 100 questions, 413 parts; single-coded parts are excluded).

| field | n_parts | % agree | kappa | kappa_lw | alpha_nominal | alpha_nom_95CI | alpha_ordinal | alpha_ord_95CI |
|---|---|---|---|---|---|---|---|---|
| label | 413 | 86.925 | 0.708 | 0.728 | 0.707 | [0.624, 0.784] | 0.752 | [0.653, 0.839] |
| evasion_type | 413 | 85.956 | 0.704 |  | 0.704 | [0.625, 0.782] |  | [nan, nan] |
| request_type | 413 | 87.167 | 0.774 |  | 0.772 | [0.695, 0.840] |  | [nan, nan] |
| evasion_type (both non-FULL) | 97 | 84.536 | 0.803 |  | 0.804 | [0.681, 0.907] |  | [nan, nan] |


Label confusion matrix (rows: coder1, columns: coder2):

|  | NONE | PARTIAL | FULL |
|---|---|---|---|
| NONE | 29 | 8 | 9 |
| PARTIAL | 3 | 57 | 26 |
| FULL | 3 | 5 | 273 |


Label reliability by ls (small_n = fewer than 20 parts):

| ls | n_parts | pct_agree | kappa | alpha_nominal | alpha_ordinal | small_n |
|---|---|---|---|---|---|---|
| 13 | 37 | 0.865 | 0.676 | 0.677 | 0.683 | False |
| 14 | 66 | 0.864 | 0.689 | 0.688 | 0.707 | False |
| 15 | 57 | 0.912 | 0.761 | 0.763 | 0.850 | False |
| 16 | 65 | 0.846 | 0.695 | 0.697 | 0.740 | False |
| 17 | 99 | 0.859 | 0.701 | 0.701 | 0.802 | False |
| 18 | 89 | 0.876 | 0.718 | 0.717 | 0.704 | False |


Label reliability by ministry_group (small_n = fewer than 20 parts):

| ministry_group | n_parts | pct_agree | kappa | alpha_nominal | alpha_ordinal | small_n |
|---|---|---|---|---|---|---|
| other | 314 | 0.857 | 0.684 | 0.683 | 0.723 | False |
| security | 99 | 0.909 | 0.787 | 0.788 | 0.843 | False |


Sanity stratum (no answer text), reported separately: 78 parts coded by both, label agreement 97.4% (kappa 0.767; undefined when both coders use one category only).

## 2. Adjudication and gold labels

Parts coded by both coders: 491; single-coded parts: 1223; pending (second coder not imported yet): 0; disagreements on any field: 119; gold labels available: 1714 (by source: adjudicated=119, agreed=372, single=1223); adjudicated: 119; open: 0.
(Run `validate.py adjudication`, fill the adjudicated_* columns, then re-run.)

Sanity stratum (no answer text): 78 parts, 96% gold NONE (expect ~100%).

## 3. M2 (LLM labels) vs gold

| model_id | n_parts | coverage | accuracy | macro_f1 | kappa | kappa_lw | accuracy_weighted | kappa_weighted | evasion_acc_nonfull | cost | kappa_per_cost | passes | ci_lo_clears | kappa_95CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.8-flash | 1636 | 1.000 | 0.868 | 0.787 | 0.711 | 0.756 | 0.876 | 0.709 | 0.661 |  |  | True | True | [0.668, 0.749] |

**Model rule** (kappa >= 0.6, max kappa per cost): models pass the kappa threshold but none has a cost given (use --cost id=value)

Confusion matrix for `gemini-3.8-flash`:

|  | m2:NONE | m2:PARTIAL | m2:FULL |
|---|---|---|---|
| gold:NONE | 134 | 45 | 27 |
| gold:PARTIAL | 22 | 206 | 73 |
| gold:FULL | 2 | 47 | 1080 |


## 4. M1 (embedding similarity) vs gold

Parts: 1606 (FULL 1110, NONE 197).

- AUC FULL vs NONE: 0.642 [0.600, 0.682]
- AUC FULL vs rest: 0.543
- Spearman(M1, LABEL_SCORE): 0.097 [0.048, 0.158]
- Youden-optimal threshold: 0.475 (TPR 0.55, FPR 0.28, J 0.27)
