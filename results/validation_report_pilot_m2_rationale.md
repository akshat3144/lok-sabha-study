# M3 validation report (stage: pilot)

Codebook v1.2; kappa threshold for M2 model choice = 0.6; bootstrap B = 1000 (cluster = question).

## 1. Inter-coder reliability (human labels)

Coders: coder1, coder2. Only parts coded by both coders are used (primary set = double-coded estimate questions: 50 questions, 194 parts; single-coded parts are excluded).

| field | n_parts | % agree | kappa | kappa_lw | alpha_nominal | alpha_nom_95CI | alpha_ordinal | alpha_ord_95CI |
|---|---|---|---|---|---|---|---|---|
| label | 194 | 92.268 | 0.842 | 0.876 | 0.843 | [0.764, 0.912] | 0.879 | [0.809, 0.941] |
| evasion_type | 194 | 90.722 | 0.822 |  | 0.822 | [0.738, 0.902] |  | [nan, nan] |
| request_type | 194 | 94.845 | 0.879 |  | 0.879 | [0.805, 0.938] |  | [nan, nan] |
| evasion_type (both non-FULL) | 58 | 91.379 | 0.875 |  | 0.876 | [0.764, 0.967] |  | [nan, nan] |


Label confusion matrix (rows: coder1, columns: coder2):

|  | NONE | PARTIAL | FULL |
|---|---|---|---|
| NONE | 18 | 1 | 0 |
| PARTIAL | 1 | 38 | 3 |
| FULL | 0 | 10 | 123 |


Label reliability by ls (small_n = fewer than 20 parts):

| ls | n_parts | pct_agree | kappa | alpha_nominal | alpha_ordinal | small_n |
|---|---|---|---|---|---|---|
| 13 | 28 | 0.893 | 0.812 | 0.815 | 0.875 | False |
| 14 | 33 | 1.000 | 1.000 | 1.000 | 1.000 | False |
| 15 | 40 | 0.875 | 0.761 | 0.761 | 0.848 | False |
| 16 | 39 | 0.949 | 0.908 | 0.909 | 0.918 | False |
| 17 | 30 | 0.933 | 0.842 | 0.844 | 0.844 | False |
| 18 | 24 | 0.875 | 0.725 | 0.729 | 0.838 | False |


Label reliability by ministry_group (small_n = fewer than 20 parts):

| ministry_group | n_parts | pct_agree | kappa | alpha_nominal | alpha_ordinal | small_n |
|---|---|---|---|---|---|---|
| other | 188 | 0.926 | 0.847 | 0.847 | 0.882 | False |
| security | 6 | 0.833 | 0.700 | 0.718 | 0.867 | True |


## 2. Adjudication and gold labels

Parts coded by both coders: 194; single-coded parts: 0; pending (second coder not imported yet): 0; disagreements on any field: 30; gold labels available: 194 (by source: adjudicated=65, agreed=129); adjudicated: 65; open: 0.
(Run `validate.py adjudication`, fill the adjudicated_* columns, then re-run.)

## 3. M2 (LLM labels) vs gold

| model_id | n_parts | coverage | accuracy | macro_f1 | kappa | kappa_lw | accuracy_weighted | kappa_weighted | evasion_acc_nonfull | cost | kappa_per_cost | passes | ci_lo_clears | kappa_95CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-2.5-flash-lite | 194 | 1.000 | 0.706 | 0.526 | 0.310 | 0.339 | 0.698 | 0.297 | 0.295 | 74.680 | 0.004 | False | False | [0.174, 0.423] |
| gemini-3.1-flash-lite | 194 | 1.000 | 0.784 | 0.603 | 0.474 | 0.541 | 0.787 | 0.484 | 0.361 | 214.740 | 0.002 | False | False | [0.368, 0.588] |
| gemini-3.5-flash-lite | 194 | 1.000 | 0.758 | 0.547 | 0.357 | 0.406 | 0.758 | 0.355 | 0.213 | 296.940 | 0.001 | False | False | [0.233, 0.488] |
| gemini-3.8-flash | 194 | 1.000 | 0.851 | 0.742 | 0.650 | 0.687 | 0.856 | 0.664 | 0.508 | 602.170 | 0.001 | True | False | [0.520, 0.782] |

**Model rule** (kappa >= 0.6, max kappa per cost): gemini-3.8-flash: kappa=0.650 >= 0.6, highest kappa per cost (0.001); lower 95% CI bound 0.520 does NOT clear the threshold

Confusion matrix for `gemini-2.5-flash-lite`:

|  | m2:NONE | m2:PARTIAL | m2:FULL |
|---|---|---|---|
| gold:NONE | 7 | 4 | 5 |
| gold:PARTIAL | 3 | 12 | 30 |
| gold:FULL | 10 | 5 | 118 |


Confusion matrix for `gemini-3.1-flash-lite`:

|  | m2:NONE | m2:PARTIAL | m2:FULL |
|---|---|---|---|
| gold:NONE | 6 | 7 | 3 |
| gold:PARTIAL | 4 | 17 | 24 |
| gold:FULL | 1 | 3 | 129 |


Confusion matrix for `gemini-3.5-flash-lite`:

|  | m2:NONE | m2:PARTIAL | m2:FULL |
|---|---|---|---|
| gold:NONE | 5 | 4 | 7 |
| gold:PARTIAL | 2 | 11 | 32 |
| gold:FULL | 1 | 1 | 131 |


Confusion matrix for `gemini-3.8-flash`:

|  | m2:NONE | m2:PARTIAL | m2:FULL |
|---|---|---|---|
| gold:NONE | 10 | 3 | 3 |
| gold:PARTIAL | 4 | 24 | 17 |
| gold:FULL | 1 | 1 | 131 |


## 4. M1 (embedding similarity) vs gold

Parts: 194 (FULL 133, NONE 16).

- AUC FULL vs NONE: 0.667 [0.531, 0.779]
- AUC FULL vs rest: 0.559
- Spearman(M1, LABEL_SCORE): 0.116 [-0.006, 0.237]
- Youden-optimal threshold: 0.430 (TPR 0.62, FPR 0.19, J 0.43)
