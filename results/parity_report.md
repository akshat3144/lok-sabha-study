# Source parity: API text vs official Word file (17th Lok Sabha), analysis plan §4.4

Identifiers and statistics only; no answer text. Sample: 300 17th-LS questions (stratified by ministry, `results/parity_sample.csv`) that have both API answer text and a .docx file. The file goes through the pipeline's 18th-LS document path (`parity_check.py`). The question text, question parts, annexure text and flags are the production ones, so only the **answer text source** differs between versions. Reproduce with `analysis/parity_check.py` (extract), `analysis/m1_embed.py --stage score` (M1), `analysis/parity_m1_tost.py` (TOST) and `analysis/parity_m2.py` (M2). Raw numbers are in `results/parity_pairs.csv`, `parity_m1.json` and `parity_m2.json`.

## Criteria and results

The 300 files contain **26 image-only .docx files** (no text layer; 8.7%). No text can be extracted from them, so they could only be handled by OCR, which this check does not cover. Each criterion is therefore reported on the 274 files with extractable text, and on all 300 where that makes sense.

| Criterion | Threshold | Result | Verdict |
|---|---|---|---|
| Part count matches (answer-part structure) | ≥ 95% of questions | **98.9%** (271/274) on files with text. **90.3%** (271/300) on all 300 | **PASS** on 274. **FAIL** on all 300, because of the 26 empty files |
| Median text similarity (API vs file answer) | ≥ 0.95 | **1.000** on 274 (mean 0.990, p5 0.957, 96.0% of questions ≥ 0.95). 1.000 also on all 300 with empty files counted as 0 | **PASS** |
| M1 equivalence (TOST on per-question difference file − API) | ±0.03 | 274 questions: mean difference **−0.0012**, 90% CI [−0.0022, −0.0002], TOST p < 0.001. 95% limits of agreement [−0.021, +0.018] | **PASS** |
| M2 label κ, API vs file | ≥ 0.708 (human inter-coder κ) | **0.880** (95% CI 0.843–0.915) on 274, 1,110 parts. 0.736 (CI 0.675–0.798) on all 300, 1,215 parts | **PASS** on both (on all 300 the CI lower bound is below 0.708) |

Supplementary M2 results, not pre-specified criteria:

| Quantity | 274 files with text | All 300 |
|---|---|---|
| Part labels matched (API vs file) | 1,110 of 1,110 | 1,215 of 1,215 |
| Label agreement | 94.95% | 87.8% |
| Evasion-type κ (human 0.704) | 0.896 (CI 0.865–0.924). Both non-FULL: κ 0.930, n = 275 | 0.748 (CI 0.689–0.808) |
| Request-type κ | 0.920 | 0.910 |
| Mean question score, FULL=1, PARTIAL=.5, NONE=0 (API / file) | 0.823 / 0.821 | 0.821 / 0.751 |
| **Mean question-score difference, file − API, 95% CI** | **−0.0015** [−0.0092, +0.0062] | −0.0695 [−0.0969, −0.0421] |
| Question-level correlation of scores | 0.963 | 0.667 |

## M1 detail

Both versions were scored with the fine-tuned model (`multi-qa-mpnet-base-cos-v1`, max length 512, CPU). Each version was scored separately, in scratch folders, so the production caches and outputs were not touched.

| Level | n | Mean difference (file − API) | 90% CI (TOST) | 95% limits of agreement | Lin's CCC |
|---|---|---|---|---|---|
| Question mean over parts (`m1_part_mean`, the analysis measure) | 274 | −0.0012 | [−0.0022, −0.0002] | [−0.021, 0.018] | 0.997 |
| Whole question vs whole answer | 274 | −0.0015 | [−0.0028, −0.0002] | [−0.027, 0.024] | 0.993 |
| Untuned model, question mean | 274 | −0.0006 | [−0.0015, 0.0003] | [−0.018, 0.017] | 0.998 |
| Part level (question-cluster bootstrap CI [−0.0025, −0.0004]) | 1,105 | −0.0013 | [−0.0021, −0.0006] | [−0.031, 0.029] | 0.998 |

- **Equivalence:** every 90% CI lies far inside ±0.03.
- **Direction of the difference:** the paired t-test (p = 0.047) and Wilcoxon test (p = 0.012) show a small negative shift, about 0.001 cosine, i.e. 4% of the margin. It is statistically detectable but practically negligible.
- **Empty files:** M1 is undefined for the 26 empty files, so they drop out. The results are identical on 274 and on "all 300".

## Why part counts differ

**Question parts are identical in both versions.** The part rows are keyed to the question's own labels, so all 1,215 part_ids match (100%), and the part-level M1 and M2 comparisons rest on these. The criterion concerns the *answer-part* structure, i.e. how the answer is split and grouped by label.

**26 questions, no answer text in the file.**
- All 26 are .docx files with no extractable text. The pipeline classifies them `image_only`, and the file side yields 0 parts.
- They are the only cause of the drop from 98.9% to 90.3%.
- In the corpus as a whole, image-only answers are rare in LS18 (36 OCR'd answers of 34,720, 0.1%) and LS17 (47 of 60,548).
- The 8.7% in this sample therefore looks like a property of the LS17 files that also have API text, not of LS18. This is an inference from counts, not a direct test.

**3 questions with real structural differences** (all extract the same text, similarity 0.95–0.99 here):

| qid | API answer parts | File answer parts | Cause |
|---|---|---|---|
| ls17-s6-U-2107-20210802 | 3 | 4 | The API text reads "(b) & (c)" as a grouped heading. In the file the "&" sits at the end of the (b) paragraph and (c) starts its own line, so (b) and (c) are split into separate parts. |
| ls17-s8-U-1922-20220314 | 2 | 3 | The API text has the grouped heading "(b) to (d)". In the file the connective "to" ends one paragraph and (d) starts a new line. (c) has no heading of its own, so the parts come out as a, b, d. |
| ls17-s11-U-3214-20230320 | 2 | 3 | The API text groups "(a) & (b)" and "(c) & (d)". In the file the "&" is on its own line and (b)/(d) are bare headings, so the file yields the groups a, "b & c", d, with an empty (d). |

- **Common cause:** the .docx puts the connective ("&", "to") and the following part label on different lines or paragraphs from the flowed API text, so the splitter groups the labels differently.
- **Effect on labels:** none for these three. M2 labels on all three questions were identical in both versions.
- **Fix:** this could be fixed by joining a trailing "&"/"to" with the next heading in `textparse`. I did not change pipeline code, per the rules.

## Text similarity

On the 274 files with text:
- **Identical answers:** 86 are byte-identical in both versions, and 196 have whitespace-normalised character similarity ≥ 0.999.
- **Differences below 0.95:** 11 questions fall below 0.95 character similarity. In 6 of them the file text differs in length from the API text by more than 15%: it is longer in 5 (file/API length ratio 1.17–2.24) and shorter in 1 (0.75). The other 5 have nearly equal length (ratio 1.000–1.010) and differ in wording or character order only. I did not read these texts closely enough to say whether the longer file text is extra material or the API text is truncated.
- **Least similar:** ls17-s9-U-3330-20220805 (similarity 0.62, length ratio 2.24) and ls17-s10-U-2101-20221220 (0.70, ratio 1.19).
- I did not read these texts closely to establish the cause.
- **Effect on M2:** only 1 of the 11 has any M2 label disagreement.

## M2 detail

- **Configuration:** `gemini-3.8-flash`, no rationale, default thinking (`thinking="auto"`, which gives level LOW for this model), temperature auto (1.0 for Gemini 3+), seed 0, prompt `m2p-3`. The no-rationale prompt hash `5cabcc2c0bd54722` equals the hash on the production labels (asserted in code).
- **API side:** all 300 questions (1,215 parts) are in `data/measures/m2_labels.parquet`, status ok. The parity API inputs (question text, parts, answer text) are identical to the production inputs, which I checked for all 300.
- **File side:** 300 questions, 1,215 parts, all valid; no invalid rows, so no retry was needed. Output: `data/measures/m2_parity_file.parquet`. The 26 empty files received the "no answer text" prompt and were labelled NONE (75 of the 77 parts the API version labelled FULL became NONE). They are shown separately and drive all of the all-300 gap.
- **Sampling noise vs source effect:** the model runs at temperature 1.0, so labels are not deterministic. No separate API-versus-API repeat was run (not requested; the headline criterion passes without it). The split below gives an indirect noise floor.

| Subset (274 files with text) | Questions | Parts | Label agreement | Label κ |
|---|---|---|---|---|
| Answer string byte-identical in both versions | 86 | 320 | 98.75% | 0.975 |
| Answer string differs (layout, line breaks, bullets or content) | 188 | 790 | 93.4% | 0.830 |

- **Reading:** identical text still gives about 1.3% disagreement (sampling noise). Text that differs in layout or content costs a further 5%. The κ for the differing group, 0.830, is still above 0.708.
- **Annexure:** 34 of the 300 questions have annexure text. It is passed identically in both versions, so this parity check does not cover annexure extraction.
- **Cost:** Gemini cost was **$0.98** (996,695 input and 62,310 output tokens, 300 sync requests, 8 workers, ~18 minutes). It is computed from token counts at the real-time list price in `m2_gemini.py` (2 × the batch price of $0.375 / $1.875 per M tokens), so it is the code's price assumption, not the Cloud billing figure. It is under the $5 cap. The M1 scoring ran on CPU and cost nothing.

## Notes and limitations

1. **Strict reading of the part-count criterion fails** on the full 300 (90.3%). It passes (98.9%) once the 26 files without a text layer are set aside. It is your call which denominator is binding, and the plan does not say. The image-only files measure the OCR route, which is a separate step (§2a) and was not tested here.
2. **If the 26 empty files were not OCR'd** and entered a model as empty, the question-score gap would be −0.07 on the full 300. It comes entirely from these files (−0.79 on them, −0.0015 on the rest). The image-only share in LS18 is about 0.1%, so the practical exposure there is small.
3. **Untested follow-up:** OCR-transcribing the 26 image-only .docx files (production route, `docx_to_pdf.py` plus OCR) and re-running M1 and M2 on them would close the gap in item 1.
4. **Part-count pass rate:** 271 of 274 files have equal answer-part counts; the 3 exceptions are explained above.
5. **Fine-tuning overlap:** the fine-tuned M1 model was trained on API-text pairs from the corpus, and 71 of the 1,215 parity parts (plus 8 validation) belong to its original training split (production `m1_scores.parquet`). Both versions use the same model, so this should not bias the difference, but the API side may be marginally favoured on those parts.
6. **Human κ comparison:** the 0.708 benchmark is a human-versus-human κ on a different sample. The pre-specified comparison to it is made here as specified.
7. **Scope:** the check compares API against file text only. No party or ministry comparisons were made.

## Files added

- `analysis/parity_m2.py`: M2 labelling and analysis.
- `analysis/parity_m1_tost.py`: M1 TOST.
- `data/measures/m2_parity_file.parquet`: labels for the file version.
- `data/measures/m2_parity_cost.json`: cost record.
- `results/parity_m1.json`, `results/parity_m2.json`: statistics.
- `results/parity_report.md`: this report. It replaces the earlier text-only draft from 2026-10-17, which had the same text statistics.

Existing pipeline files and the production M1/M2 outputs were not changed.
