# M1 -- fine-tuned bi-encoder answer relevance (+ source-parity check)

Method: Alvarez & Morrier (2025, *Political Analysis*). Fine-tune `sentence-transformers/multi-qa-mpnet-base-cos-v1`
with MultipleNegativesRankingLoss on in-domain (question part, answer part) pairs from a **5 % train / 1 % validation
sample of questions (split by qid)**, then score cosine(question part, answer part) for **all** 1,588,396 parts.
Also scores the whole question vs whole answer. An untuned baseline (same model, no fine-tuning) is scored in the same run.

## Files
| file | role |
|---|---|
| `analysis/m1_embed.py` | train / score / calibrate / estimate / pack (CLI below) |
| `analysis/parity_check.py` | API-text vs Word/PDF-file parity on ~300 17th-LS questions |
| `analysis/requirements_gpu.txt` | pip requirements for the GPU box |
| `analysis/tests/test_m1.py` | 30 tests (no GPU needed): `python -I -X utf8 -m pytest analysis/tests/test_m1.py -q` |

## Design decisions (all overridable by flag)
- **Split**: SHA-1 hash of `seed:qid` -> train < 5 %, val < 6 %, rest `heldout`. Deterministic and *stable when the corpus is
  rebuilt* (existing qids never change split). The `split` column is written to the outputs; report headline results on
  `heldout` (94 % of parts) or all parts, and say so; train/val parts are in-sample.
- **Training pairs only**: drop parts with no answer, cleaned answer < 20 chars, empty question, exact duplicate pairs.
  Scoring covers all parts; parts with no question or no answer text get `m1_cos = NaN`, `m1_valid = False`.
- **No false in-batch negatives**: batches never contain two parts of the same qid or two identical answer texts.
- **Pre-processing** (`clean_answer`): remove minister header lines, "ANSWER", `*****` rules, "STATEMENT REFERRED TO IN REPLY OF
  PARTS ... QUESTION NO. ..." headings, the lay sentence "A statement is laid on the Table of the House" (statement content is
  kept) and a leading "(a) to (c):" label; collapse whitespace. The *deferral* "Information is being collected and will be laid
  on the Table of the House" is **kept** (it is substantive evasion).
- **Truncation**: texts are cut at the model max length (512 tokens when scoring, 256 when training). The % truncated (parts and
  whole texts) is printed and stored in `m1_scores_run.json`; per-row flags `q_truncated` / `a_truncated` are in the scores file.
- **Efficiency**: unique texts only (shared answers encoded once), fp16, longest first (fail fast on OOM), embeddings written to a
  memmap with a checkpoint marker every `--ckpt-batches` batches (default 100 x 512 = 51,200 texts); training checkpoints every
  `--ckpt-every` steps (full optimizer state). **Everything resumes by re-running the same command.**
- Tuned and untuned embeddings are stored fp16 (cosine error ~1e-3); embeddings are deleted after cosines are computed
  (`--keep-embeddings` to keep).

## Run on Jarvis Labs (A100 40GB, PyTorch template)
Copy only these to your own private instance (raw text is copyright of the Lok Sabha Secretariat; do not put it anywhere else):
```bash
# local (Windows, Git Bash) - make the slim question table (qid, question_text, answer_text), then upload
python -I -X utf8 analysis/m1_embed.py --stage pack            # -> data/measures/m1_input_questions.parquet
#   upload: analysis/m1_embed.py analysis/requirements_gpu.txt data/measures/parts.parquet data/measures/m1_input_questions.parquet
# on the box
pip install -r analysis/requirements_gpu.txt                   # keep the template's torch
python analysis/m1_embed.py --stage estimate --questions m1_input_questions.parquet --parts parts.parquet   # ~6 min on CPU: cleans and tokenises everything, prints truncation + cost
python analysis/m1_embed.py --stage all --parts parts.parquet --questions m1_input_questions.parquet \
       --out-dir out --batch-size 512 --train-batch-size 64 2>&1 | tee m1.log
# robustness encoder (same pipeline, separate outputs *__bge-base-en-v1.5*):
python analysis/m1_embed.py --stage all --model BAAI/bge-base-en-v1.5 --parts parts.parquet --questions m1_input_questions.parquet --out-dir out
# download out/m1_scores*.parquet out/m1_question*.parquet out/*_run.json out/m1_work/*/train_report.json  (not the models, not m1_work embeddings)
```
If the pod is paused/killed: run the same command again. It prints how many chunks are left and the training step it resumes from.
Outputs go to `<out-dir>/m1_scores.parquet` and `m1_question.parquet` (copy into `data/measures/`). The fine-tuned model is in
`models/m1/<slug>/` next to the script (needed again for `parity_check.py --stage score`).

**Cost / time** (A100 40GB at $0.89/h), from `--stage estimate` on the real corpus (1,588,396 parts, 393,501 questions;
1.62 M unique question-side and 1.04 M unique answer-side texts after de-duplication; tokenised with a bert-uncased tokenizer, so
approximate): fine-tuning ~6 min (~80 k train-split parts, ~70 k pairs after filters, 1 epoch, batch 64), encoding with tuned + untuned
model ~54 min at an assumed 150 k tokens/s => **~1 h, ~$0.9; budget $1.5-2.0** for model download, I/O and slower long sequences.
Truncation at 512 tokens: 4.5 % of part answers, 0.01 % of part questions, 13 % of whole answers (question-level score).
`--tok-per-s` changes the assumed throughput; the live log shows texts/s, ETA and $ remaining per chunk, so you can stop early
if the real rate is worse (nothing is lost: re-run resumes). `--untuned-only` halves the scoring cost if you only need the baseline.

## Outputs
`m1_scores.parquet`: `part_id, qid, part_label, m1_cos` (fine-tuned), `m1_cos_untuned`, `m1_valid`, `split`, `q_truncated`,
`a_truncated`, `model_id, model_sha` (base model commit), `ft_id` (hash of fine-tuning configuration + training pairs),
`max_seq_len, run_date`.
`m1_question.parquet`: `qid, m1_q_cos` (whole question vs whole answer), `m1_part_mean, m1_part_min, n_parts_scored`, untuned
counterparts, `split`, model columns.
`m1_scores_run.json`: arguments, versions, truncation, counts. `m1_work/<slug>/train_report.json`: pairs used, retrieval
accuracy/MRR on held-out validation pairs **before vs after** tuning (sanity check that the fine-tuning learned something).

## Coverage threshold from human labels
When `data/coding/human_labels.csv` (`part_id,label` with FULL/PARTIAL/NONE; several coders -> majority) exists:
```bash
python -I -X utf8 analysis/m1_embed.py --stage calibrate --labels data/coding/human_labels.csv [--score-col m1_cos_untuned]
```
Picks the threshold maximising Youden's J for FULL vs NONE (PARTIAL excluded from the fit and reported separately), reports AUC,
bootstrap 95 % CIs, mean score and share above threshold by label -> `m1_scores_calibration_<col>.json`, and writes
`m1_coverage.parquet` (`qid, m1_part_coverage` = share of a question's scored parts above the threshold). Library use:
`calibrate_threshold(scores_df, labels_df)`, `part_coverage(scores_df, threshold)`.
Use the *validation-style* caution: the human-labelled parts should be `heldout` (not train/val) parts.

## Parity check (18th-LS files vs API text)
```bash
python -I -X utf8 analysis/parity_check.py --list        # dry run: 300 questions, 59 ministries, writes results/parity_sample.csv
python -I -X utf8 analysis/parity_check.py --stage download   # polite (1 worker, ~1.5 s/file, resumable), ~8 min, data/raw/parity/ls17/<qid>.<ext>
python -I -X utf8 analysis/parity_check.py --stage extract    # same extraction code as the 18th-LS pipeline; writes parity parts
python -I -X utf8 analysis/parity_check.py --stage report     # results/parity_report.md (text + part-count parity)
# on the GPU box (needs the fine-tuned model), for the M1 part of the report:
python analysis/parity_check.py --stage score --parity-parts data/measures/parity_parts.parquet --parity-questions data/measures/parity_questions.parquet
python -I -X utf8 analysis/parity_check.py --stage report --threshold <calibrated threshold>
```
The M1 comparison scores both versions in one run with the same fine-tuned model (tests at question level, part level with a
question-cluster bootstrap, Bland-Altman limits, TOST equivalence, Lin's CCC, coverage-classification kappa at the threshold).
The scoring set is tiny (~600 parts): it also runs on a laptop CPU with the base model if the fine-tuned weights are copied back.

## Tests
`python -I -X utf8 -m pytest analysis/tests/test_m1.py -q` -- cleaning, qid split, training-pair filters, leak-free batching,
resume logic (embedding checkpoints), cosines/aggregation, Youden calibration, parity selection/statistics/extraction round trip.
