# M2 — Gemini labelling of question parts

Files: `m2_prompt.py` (prompt template + `PROMPT_HASH`), `m2_gemini.py` (CLI), `requirements_m2.txt`, `tests/test_m2.py`.
Contract: `codebook.py` (labels, schema, `instructions_text()`, `EVASION_CHOICES`, `ANNEXURE_CHARS`).
All commands: `python -I -X utf8 analysis/m2_gemini.py <cmd> ...` from `lok_sabha/`.

## Design
- Unit of a request = one question with all its parts (qids from `parts.parquet`; full texts from
  `questions_analysis.parquet`). System instruction = codebook (identical for all requests, cache-friendly);
  user turn = full question, parsed part list, full answer (capped 40k chars), annexure (first `ANNEXURE_CHARS`).
- Output: Gemini structured output (`responseMimeType=application/json` + `RESPONSE_SCHEMA`; the schema puts
  `rationale` (<= `RATIONALE_MAX_WORDS` = 15 words, stated in the prompt, truncated defensively when stored) before
  `label`, with `propertyOrdering`; `--strip-property-ordering` drops it if Vertex rejects it). `--no-rationale`
  = cheaper variant without the field (own prompt hash and work directory).
- Sampling: `--temperature` default auto = 0 for Gemini 2.x, 1.0 for Gemini 3+ (see docs findings), fixed `seed`.
  Thinking at the lowest setting the model allows (`--thinking auto`: 2.5 -> `thinkingBudget=0`; 3.x *-lite ->
  `thinkingLevel=MINIMAL`; 3.8-flash -> `LOW`; never both fields). Override: `off|budget:N|level:LOW|MEDIUM|HIGH`.
- Validation: every part label exactly once, enums valid. Failure -> one retry with the error appended
  (pilot: synchronous; batch: invalid rows are retried synchronously, at most `--max-sync-retries`, real-time
  price). Still failing -> `status="invalid"` (label columns empty, error in `rationale`).
- Every label row carries `model_id`, `model_version` (as returned by the API), `codebook_version`, `prompt_hash`,
  `qa_mismatch`, `answer_garbled` (kept in labelling, excluded in analysis), token counts (request-level tokens are
  split evenly over the parts so sums are exact), `attempts`, `source` (batch / sync / sync_retry).
- Prompt reproducibility: `python -I -X utf8 analysis/m2_prompt.py` prints the exact system prompt and an example
  user prompt; any edit to m2_prompt.py or codebook.py changes `PROMPT_HASH` (and the batch work directory).

## Commands
```
gcloud auth application-default login
gcloud config set project MY_PROJECT        # or pass --project

# offline cost estimate (no credentials)
python -I -X utf8 analysis/m2_gemini.py estimate                 # all four models, all rows
python -I -X utf8 analysis/m2_gemini.py estimate --models gemini-3.1-flash-lite --ls 18

# pilot: synchronous, several models; ids file = .txt/.csv/.parquet with qid or part_id
python -I -X utf8 analysis/m2_gemini.py pilot --project MY_PROJECT --qids-file pilot_pairs.csv \
    --models gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-3.8-flash
#   -> data/measures/m2_pilot_<model>.parquet (all parts of those questions; filter to the 50 part_ids)

# full run (resumable; re-run the same command after any interruption)
python -I -X utf8 analysis/m2_gemini.py batch --project MY_PROJECT --location global \
    --gcs-bucket MY_BUCKET --model gemini-3.1-flash-lite --max-usd 150
#   --dry-run (plan+cost only)  --build-only (write JSONL locally)  --no-wait (submit, exit; re-run to collect)
#   -> data/measures/m2_labels.parquet ; work files in data/measures/m2_batch/<model>__<prompthash>_<cfg>/

# optional OCR of needs_ocr PDFs (data/raw/docs_fallback/ls18/<qid>.pdf) -> data/measures/ocr_text.parquet
python -I -X utf8 analysis/m2_gemini.py ocr --project MY_PROJECT --limit 50
```
If `ocr_text.parquet` exists, rows whose `answer_text` is empty get the OCR text in later estimate/pilot/batch runs.

## Batch mechanics and limits
- Vertex limits (Google docs "Batch inference", read 2026-10-17): up to 200,000 requests and 1 GB input file per
  job; a job may queue up to 72 h; most jobs finish within 24 h of starting; batch = 50% of real-time price.
  Defaults: 100,000 requests and 900 MB per chunk (requests average ~7 KB, so the full corpus is ~4 chunks).
- Input JSONL: `{"request": {systemInstruction, contents, generationConfig}}`; output lines echo `request` and carry
  `response` (+`usageMetadata`, `modelVersion`) or `status`. Output order is not guaranteed, so results are matched to
  questions through a SHA-1 of the user prompt stored in `chunk_NNNN.manifest.json`.
- Storage goes through `gcloud storage cp/ls` (no extra Python package); layout
  `gs://BUCKET/lok_sabha_m2/<run>/in/chunk_NNNN.jsonl` and `.../out/chunk_NNNN/`.
- State machine per chunk (`state.json`, atomic writes): planned -> built -> uploaded -> submitted -> done. The job
  name is persisted immediately after creation; a submitted chunk is never re-submitted. A finished chunk lists
  `missing` qids (no usable output line, e.g. job expired/paused); the next run re-plans only those into new chunks.
- Cost guard: the plan's estimated cost (chars/4 heuristic, no caching, no thinking) is checked against `--max-usd`
  before anything is built, uploaded or submitted; exceeding it exits with code 2.
- Prices (`BATCH_PRICES`, source https://ai.google.dev/gemini-api/docs/pricing as of 2026-10-16) are Gemini
  Developer API prices; confirm Vertex prices in the Cloud console. gemini-3.8-flash doubles on 2027-01-01.
- Truncation marker is exactly `[... truncated ...]`, on its own line after the cut text (question, answer, annexure).

## Docs findings (read 2026-10-17; Vertex docs moved to docs.cloud.google.com/gemini-enterprise-agent-platform/...)
- Batch overview (limits, caching, locations, models):
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference
  - 200,000 requests and 1 GB (Cloud Storage input) per job; global endpoint supported for base Gemini models
    (not tuned); 3.8 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite and 2.5 Flash-Lite are all listed as supported.
  - **Explicit context caching is not supported in batch.** Implicit caching is (90% discount on cached tokens; the
    cache discount takes precedence over, and does not stack with, the 50% batch discount).
- Implicit caching minimum prefix (Vertex context-cache overview):
  https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview
  Gemini 2 family 2,048 tokens; Gemini 3 (with exceptions) 4,096; 3.7/3.8 Flash 6,144. Our system prompt is ~730
  tokens, so implicit caching cannot trigger; padding the prefix to 4,096+ tokens would cost about as much as it
  saves (cached input at 10% of the real-time price = 20% of the batch price). Hence no caching code: the system
  prompt was tightened instead (793 -> 731 tokens; the codebook text itself is ~640 of those and is a shared
  contract) and the user template trimmed (full corpus input 573M -> 531M tokens).
- Thinking (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking): MINIMAL is supported (and default)
  on 3.5 and 3.1 Flash-Lite, **not on 3.8 Flash** (LOW/MEDIUM/HIGH, default MEDIUM). 2.5 Flash/Flash-Lite: thinking is
  off with `thinking_budget=0`, though the same page lists a 512 minimum for Flash-Lite (conflict): if 0 is rejected
  use `--thinking budget:512`. thinkingLevel and thinkingBudget together return an error.
- Temperature: Google "strongly recommend[s]" keeping temperature at 1.0 for Gemini 3
  (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide); lower values may loop
  or degrade. Hence auto = 1.0 + fixed seed for 3.x, 0 for 2.5. Compare `--temperature 0` in the pilot if wanted.
- GCS bucket region: the current batch pages state no requirement for Cloud Storage (only BigQuery output must match
  the job region); an older guide said input in us-central1. Not confirmed. Recommended: `--location global` with a
  bucket in the US (us-central1 or the US multi-region); if the job errors, retry `--location us-central1`.

- Tests: `python -I -X utf8 -m pytest analysis/tests/test_m2.py -q` (offline, mock client / fake GCS / fake Vertex).
