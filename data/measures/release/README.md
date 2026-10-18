# Data release: Answering or Evading? Ministerial Responsiveness in India's Lok Sabha from 1999 to 2026

This folder holds the derived data behind the paper. It contains **no question or answer text**. The texts are records of the Lok Sabha Secretariat; rebuild them from the public source with `collect.py` and `pipeline/`. Built by `analysis/make_release.py`.

| File | Rows | Content |
|---|---|---|
| `questions.csv.gz` | 393,501 | One row per question: identifiers and metadata |
| `part_labels.csv.gz` | 595,010 | Gemini 3.8 Flash label for each question part in the labelled sample |
| `m1_question.csv.gz` | 393,501 | Embedding similarity scores per question (secondary measure) |
| `ocr_provenance.csv` | 140 | Answers transcribed from scanned images, with transcription source and agreement |

Human gold labels, sampling frames and the codebook are in `data/coding/` (`gold_labels.csv`, `main_sample.csv`, `pilot_sample.csv`, `codebook_frozen.json`). The model prompt is in `analysis/m2_prompt.py`, and the codebook text is in `analysis/CODEBOOK.md`.

## Key identifiers
- `qid`: `ls{Lok Sabha}-s{session}-{S|U}-{question number}-{yyyymmdd}`, where S means starred and U unstarred.
- `part_id`: `{qid}#{part label}`. The label `_all` marks a question without parsed parts.

## questions.csv.gz
- **Question:** `ls`, `session`, `date`, `qtype`, `ques_no`.
- **Ministry:** the ministry as listed (`ministry`), the harmonised name (`ministry_canonical`) and the group (`ministry_group`).
- **Structure:** `n_members` (askers listed), `n_question_parts`, `n_answer_parts`, `question_len` and `answer_len` (characters).
- **Quality flags:** `answer_missing`, `question_missing`, `statement_laid`, `grouped_answer`, `qa_mismatch`, `answer_garbled`, `has_annexure`, `answer_ocr`, and the answer source `source_a`.
- **Politics:**
  - `govt_period` and `majority_type`;
  - asker counts by role on the question date (`n_askers`, `n_gov`, `n_opp`, `n_support`);
  - `asker_bloc` and `asker_bloc_strict`;
  - `asker_parties`, separated by semicolons.
- **Official files:** `pdf_url` and `docx_url`, the links to the official answer files.
- **Labelled sample:**
  - `in_m2_sample` and `m2_role` (census for LS17 and LS18, a one-in-six stratified random sample for LS13 to LS16, or gold, meaning gold questions labelled only for bias correction).
  - `m2_weight`, the design weight N_h/n_h. It is 0 for the gold extras.
  - Population estimates weight by `m2_weight`.

## part_labels.csv.gz
- **Labels:**
  - `label` is FULL, PARTIAL or NONE, scored 1, 0.5 and 0.
  - `evasion_type` is one of DEFERRAL, NOT_MAINTAINED, JURISDICTION, WITHHELD, DOES_NOT_ARISE, DIFFERENT_QUESTION, INCOMPLETE, ANNEXURE_ONLY, GENERIC or OTHER, or NONE_APPLICABLE when the label is FULL.
  - `request_type` is INFORMATION, REASON, ACTION or OPINION.
- **Provenance:** `source` is batch or a synchronous retry. Also `model_id`, `model_version`, `codebook_version` and `prompt_hash`.
- The labels are validated against human gold labels (κ 0.711). They are slightly lenient (+0.024 on the 0 to 1 scale), so use the bias correction described in the paper for population means.

## License
Derived data in this folder: CC BY 4.0 (see `LICENSE-DATA.md` at the repository root). It contains no question or answer text.
