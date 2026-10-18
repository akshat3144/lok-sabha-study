"""Public data release: everything the paper promises, and NO question or answer text.

    python -I -X utf8 analysis/make_release.py      -> data/measures/release/

Files (gzip CSV, UTF-8):
  questions.csv.gz      one row per question in the corpus (393,501): identifiers and metadata, no text
  part_labels.csv.gz    Gemini 3.8 Flash label per question part for the labelled sample (FULL/PARTIAL/NONE,
                        evasion type, request type), with model id, codebook version and prompt hash
  m1_question.csv.gz    embedding similarity scores per question (secondary measure)
  ocr_provenance.csv    which answers were transcribed from scans, by which source, with agreement scores
  README.md             column dictionary and provenance
Human gold labels, samples and the codebook are released from data/coding/ directly.
The texts are records of the Lok Sabha Secretariat; rebuild them from the public source with collect.py and pipeline/.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "measures", "release")

Q_COLS = ["qid", "ls", "session", "date", "qtype", "ques_no", "ministry", "ministry_canonical", "ministry_group",
          "n_members", "n_question_parts", "n_answer_parts", "answer_missing", "question_missing", "statement_laid",
          "grouped_answer", "qa_mismatch", "answer_garbled", "has_annexure", "answer_len", "question_len", "source_a",
          "answer_ocr", "govt_period", "majority_type", "n_askers", "n_gov", "n_opp", "n_support", "asker_bloc",
          "asker_bloc_strict", "asker_parties", "pdf_url", "docx_url"]
L_COLS = ["part_id", "qid", "part_label", "label", "evasion_type", "request_type", "status", "source", "model_id",
          "model_version", "codebook_version", "prompt_hash"]
TEXT_COLS = {"question_text", "answer_text", "question_parts", "answer_parts", "annexure_text", "subject", "members",
             "supplementary_json", "text", "rationale", "raw_response", "minister_name", "minister_name_raw"}


def main():
    os.makedirs(OUT, exist_ok=True)
    q = pd.read_parquet(os.path.join(ROOT, "data", "final", "questions_analysis.parquet"), columns=Q_COLS)
    s = pd.read_csv(os.path.join(ROOT, "data", "measures", "m2_sample.csv"), usecols=["qid", "role", "m2_weight"])
    q = q.merge(s.rename(columns={"role": "m2_role"}), on="qid", how="left")
    q["in_m2_sample"] = q["m2_role"].notna()
    q["asker_parties"] = q["asker_parties"].map(lambda v: ";".join(v) if v is not None and not isinstance(v, str) else v)
    lab = pd.read_parquet(os.path.join(ROOT, "data", "measures", "m2_labels.parquet"), columns=L_COLS)
    m1 = pd.read_parquet(os.path.join(ROOT, "data", "measures", "m1_question.parquet"),
                         columns=["qid", "m1_q_cos", "m1_part_mean", "m1_part_min", "n_parts_scored", "model_id"])
    ocr = pd.read_parquet(os.path.join(ROOT, "data", "measures", "ocr_text_final.parquet"),
                          columns=["qid", "source", "agreement", "adjudicated", "status"])
    for name, d in (("questions", q), ("part_labels", lab), ("m1_question", m1), ("ocr_provenance", ocr)):
        leak = TEXT_COLS & set(d.columns)
        if leak:
            raise SystemExit(f"{name}: text column(s) {leak} must not be released")
    q.to_csv(os.path.join(OUT, "questions.csv.gz"), index=False, compression="gzip")
    lab.to_csv(os.path.join(OUT, "part_labels.csv.gz"), index=False, compression="gzip")
    m1.to_csv(os.path.join(OUT, "m1_question.csv.gz"), index=False, compression="gzip")
    ocr.to_csv(os.path.join(OUT, "ocr_provenance.csv"), index=False)
    for f in sorted(os.listdir(OUT)):
        print(f"{f:24s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:6.1f} MB")
    print(f"questions {len(q):,}; labelled questions {q['in_m2_sample'].sum():,}; part labels {len(lab):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
