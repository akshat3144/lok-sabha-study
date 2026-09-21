"""Build the PART-level table every measurement uses (M1 embeddings, M2 Gemini labels, M3 human coding).

    python -I -X utf8 analysis/parts.py          -> data/measures/parts.parquet

One row per (question, question part) with the answer text that addresses that part
(clean.align_parts). Questions without parsed parts become one "_all" part (whole question vs whole answer).
part_id = f"{qid}#{part_label}" is the join key used everywhere downstream.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from clean import align_parts  # noqa: E402

SRC = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")
OUT_DIR = os.path.join(ROOT, "data", "measures")
OUT = os.path.join(OUT_DIR, "parts.parquet")


def _plain(parts):
    """parquet gives numpy arrays of dicts with numpy-array 'labels'; turn into plain python"""
    out = []
    for p in (list(parts) if parts is not None else []):
        p = dict(p)
        p["labels"] = [str(x) for x in (list(p.get("labels")) if p.get("labels") is not None else [])]
        out.append(p)
    return out


import re  # noqa: E402

# an answer that is ONLY the "statement is laid on the Table" sentence carries no content
_LAY_ONLY = re.compile(r"^\W*(\(?[a-z]\)?\s*(to|&|and|-)?\s*\(?[a-z]?\)?\s*:?)?\s*(a\s+)?statement\s+(is\s+)?"
                       r"(laid|placed|enclosed|attached)(\s+(on|at))?(\s+the)?(\s+table\s+of\s+the\s+house)?\W*$", re.I)


def has_content(ans):
    a = (ans or "").strip()
    return bool(a) and not _LAY_ONLY.match(a)


def build(df):
    rows = []
    for r in df.itertuples(index=False):
        qp, ap = _plain(r.question_parts), _plain(r.answer_parts)
        aligned = align_parts(qp, ap) if qp else {}
        if not aligned:  # no parsed parts: whole question vs whole answer
            aligned = {"_all": {"question": r.question_text, "answer": r.answer_text or None,
                                "answer_labels": [], "shared": False}}
        n = len(aligned)
        for label, v in aligned.items():
            rows.append({"part_id": f"{r.qid}#{label}", "qid": r.qid, "part_label": label, "n_parts": n,
                         "question_part": v["question"] or "", "answer_part": v["answer"] or "",
                         "answer_labels": list(v["answer_labels"]), "shared_answer": bool(v["shared"]),
                         "answer_missing_for_part": not has_content(v["answer"])})
    return pd.DataFrame(rows)


def main():
    cols = ["qid", "question_parts", "answer_parts", "question_text", "answer_text"]
    df = pd.read_parquet(SRC, columns=cols)
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = build(df)
    parts.to_parquet(OUT, index=False)
    print(f"{len(df)} questions -> {len(parts)} parts; "
          f"{parts['answer_missing_for_part'].mean():.1%} parts without answer text; "
          f"{(parts['part_label'] == '_all').mean():.1%} '_all' parts; written {OUT}")


if __name__ == "__main__":
    main()
