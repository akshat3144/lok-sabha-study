"""Fill answers that exist only as scanned images (needs_ocr) from Gemini OCR text, via the SAME parser as every
other answer document (clean.split_doc_lines -> textparse.clean_text -> clean.process_answer).

OCR text: data/measures/ocr_text_final.parquet (pipeline/ocr_final.py: gemini-3.8-flash primary, gemini-3.1-flash-lite
check, disagreements adjudicated against the scan; image-only .docx are first turned into PDFs by analysis/docx_to_pdf.py). Only rows whose answer is still missing are touched; needs_ocr stays True as
provenance and answer_ocr = True marks the filled rows (source_a = 'ocr').

    python -I -X utf8 pipeline/apply_ocr.py            # patch data/final/questions_analysis.parquet in place
link_roles.py also calls apply_ocr() so a rebuild of the final table keeps the OCR answers.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textparse as tp  # noqa: E402
from clean import extract_question_body, process_answer, process_question, split_doc_lines  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_PARQUET = os.path.join(ROOT, "data", "measures", "ocr_text_final.parquet")  # pipeline/ocr_final.py
FINAL = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")


_QNO = re.compile(r"QUESTION\s*NO\.?\s*[:.\-]?\s*[*†‡#\s]*(\d{1,5})", re.I)


def qno_in_text(text):
    """question number printed in the document header (first 800 chars), or None"""
    m = _QNO.search((text or "")[:800])
    return int(m.group(1)) if m else None


def parse_ocr(text, question_text):
    """-> (answer dict from process_answer, answer_has_hindi, found_answer_heading, question)
    question = (qtext, qparts, qlabels, has_hindi, from_ocr): the record's question, or the one printed in the
    scanned document when the record has none (same rule as clean.build_record for full reply documents)."""
    nl = [tp.normalize_line(l) for l in (text or "").splitlines()]
    nl = [l for l in nl if l]
    ql, al, found = split_doc_lines(nl)
    atxt, ah = tp.clean_text(lines=al, wrap=False, strip_numbers=True)
    qh, from_ocr = False, False
    if not (question_text or "").strip():
        qtxt, qh = tp.clean_text(lines=ql, wrap=False, strip_numbers=True)
        body, _how = extract_question_body(qtxt)
        if body:
            question_text, from_ocr = tp.reflow(body), True
    qtext, qparts, qlabels = process_question(question_text or "")
    q_scheme = (qparts[-1]["labels"] and tp.label_scheme(qparts[-1]["labels"][0])) if qparts else None
    ans = process_answer(atxt, q_scheme=q_scheme, wrap=True, q_labels=qlabels)
    return ans, bool(ah), found, (qtext, qparts, qlabels, bool(qh), from_ocr)


def apply_ocr(df, ocr_path=OCR_PARQUET, log=print):
    if not os.path.exists(ocr_path):
        log("[ocr] no OCR file; nothing applied")
        return df
    ocr = pd.read_parquet(ocr_path)
    ocr = ocr[(ocr["status"] == "ok") & ocr["text"].fillna("").str.strip().ne("")]
    texts = dict(zip(ocr["qid"], ocr["text"]))
    df = df.copy()
    if "answer_ocr" not in df:
        df["answer_ocr"] = False
    idx = df.index[df["qid"].isin(texts) & df["answer_missing"]]
    n_ok = n_nohead = n_wrong = n_q = 0
    for i in idx:
        r = df.loc[i]
        hdr = qno_in_text(texts[r["qid"]])
        if hdr is not None and hdr != int(r["ques_no"]):
            n_wrong += 1  # the scanned document belongs to a different question (or the OCR misread it)
            log(f"[ocr] {r['qid']}: document says question {hdr}, record says {r['ques_no']} -> not used")
            continue
        ans, ah, found, (qtext, qparts, qlabels, qh, q_from_ocr) = parse_ocr(texts[r["qid"]], r["question_text"])
        if not ans["answer_text"].strip() and ans["annexure_text"].strip():
            # "A statement is laid/annexed" and the whole reply sits in the annexure: that IS the reply
            # (same rule as clean.build_record for fallback bodies)
            a = ans["annexure_text"]
            ans.update(answer_text=a, answer_parts=[{"label": "_all", "labels": [], "grouped": False,
                                                     "inferred": False, "text": a}],
                       answer_labels=[], statement_empty=False, answer_unlabeled=True, annexure_text="",
                       has_annexure=True)
        t = ans["answer_text"]
        if not t.strip():
            continue
        n_nohead += not found
        covered = set(ans["answer_labels"])
        n_eng = len(re.findall(r"[A-Za-z]{3,}", t))
        upd = {
            "answer_text": t, "answer_parts": ans["answer_parts"], "answer_labels": ans["answer_labels"],
            "n_answer_parts": len([p for p in ans["answer_parts"] if p["label"] not in ("_pre", "_all")]),
            "all_parts_covered": (all(l in covered for l in qlabels) if qlabels and ans["answer_labels"] else
                                  (False if qlabels else None)),
            "minister_name": ans["minister_name"], "minister_title": ans["minister_title"],
            "minister_name_raw": ans["minister_name_raw"], "minister_rank": ans["minister_rank"],
            "answer_missing": False, "statement_laid": ans["statement_laid"],
            "statement_empty": ans["statement_empty"], "answer_start_repaired": ans["answer_start_repaired"],
            "answer_label_inferred": any(p["inferred"] for p in ans["answer_parts"]),
            "answer_garbled": bool(t) and n_eng < 4 and ("?" in t or ah),
            "grouped_answer": ans["grouped_answer"], "answer_unlabeled": ans["answer_unlabeled"],
            "boilerplate": tp.boilerplate_flags(t), "has_annexure": ans["has_annexure"],
            "annexure_len": len(ans["annexure_text"]), "annexure_text": ans["annexure_text"],
            "answer_has_hindi": bool(r["answer_has_hindi"]) or ah, "answer_len": len(t),
            "source_a": "ocr", "answer_ocr": True,
            "doc_note": f"{r['doc_note']};ocr" if r["doc_note"] else "ocr",
        }
        if q_from_ocr:
            n_q += 1
            upd.update({"question_text": qtext, "question_parts": qparts, "question_labels": qlabels,
                        "n_question_parts": len([p for p in qparts if p["label"] != "_pre"]),
                        "question_missing": not qtext.strip(), "question_len": len(qtext),
                        "question_has_hindi": qh, "source_q": "ocr"})
        for k, v in upd.items():
            df.at[i, k] = v
        n_ok += 1
    log(f"[ocr] {len(texts)} OCR texts; {len(idx)} rows with a missing answer matched; {n_ok} filled "
        f"({n_q} questions also taken from the scan; {n_nohead} without an ANSWER heading; {n_wrong} rejected: question number mismatch)")
    return df


def main():
    df = pd.read_parquet(FINAL)
    out = apply_ocr(df)
    out.to_parquet(FINAL + ".part", index=False)
    os.replace(FINAL + ".part", FINAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
