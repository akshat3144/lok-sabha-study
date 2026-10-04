"""apply_ocr: OCR text of a scanned reply goes through the normal parser; wrong documents are rejected."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pandas as pd  # noqa: E402

import apply_ocr as A  # noqa: E402

DOC = """GOVERNMENT OF INDIA
MINISTRY OF CIVIL AVIATION
LOK SABHA
STARRED QUESTION NO. : 367
(To be answered on the 18th July 2019)
DELHI-RAJKOT-DELHI FLIGHT
*367. SHRI MOHANBHAI KALYANJI KUNDARIYA
Will the Minister of CIVIL AVIATION be pleased to state:-
(a) whether the Government has received representations to start additional flights on the Delhi-Rajkot sector;
(b) if so, the details thereof?
ANSWER
MINISTER OF STATE (IC) IN THE MINISTRY OF CIVIL AVIATION (SHRI HARDEEP SINGH PURI)
(a): Yes, Sir. A representation has been received from the Hon'ble Member of Parliament.
(b): The request was denied by AAI as the slot times requested were beyond watch hours of the airport.
"""


def _row(qid="ls17-s1-S-367-20190718", ques_no=367, question=""):
    # list-valued columns exist (object dtype) in the real table, as here
    return {"qid": qid, "ques_no": ques_no, "question_text": question, "question_labels": [], "question_parts": [],
            "answer_parts": [], "answer_labels": [], "boilerplate": [], "answer_text": "",
            "answer_missing": True, "answer_has_hindi": False, "doc_note": "fallback_image_only"}


def _run(tmp_path, rows, text=DOC):
    p = os.path.join(str(tmp_path), "ocr.parquet")
    pd.DataFrame({"qid": [r["qid"] for r in rows], "text": [text] * len(rows), "status": ["ok"] * len(rows)}) \
        .to_parquet(p, index=False)
    return A.apply_ocr(pd.DataFrame(rows), ocr_path=p, log=lambda *a: None)


def test_qno_in_text():
    assert A.qno_in_text(DOC) == 367
    assert A.qno_in_text("UNSTARRED QUESTION NO. †1940\nTO BE ANSWERED") == 1940
    assert A.qno_in_text("no header here") is None


def test_fills_answer_and_missing_question(tmp_path):
    out = _run(tmp_path, [_row()])
    r = out.iloc[0]
    assert bool(r["answer_ocr"]) and not r["answer_missing"]
    assert list(r["answer_labels"]) == ["a", "b"]
    assert r["answer_text"].startswith("(a)") and "watch hours" in r["answer_text"]
    assert r["source_q"] == "ocr" and list(r["question_labels"]) == ["a", "b"]
    assert r["all_parts_covered"] is True and r["minister_title"]


def test_rejects_document_of_another_question(tmp_path):
    out = _run(tmp_path, [_row(ques_no=36)])
    assert not bool(out.iloc[0]["answer_ocr"]) and out.iloc[0]["answer_missing"]


def test_keeps_existing_question(tmp_path):
    q = "(a) whether flights will be added; (b) if so, the details thereof?"
    out = _run(tmp_path, [_row(question=q)])
    assert out.iloc[0]["question_text"] == q  # untouched: only the answer comes from the scan
