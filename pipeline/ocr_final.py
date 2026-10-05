"""Final OCR text per scanned answer = primary transcription (gemini-3.8-flash) + adjudication of disagreements.

Inputs (data/measures/):
  ocr_text__gemini-3.8-flash.parquet   primary transcription
  ocr_text.parquet                     independent check transcription (gemini-3.1-flash-lite)
  ../derived/ocr_adjudication/verdicts/<qid>.json   human-style review against the scan for every document whose two
                                       transcriptions agree below AGREE (word-level sequence similarity)
Rule: agreement >= AGREE -> 3.8; otherwise the reviewer's verdict ("A" -> 3.1, "B"/"both_ok" -> 3.8,
"neither" -> the reviewer's corrected answer text, source "manual"). If the chosen text failed, the other is used.
Output: data/measures/ocr_text_final.parquet  (qid, text, source, agreement, adjudicated, status) read by apply_ocr.py.

    python -I -X utf8 pipeline/ocr_final.py
"""
import difflib
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAS = os.path.join(ROOT, "data", "measures")
PRIMARY = os.path.join(MEAS, "ocr_text__gemini-3.8-flash.parquet")
CHECK = os.path.join(MEAS, "ocr_text.parquet")
VERDICTS = os.path.join(ROOT, "data", "derived", "ocr_adjudication", "verdicts")
OUT = os.path.join(MEAS, "ocr_text_final.parquet")
AGREE = 0.98
# documented overrides of a verdict (qid -> (choice, reason))
OVERRIDES = {
    "ls18-s2-U-2374-20240806": ("primary", "3.8 re-run in 7-page chunks after the single 42-page pass looped (MAX_TOKENS); "
                                           "answer body word-identical to the reviewer-verified 3.1 text, which had "
                                           "replaced the 41-page annexure with an 'omitted for brevity' placeholder"),
}


def words(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def agreement(a, b):
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def build():
    p = pd.read_parquet(PRIMARY).set_index("qid")
    c = pd.read_parquet(CHECK).set_index("qid")
    ok = lambda d, q: q in d.index and d.at[q, "status"] == "ok" and bool(str(d.at[q, "text"]).strip())  # noqa: E731
    rows = []
    for q in sorted(set(p.index) | set(c.index)):
        ag = agreement(p.at[q, "text"], c.at[q, "text"]) if ok(p, q) and ok(c, q) else float("nan")
        vf = os.path.join(VERDICTS, f"{q}.json")
        v = json.load(open(vf, encoding="utf-8")) if os.path.exists(vf) else None
        if v is None and not (ag >= AGREE):
            raise SystemExit(f"{q}: agreement {ag:.3f} < {AGREE} but no adjudication verdict")
        choice = {"A": "check", "B": "primary", "both_ok": "primary", "neither": "manual"}[v["verdict"]] if v else "primary"
        choice = OVERRIDES.get(q, (choice, ""))[0]
        if choice == "manual":
            text, src = "ANSWER\n" + v["fix"].removeprefix("ANSWER").lstrip() if v.get("fix") else "", "manual"
        else:
            d, src = (p, "gemini-3.8-flash") if choice == "primary" else (c, "gemini-3.1-flash-lite")
            if not ok(d, q):
                d, src = (c, "gemini-3.1-flash-lite") if d is p else (p, "gemini-3.8-flash")
            text = d.at[q, "text"] if ok(d, q) else ""
        rows.append({"qid": q, "text": text, "source": src, "agreement": ag, "adjudicated": v is not None, "override": OVERRIDES.get(q, (None, ""))[1],
                     "status": "ok" if text.strip() else "empty"})
    return pd.DataFrame(rows)


def main():
    d = build()
    d.to_parquet(OUT, index=False)
    print(d["source"].value_counts().to_string())
    print(f"{len(d)} documents; adjudicated {int(d.adjudicated.sum())}; median agreement {d.agreement.median():.4f}; "
          f">= {AGREE}: {(d.agreement >= AGREE).sum()} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
