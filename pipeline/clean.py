"""Text-cleaning pipeline for Lok Sabha Question Hour (13th-18th LS).

    python -I pipeline/clean.py                 # all LS found in data/raw
    python -I pipeline/clean.py --ls 17 18      # subset
    python -I pipeline/clean.py --workers 4     # parallel docx extraction (default: cpu//2)

Reads  data/raw/api/ls{N}/s{SS}_p{PPPP}.json.gz  and  data/raw/docs/ls18/*.docx|pdf  (read-only).
Also reads (read-only) data/raw/docs_fallback/ls{N}/<qid>.<docx|pdf|htm>: replacement reply files used ONLY when
the answer would otherwise be missing / an empty "a statement is laid on the Table" stub (source_a becomes
fallback_docx / fallback_pdf / fallback_htm, fallback_file records the path; image-only files set needs_ocr).
data/ref/ministry_map.csv (built by pipeline/ministry_build.py) supplies ministry_canonical / ministry_group;
unmapped labels are reported in clean_report.json ("ministry_unmapped") and make the run exit with status 2.
Writes data/clean/questions.parquet, data/clean/clean_report.json and a docx line cache under
data/clean/cache/ (safe to delete).  Idempotent: every run rebuilds the table from whatever raw files
exist right now, de-duplicated on qid.  *.part files and unreadable/partial files are ignored.

Row schema (one per question): see COLUMNS / SCHEMA below.  Parts are lists of
{label, labels, grouped, text}: `label` is the first label, `labels` every label covered (grouped answers
such as "(a) to (c):" give ["a","b","c"]).  Pseudo-labels "_pre" (text before the first label) and "_all"
(answer with no labels at all) are used so no text is lost; they never count as covered labels.
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import textparse as tp  # noqa: E402
from docx_extract import EXTRACT_VERSION, extract_lines  # noqa: E402
import htm_extract  # noqa: E402

ROOT = os.path.dirname(HERE)
RAW_API = os.path.join(ROOT, "data", "raw", "api")
RAW_DOCS = os.path.join(ROOT, "data", "raw", "docs")
RAW_FALLBACK = os.path.join(ROOT, "data", "raw", "docs_fallback")  # <ls>/<qid>.<docx|pdf|htm>
MINISTRY_MAP_CSV = os.path.join(ROOT, "data", "ref", "ministry_map.csv")
CLEAN_DIR = os.path.join(ROOT, "data", "clean")
CACHE_DIR = os.path.join(CLEAN_DIR, "cache", "doc_lines")
LOK_SABHAS = [13, 14, 15, 16, 17, 18]


def safe(s):  # identical to collect.py
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def doc_key(ls, q):
    return (f"ls{ls}_s{safe(q.get('sessionNo'))}_{safe((q.get('type') or '').strip())}_"
            f"{safe(q.get('quesNo'))}_{safe(q.get('date'))}")


# --------------------------------------------------------------------------------------
# Document (docx / pdf) -> lines, with on-disk cache
# --------------------------------------------------------------------------------------
def pdf_lines(path):
    from pypdf import PdfReader

    r = PdfReader(path)
    lines = []
    for pg in r.pages:
        try:
            t = pg.extract_text() or ""
        except Exception:
            t = ""
        lines.extend(t.split("\n"))
    return lines, {"n_pages": len(r.pages), "n_media": 0}


def load_doc_lines(path):
    """Return (lines, stats, error).  Cached in data/clean/cache keyed by size+mtime."""
    try:
        st = os.stat(path)
    except OSError as e:
        return [], {}, f"stat:{e}"
    base = os.path.splitext(os.path.basename(path))[0]
    ls_dir = os.path.basename(os.path.dirname(path))
    if os.path.basename(os.path.dirname(os.path.dirname(path))) == "docs_fallback":
        ls_dir = "fb_" + ls_dir  # separate cache namespace for the replacement files
    cpath = os.path.join(CACHE_DIR, ls_dir, base + ".json.gz")
    stamp = [st.st_size, int(st.st_mtime), EXTRACT_VERSION]
    if path.lower().endswith((".htm", ".html")):
        stamp.append(htm_extract.EXTRACT_VERSION)
    try:
        with gzip.open(cpath, "rt", encoding="utf8") as f:
            c = json.load(f)
        if c.get("stamp") == stamp:
            return c["lines"], c["stats"], c.get("error")
    except Exception:
        pass
    err = None
    try:
        if path.lower().endswith(".docx"):
            lines, stats = extract_lines(path)
        elif path.lower().endswith((".htm", ".html")):
            lines, stats = htm_extract.htm_lines(path)
        else:
            lines, stats = pdf_lines(path)
    except Exception as e:  # corrupt / partial file
        lines, stats, err = [], {}, f"{type(e).__name__}:{e}"[:200]
        return lines, stats, err  # do not cache failures (file may still be arriving)
    try:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        tmp = cpath + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf8") as f:
            json.dump({"stamp": stamp, "lines": lines, "stats": stats, "error": err}, f)
        os.replace(tmp, cpath)
    except Exception:
        pass
    return lines, stats, err


def _worker(path):
    return path, load_doc_lines(path)


# --------------------------------------------------------------------------------------
# Splitting a full reply document into question block / answer block
# --------------------------------------------------------------------------------------
_ANS_HEAD = re.compile(r"^ANSWERS?\s*[:.]?$|^A\s?N\s?S\s?W\s?E\s?R\s*[:.]?$", re.I)
_ANS_HEAD_INLINE = re.compile(r"^ANSWERS?\s*[:\-]?\s+(?=(?:THE\s+)?(?:PRIME\s+)?MINISTER)", re.I)


def split_doc_lines(lines):
    """Split normalised doc lines at the ANSWER heading. Returns (q_lines, a_lines, found)."""
    for i, l in enumerate(lines):
        if _ANS_HEAD.match(l):
            return lines[:i], lines[i + 1:], True
        if _ANS_HEAD_INLINE.match(l):
            return lines[:i], [_ANS_HEAD_INLINE.sub("", l)] + lines[i + 1:], True
    # no ANSWER heading: look for a minister header line
    for i, l in enumerate(lines):
        if re.match(r"^(THE\s+)?(PRIME\s+)?MINISTER\b", l) and i > 3 and l.isupper():
            return lines[:i], lines[i:], True
    return lines, [], False


def extract_question_body(text, window=1500):
    """Cut the front matter (ministry header, 'QUESTION NO.', title, member) off a docx question block."""
    m = tp._PLEASED.search(text[:window])
    if m:
        return text[m.end():].lstrip(" \n:-–—").strip(), "pleased"
    ls = text.split("\n")
    # first part label line
    for i, l in enumerate(ls[:25]):
        if re.match(r"^\(\s*a\s*\)|^a\)|^\(\s*i\s*\)", l, re.I):
            return "\n".join(ls[i:]).strip(), "label"
    # last line ending with ':' in the first 14 lines (member line)
    k = None
    for i, l in enumerate(ls[:14]):
        if l.endswith(":"):
            k = i
    if k is not None:
        return "\n".join(ls[k + 1:]).strip(), "colon"
    return text.strip(), "none"


# --------------------------------------------------------------------------------------
# Question / answer processing (shared by API and docx paths)
# --------------------------------------------------------------------------------------
def _mk_part(label, labels, text, inferred=False):
    return {"label": label, "labels": labels, "grouped": len(labels) > 1, "text": text, "inferred": inferred}


def process_question(qtext):
    """qtext: cleaned plain text of the question (front matter already removed for docx)."""
    m = tp._PLEASED.search(qtext[:1500] or "")
    if m:  # some API records carry the whole front matter (ministry header, title, member names)
        qtext = qtext[m.end():].lstrip(" \n:-–—").strip()
    pre, parts = tp.split_parts(qtext, "question")
    out_parts = []
    if parts:
        if pre:
            out_parts.append(_mk_part("_pre", [], pre))
        out_parts += [_mk_part(p["label"], p["labels"], p["text"]) for p in parts]
    labels = [l for p in parts for l in p["labels"]]
    return qtext, out_parts, labels


def process_answer(atext, q_scheme=None, wrap=False, q_labels=None):
    """atext: cleaned answer text including the minister header, if present.

    Returns dict with minister info, answer_text, parts, flags, annexure."""
    res = {"minister_title": None, "minister_name": None, "minister_name_raw": None, "minister_rank": None,
           "answer_text": "", "answer_parts": [], "answer_labels": [], "statement_laid": False,
           "grouped_answer": False, "answer_start_repaired": False, "statement_empty": False,
           "has_annexure": False, "annexure_text": "", "answer_unlabeled": False}
    if not atext or not atext.strip():
        return res
    atext, repaired = tp.repair_truncated_start(atext)
    res["answer_start_repaired"] = repaired
    info, rest = tp.parse_minister_header(atext)
    res.update(info)

    def fin(s):
        s = tp.reflow(s) if wrap else s
        return s.strip()

    laid_labels, body = tp.split_statement(rest)
    if laid_labels is not None:
        # statement content follows the "*****" line, so split the annexure off the statement body
        res["statement_laid"] = True
        main, annex, referenced = tp.split_annexure(body)
        pre, parts = tp.split_parts(main, "answer", scheme_hint=q_scheme, allowed=q_labels)
        if parts and laid_labels and not set(laid_labels) & {l for p in parts for l in p["labels"]}:
            parts, pre = [], ""  # nested sub-list inside an unlabeled statement
        res["statement_empty"] = not main.strip()
        if not parts and laid_labels and main.strip():
            # unlabeled statement: it answers every part named in the lay sentence
            parts = [{"label": laid_labels[0], "labels": laid_labels, "grouped": len(laid_labels) > 1,
                      "text": main.strip()}]
            pre = ""
    else:
        main, annex, referenced = tp.split_annexure(rest)
        pre, parts = tp.split_parts(main, "answer", scheme_hint=q_scheme, allowed=q_labels)
    res["annexure_text"] = annex
    res["has_annexure"] = bool(annex) or referenced

    out = []
    if parts:
        first = parts[0]["labels"][0]
        if (pre and tp.label_scheme(first) == "alpha" and first != "a" and ord(first) - 96 <= 4
                and len(pre.split()) >= 3):
            # the answer to the first part(s) carries no label: "Yes, Sir.\n(b): ..."
            missing = [chr(c) for c in range(97, ord(first))]
            out.append(_mk_part(missing[0], missing, fin(pre), inferred=True))
        elif pre:
            out.append(_mk_part("_pre", [], fin(pre)))
        for p in parts:
            out.append(_mk_part(p["label"], p["labels"], fin(p["text"])))
    elif main.strip():
        out.append(_mk_part("_all", [], fin(main)))
        res["answer_unlabeled"] = True
    res["answer_parts"] = out
    res["answer_labels"] = [l for p in out for l in p["labels"]]
    res["grouped_answer"] = any(p["grouped"] and not p["inferred"] for p in out)
    res["answer_text"] = fin(main) if main.strip() else ""
    return res


def norm_ministry(s):
    s = re.sub(r"\s+", " ", (s or "").replace("&", " AND ")).strip(" ,.-")
    s = re.sub(r"\s+", " ", s).upper()
    return re.sub(r"^THE\s+", "", s)


_MINISTRY_MAP = None
UNMAPPED_MINISTRIES = Counter()


def _squash(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def load_ministry_map(path=None):
    """data/ref/ministry_map.csv -> ({normalised label: (canonical, group)}, {squashed label: (canonical, group)})."""
    exact, squashed = {}, {}
    with open(path or MINISTRY_MAP_CSV, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            if r["ministry_canonical"] and r["ministry_group"]:
                v = (r["ministry_canonical"], r["ministry_group"])
                exact[r["ministry"]] = v
                squashed[_squash(r["ministry"])] = v
    return exact, squashed


def ministry_info(raw):
    """(ministry_canonical, ministry_group) for a raw API ministry label; (None, None) + counted if unmapped."""
    global _MINISTRY_MAP
    if _MINISTRY_MAP is None:
        _MINISTRY_MAP = load_ministry_map()
    key = norm_ministry(raw)
    v = _MINISTRY_MAP[0].get(key) or _MINISTRY_MAP[1].get(_squash(key))
    if v is None:
        UNMAPPED_MINISTRIES[raw] += 1
        return None, None
    return v


def norm_date(d):
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime((d or "").strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def to_int(x):
    try:
        return int(re.sub(r"\D", "", str(x)))
    except ValueError:
        return None


# --------------------------------------------------------------------------------------
# Record building
# --------------------------------------------------------------------------------------
_LONE_ANNEX = re.compile(r"^(?:annexure|annex|statement)\b[\s\-–:.\dIVXivx]*$", re.I)


def _fallback_text(fallback, a_text, a_ok):
    """Decide what a replacement file contributes.

    fallback = (path, lines, stats, err).  Returns dict(kind, note, needs_ocr, mode, a_text, q_body, hindi,
    annex, qh).  mode is 'full' (docx/pdf reply holding question + answer; `a_text` replaces the answer),
    'body' (htm statement/annexure body appended to the API reply header; `a_text` is the combined text),
    'annex' (the answer is already fine, the htm text only fills annexure_text) or None (nothing usable)."""
    path, lines, stats, err = fallback
    kind = os.path.splitext(path)[1].lower().lstrip(".")
    kind = "htm" if kind == "html" else kind
    out = {"kind": kind, "note": None, "needs_ocr": False, "mode": None, "a_text": a_text, "q_body": "",
           "hindi": False, "annex": "", "qh": False}
    nl = [tp.normalize_line(l) for l in lines]
    nl = [l for l in nl if l]
    if err:
        out["note"] = "fallback_bad_" + err.split(":")[-1][:30]
        return out
    if not nl:
        img = bool(stats.get("n_media")) or (kind == "pdf" and bool(stats.get("n_pages")))
        out["note"] = "fallback_image_only" if img else "fallback_empty_doc"
        out["needs_ocr"] = img and not a_ok
        return out
    if kind == "htm":
        ql0, _al0, found0 = split_doc_lines(nl)
        full = found0 and bool(tp._PLEASED.search(" ".join(ql0)[:1500]))  # some .htm are complete replies
        if not full:
            ls_ = list(nl)
            if len(ls_) > 1 and _LONE_ANNEX.match(ls_[0]) and tp._STATEMENT_HEAD.match(ls_[1]):
                ls_ = ls_[1:]
            btxt, bh = tp.clean_text(lines=ls_, wrap=False, strip_numbers=False)
            if not btxt:
                out["note"] = "fallback_empty_doc"
                return out
            out["hindi"] = bh or bool(stats.get("legacy_hindi"))
            if a_ok:
                out.update(mode="annex", annex=btxt)
            else:
                out.update(mode="body", a_text=(a_text.rstrip() + "\n" + btxt) if a_text else btxt)
            return out
    if a_ok:
        return out  # full reply documents only replace missing answers
    ql, al, found = split_doc_lines(nl)
    atxt, ah = tp.clean_text(lines=al, wrap=False, strip_numbers=True)
    if atxt:
        qtxt, qh = tp.clean_text(lines=ql, wrap=False, strip_numbers=True)
        body, _ = extract_question_body(qtxt)
        out.update(mode="full", a_text=atxt, hindi=ah, q_body=body, qh=qh)
        return out
    eng = len(re.findall(r"[A-Za-z]{3,}", " ".join(nl)))
    out["note"] = "fallback_hindi_only" if eng < 15 else "fallback_no_answer_heading"
    return out


def build_record(ls, q, doc=None, fallback=None):
    """q: API record. doc: (path, lines, stats, err) or None.
    fallback: same tuple for a replacement file from data/raw/docs_fallback (used only when the answer
    would otherwise be missing / an empty 'statement is laid' stub)."""
    qtype_raw = (q.get("type") or "").strip().upper()
    qt = "S" if qtype_raw.startswith("S") else "U"
    date_iso = norm_date(q.get("date"))
    session = to_int(q.get("sessionNo"))
    ques_no = to_int(q.get("quesNo"))
    qid = f"ls{ls}-s{session}-{qt}-{ques_no}-{(date_iso or '').replace('-', '')}"
    members = [re.sub(r"\s+", " ", m).strip() for m in (q.get("member") or []) if m and m.strip()]

    # --- API text
    q_api, hq1 = tp.clean_text(raw_html=q.get("questionText"))
    a_api, ha1 = tp.clean_text(raw_html=q.get("answerText"))
    source, source_q, source_a = "api", ("api" if q_api else None), ("api" if a_api else None)
    q_has_hindi, a_has_hindi = hq1, ha1
    extract_error, doc_file, doc_pages = None, None, None
    q_text, a_text = q_api, a_api
    wrap = False
    doc_note = None

    if doc is not None and (not a_api or not q_api):
        path, lines, dstats, err = doc
        doc_file = os.path.relpath(path, ROOT).replace("\\", "/")
        extract_error = err
        doc_pages = dstats.get("n_pages")
        src = "docx" if path.lower().endswith(".docx") else "pdf"
        nl = [tp.normalize_line(l) for l in lines]
        nl = [l for l in nl if l]
        ql, al, found = split_doc_lines(nl)
        qtxt, qh = tp.clean_text(lines=ql, wrap=False, strip_numbers=True)
        atxt, ah = tp.clean_text(lines=al, wrap=False, strip_numbers=True)
        body, how = extract_question_body(qtxt)
        if err:
            doc_note = "bad_" + err.split(":")[-1][:30]  # e.g. bad_legacy_ole_doc, bad_no_document_xml
        elif not nl:
            doc_note = "image_only" if dstats.get("n_media") else "empty_doc"
        elif not found:
            doc_note = "no_answer_heading"
        if body or atxt:
            source = src
            if not q_api and body:
                q_text, source_q, q_has_hindi = tp.reflow(body), src, qh
            if not a_api and atxt:
                a_text, source_a, a_has_hindi = atxt, src, ah
                wrap = True
        if doc_note is None and how == "none":
            doc_note = "question_front_matter_unparsed"

    # The API occasionally files the question front matter under answerText (or the reply under
    # questionText).  Swap when both are wrong, otherwise blank the offending field; flag either way.
    q_is_ans = bool(q_text) and bool(re.match(r"^\W*(?:ANSWER\b|(?:THE\s+)?(?:PRIME\s+)?MINISTER\b)", q_text, re.I)) \
        and bool(tp.parse_minister_header(q_text)[0]["minister_title"])
    a_is_q = bool(a_text) and (bool(tp._PLEASED.search(a_text[:600]))
                               or bool(re.match(r"^\W*(?:O\.?I\.?H\.?\s*)?GOVERNMENT OF INDIA", a_text)))
    qa_mismatch = q_is_ans or a_is_q
    if a_is_q and q_is_ans:
        q_text, a_text = a_text, q_text
        source_q, source_a = source_a, source_q
    elif a_is_q:
        a_text, source_a = "", None
    elif q_is_ans:
        q_text, source_q = "", None


    fallback_file, needs_ocr, fb_mode, fb_kind, fb_annex = None, False, None, None, ""
    if fallback is not None:
        probe = process_answer(a_text, wrap=wrap) if a_text else None
        a_ok = bool(probe and probe["answer_text"].strip())
        fb = _fallback_text(fallback, a_text, a_ok)
        fb_kind = fb["kind"]
        if fb["note"] or fb["mode"]:
            fallback_file = os.path.relpath(fallback[0], ROOT).replace("\\", "/")
        if fb["note"]:
            doc_note = f"{doc_note};{fb['note']}" if doc_note else fb["note"]
        needs_ocr = fb["needs_ocr"]
        fb_mode = fb["mode"]
        if fb_mode in ("full", "body"):
            a_text, source_a, source = fb["a_text"], "fallback_" + fb_kind, "fallback_" + fb_kind
            a_has_hindi = a_has_hindi or fb["hindi"]
            wrap = fb_mode == "full"
            if fb_mode == "full" and not q_text and fb["q_body"]:
                q_text, source_q, q_has_hindi = tp.reflow(fb["q_body"]), "fallback_" + fb_kind, fb["qh"]
        elif fb_mode == "annex":
            fb_annex = fb["annex"]

    qtext, qparts, qlabels = process_question(q_text)
    q_scheme = (qparts[-1]['labels'] and tp.label_scheme(qparts[-1]['labels'][0])) if qparts else None
    ans = process_answer(a_text, q_scheme=q_scheme, wrap=wrap, q_labels=qlabels)
    if fb_mode == "body" and not ans["answer_text"].strip() and ans["annexure_text"].strip():
        # the replacement file is itself only an annexure/table: that IS the reply content
        t = ans["annexure_text"]
        ans.update(answer_text=t, answer_parts=[_mk_part("_all", [], t)], answer_labels=[], statement_empty=False,
                   answer_unlabeled=True, annexure_text="", has_annexure=True)
        doc_note = f"{doc_note};fallback_annexure_only" if doc_note else "fallback_annexure_only"
    if fb_annex and not ans["annexure_text"].strip():
        ans.update(annexure_text=fb_annex, has_annexure=True)
    ans_text = ans["answer_text"]

    n_eng = len(re.findall(r"[A-Za-z]{3,}", ans_text))
    answer_garbled = bool(ans_text) and n_eng < 4 and ("?" in ans_text or bool(a_has_hindi))
    covered = set(ans["answer_labels"])
    all_cov = (all(l in covered for l in qlabels) if qlabels and ans["answer_labels"] else
               (False if qlabels else None))
    answer_missing = not ans_text.strip()
    needs_ocr = bool(needs_ocr or (answer_missing and doc_note and "image_only" in doc_note.split(";")))
    mmap = ministry_info(q.get("ministry"))
    rec = {
        "qid": qid, "ls": ls, "session": session, "date": date_iso,
        "qtype": "starred" if qt == "S" else "unstarred", "ques_no": ques_no,
        "subject": re.sub(r"\s+", " ", q.get("subjects") or "").strip(),
        "ministry": norm_ministry(q.get("ministry")), "ministry_raw": q.get("ministry"),
        "ministry_canonical": mmap[0], "ministry_group": mmap[1],
        "members": members, "n_members": len(members),
        "question_text": qtext, "answer_text": ans_text,
        "question_parts": qparts, "answer_parts": ans["answer_parts"],
        "question_labels": qlabels, "answer_labels": ans["answer_labels"],
        "n_question_parts": len([p for p in qparts if p["label"] != "_pre"]),
        "n_answer_parts": len([p for p in ans["answer_parts"] if p["label"] not in ("_pre", "_all")]),
        "all_parts_covered": all_cov,
        "minister_name": ans["minister_name"], "minister_title": ans["minister_title"],
        "minister_name_raw": ans["minister_name_raw"], "minister_rank": ans["minister_rank"],
        "answer_missing": answer_missing, "question_missing": not qtext.strip(),
        "statement_laid": ans["statement_laid"], "statement_empty": ans["statement_empty"],
        "answer_start_repaired": ans["answer_start_repaired"],
        "answer_label_inferred": any(p["inferred"] for p in ans["answer_parts"]),
        "qa_mismatch": qa_mismatch, "answer_garbled": answer_garbled, "grouped_answer": ans["grouped_answer"],
        "answer_unlabeled": ans["answer_unlabeled"] and not answer_missing,
        "boilerplate": tp.boilerplate_flags(ans_text),
        "has_annexure": ans["has_annexure"], "annexure_len": len(ans["annexure_text"]),
        "annexure_text": ans["annexure_text"],
        "question_has_hindi": bool(q_has_hindi), "answer_has_hindi": bool(a_has_hindi),
        "answer_len": len(ans_text), "question_len": len(qtext),
        "source": source, "source_q": source_q, "source_a": source_a,
        "doc_key": doc_key(ls, q), "doc_file": doc_file, "doc_pages": doc_pages,
        "doc_note": doc_note, "extract_error": extract_error,
        "has_supplementary": bool(q.get("supplementaryQuestionResDtoList")),
        "supplementary_json": json.dumps(q["supplementaryQuestionResDtoList"], ensure_ascii=False)
        if q.get("supplementaryQuestionResDtoList") else None,
        "pdf_url": q.get("questionsFilePath"), "docx_url": q.get("questionsDocPath"),
        "needs_ocr": needs_ocr, "fallback_file": fallback_file,
    }
    return rec


def align_parts(question_parts, answer_parts):
    """Helper for downstream use: for each question part label, the answer text addressing it.

    Returns {label: {"question": str, "answer": str, "answer_labels": [...], "shared": bool}}.
    Falls back to the whole '_all' answer when the answer is unlabeled."""
    out = {}
    all_txt = next((p["text"] for p in answer_parts if p["label"] == "_all"), None)
    for qp in question_parts:
        if qp["label"] == "_pre":
            continue
        ap = next((a for a in answer_parts if qp["label"] in a["labels"]), None)
        if ap is not None:
            out[qp["label"]] = {"question": qp["text"], "answer": ap["text"], "answer_labels": ap["labels"],
                                "shared": len(ap["labels"]) > 1}
        elif all_txt is not None:
            out[qp["label"]] = {"question": qp["text"], "answer": all_txt, "answer_labels": [], "shared": True}
        else:
            out[qp["label"]] = {"question": qp["text"], "answer": None, "answer_labels": [], "shared": False}
    return out


# --------------------------------------------------------------------------------------
# I/O  (streaming: raw pages are read twice, rows are written in batches, so memory stays flat even
# for the full ~400k-question corpus)
# --------------------------------------------------------------------------------------
def list_pages(lss):
    out = []
    for ls in lss:
        d = os.path.join(RAW_API, f"ls{ls}")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json.gz"):  # skips *.part
                out.append((ls, os.path.join(d, fn)))
    return out


def read_page(path):
    try:
        with gzip.open(path, "rb") as f:
            return json.loads(f.read())[0].get("listOfQuestions") or []
    except Exception:
        return None  # partial / being written


def api_qid(ls, q):
    qt = "S" if (q.get("type") or "").strip().upper().startswith("S") else "U"
    return f"ls{ls}-s{to_int(q.get('sessionNo'))}-{qt}-{to_int(q.get('quesNo'))}-{(norm_date(q.get('date')) or '').replace('-', '')}"


def find_doc(ls, key):
    d = os.path.join(RAW_DOCS, f"ls{ls}")
    for ext in (".docx", ".pdf"):
        p = os.path.join(d, key + ext)
        if os.path.exists(p):
            return p
    return None


_FB_PREF = {".docx": 0, ".pdf": 1, ".htm": 2, ".html": 2}


def fallback_index(root=None):
    """{qid: path} for the replacement answer files data/raw/docs_fallback/ls<N>/<qid>.<ext>.
    *.part / *.tmp files are ignored; docx beats pdf beats htm when several exist for a qid."""
    root = root or RAW_FALLBACK
    idx = {}
    if not os.path.isdir(root):
        return idx
    for d in sorted(os.listdir(root)):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        for fn in os.listdir(dd):
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext not in _FB_PREF:
                continue
            p = os.path.join(dd, fn)
            if stem not in idx or _FB_PREF[ext] < _FB_PREF[os.path.splitext(idx[stem])[1].lower()]:
                idx[stem] = p
    return idx


def schema():
    import pyarrow as pa

    part = pa.struct([("label", pa.string()), ("labels", pa.list_(pa.string())), ("grouped", pa.bool_()),
                      ("text", pa.string()), ("inferred", pa.bool_())])
    S, I, B = pa.string(), pa.int64(), pa.bool_()
    return pa.schema([
        ("qid", S), ("ls", I), ("session", I), ("date", S), ("qtype", S), ("ques_no", I), ("subject", S),
        ("ministry", S), ("ministry_raw", S), ("ministry_canonical", S), ("ministry_group", S), ("members", pa.list_(S)), ("n_members", I),
        ("question_text", S), ("answer_text", S), ("question_parts", pa.list_(part)),
        ("answer_parts", pa.list_(part)), ("question_labels", pa.list_(S)), ("answer_labels", pa.list_(S)),
        ("n_question_parts", I), ("n_answer_parts", I), ("all_parts_covered", B),
        ("minister_name", S), ("minister_title", S), ("minister_name_raw", S), ("minister_rank", S),
        ("answer_missing", B), ("question_missing", B), ("statement_laid", B), ("statement_empty", B),
        ("grouped_answer", B), ("answer_unlabeled", B), ("answer_start_repaired", B),
        ("answer_label_inferred", B), ("qa_mismatch", B), ("answer_garbled", B),
        ("boilerplate", pa.list_(S)), ("has_annexure", B), ("annexure_len", I),
        ("annexure_text", S), ("question_has_hindi", B), ("answer_has_hindi", B), ("answer_len", I),
        ("question_len", I), ("source", S), ("source_q", S), ("source_a", S), ("doc_key", S),
        ("doc_file", S), ("doc_pages", I), ("doc_note", S), ("extract_error", S), ("has_supplementary", B),
        ("supplementary_json", S), ("pdf_url", S), ("docx_url", S), ("needs_ocr", B), ("fallback_file", S),
    ])


class Stats:
    """Incremental per-LS validation counters (no rows kept in memory)."""

    def __init__(self):
        self.d = defaultdict(lambda: defaultdict(int))
        self.sources = defaultdict(Counter)
        self.boiler = defaultdict(Counter)
        self.notes = defaultdict(Counter)
        self.alen = defaultdict(list)
        self.fb = defaultdict(Counter)       # per LS: fallback outcome counters
        self.missing = defaultdict(Counter)  # per LS: why answer_text is still empty
        self.groups = defaultdict(Counter)   # per LS: ministry_group counts

    def add(self, r):
        s, ls = self.d[r["ls"]], r["ls"]
        s["rows"] += 1
        s["q_text"] += r["question_len"] > 0
        s["a_text"] += r["answer_len"] > 0
        s["q_part"] += r["n_question_parts"] > 0
        s["a_part"] += r["n_answer_parts"] > 0
        if r["n_question_parts"] > 0:
            s["qwith"] += 1
            s["covered"] += bool(r["all_parts_covered"])
            s["cov_or_unl"] += bool(r["all_parts_covered"] or r["answer_unlabeled"])
            s["qwith_ans"] += not r["answer_missing"]
            s["covered_of_ans"] += bool(r["all_parts_covered"])
        if not r["answer_missing"]:
            s["with_ans"] += 1
            s["minister"] += bool(r["minister_name"])
            s["minister_title"] += bool(r["minister_title"])
        for k in ("answer_missing", "statement_laid", "statement_empty", "grouped_answer", "has_annexure",
                  "answer_unlabeled", "answer_start_repaired", "answer_label_inferred", "qa_mismatch",
                  "answer_garbled", "question_has_hindi", "answer_has_hindi", "needs_ocr"):
            s[k] += bool(r[k])
        self.sources[ls][r["source"]] += 1
        for b in r["boilerplate"]:
            self.boiler[ls][b] += 1
        if r["doc_note"]:
            self.notes[ls][r["doc_note"]] += 1
        if r["extract_error"]:
            s["extract_errors"] += 1
        self.alen[ls].append(r["answer_len"])
        if r["fallback_file"]:
            ext = os.path.splitext(r["fallback_file"])[1].lstrip(".")
            self.fb[ls]["files_consulted_" + ext] += 1
            if (r["source_a"] or "").startswith("fallback_"):
                self.fb[ls]["answer_recovered_" + ext] += 1
            elif r["answer_missing"]:
                self.fb[ls]["not_recovered_" + ext] += 1
            else:
                self.fb[ls]["answer_ok_annexure_attached_" + ext] += 1
        self.fb[ls]["needs_ocr"] += bool(r["needs_ocr"])
        if r["answer_missing"]:
            if r["needs_ocr"]:
                why = "needs_ocr (image-only document, no extractable text)"
            elif r["statement_empty"]:
                why = "statement_laid_but_body_missing" + (" (replacement file lacks the body)" if r["fallback_file"] else "")
            elif r["doc_note"]:
                why = "doc:" + r["doc_note"]
            elif not r["source_a"]:
                why = "no_answer_in_api_or_documents"
            else:
                why = "other"
            self.missing[ls][why] += 1
        self.groups[ls][r["ministry_group"]] += 1

    def report(self):
        out = {}
        for ls in sorted(self.d):
            s = self.d[ls]
            n = s["rows"]

            def p(k, den=n):
                return round(100 * s[k] / max(den, 1), 1)

            al = sorted(self.alen[ls])
            out[ls] = {
                "rows": n, "sources": dict(self.sources[ls]),
                "pct_question_text": p("q_text"), "pct_answer_text": p("a_text"),
                "pct_ge1_question_part": p("q_part"), "pct_ge1_answer_part": p("a_part"),
                "pct_all_qparts_covered_(of_rows_with_qparts)": p("covered", s["qwith"]),
                "pct_all_qparts_covered_(of_rows_with_qparts_and_answer)": p("covered_of_ans", s["qwith_ans"]),
                "pct_covered_or_answer_unlabeled_(of_rows_with_qparts)": p("cov_or_unl", s["qwith"]),
                "pct_minister_name_parsed_(of_rows_with_answer)": p("minister", s["with_ans"]),
                "pct_minister_title_parsed_(of_rows_with_answer)": p("minister_title", s["with_ans"]),
                "pct_answer_missing": p("answer_missing"),
                "pct_statement_laid": p("statement_laid"),
                "pct_statement_laid_but_empty": p("statement_empty"),
                "pct_grouped_answer": p("grouped_answer"),
                "pct_has_annexure": p("has_annexure"),
                "pct_answer_unlabeled": p("answer_unlabeled"),
                "pct_answer_start_repaired": p("answer_start_repaired"),
                "pct_qa_mismatch": p("qa_mismatch"),
                "pct_answer_garbled": p("answer_garbled"),
                "n_needs_ocr": s["needs_ocr"],
                "pct_question_has_hindi_removed": p("question_has_hindi"),
                "pct_answer_has_hindi_removed": p("answer_has_hindi"),
                "boilerplate_counts": dict(self.boiler[ls]),
                "doc_notes": dict(self.notes[ls]),
                "extract_errors": s["extract_errors"],
                "median_answer_len": al[n // 2] if al else 0,
                "fallback": dict(self.fb[ls]),
                "answer_missing_reasons": dict(self.missing[ls]),
                "ministry_group_counts": dict(self.groups[ls]),
            }
        return out


def _batch_table(rows, sch):
    import pyarrow as pa

    return pa.Table.from_pydict({f.name: [r.get(f.name) for r in rows] for f in sch}, schema=sch)


def fuzzy_canonical(variants, cutoff, rare, frequent):
    """variants: {letters-only key: Counter(spelling)}.  Returns {key: canonical spelling}.

    1. keys that differ only by spaces/punctuation collapse to the spelling with most word breaks;
    2. rare keys (count < rare) within `cutoff` similarity of a frequent key (count >= frequent) are
       merged into it (single-letter typos such as 'Minstry')."""
    import math

    from rapidfuzz import fuzz, process

    # word-level document frequency over *distinct spellings*: glued tokens ('Anupriyapatel',
    # 'Ministryofhealth', 'A Viation') are rare, proper words are common
    df = Counter(w for c in variants.values() for v in c for w in set(re.findall(r"[a-z]+", v.lower())))

    def plaus(v):
        ws = re.findall(r"[a-z]+", v.lower())
        return sum(math.log(df[w]) for w in ws) / max(len(ws), 1)

    canon = {k: max(c.items(), key=lambda kv: (round(plaus(kv[0]), 3), kv[1]))[0] for k, c in variants.items()}
    total = {k: sum(c.values()) for k, c in variants.items()}
    freq_keys = [k for k, n in total.items() if n >= frequent]
    out = dict(canon)
    if freq_keys:
        for k, n in total.items():
            if n < rare and k not in freq_keys:
                hit = process.extractOne(k, freq_keys, scorer=fuzz.ratio, score_cutoff=cutoff)
                if hit:
                    out[k] = canon[hit[0]]
    return out


def apply_canonical(path, name_map, title_map):
    """Second pass: rewrite minister_name / minister_title columns through the canonical maps."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    tmp = path + ".tmp2"
    w = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
    for b in pf.iter_batches(batch_size=20000):
        t = pa.Table.from_batches([b])
        for fld, mp in (("minister_name", name_map), ("minister_title", title_map)):
            vals = [mp.get(re.sub(r"[^a-z]", "", v.lower()), v) if v else v for v in t.column(fld).to_pylist()]
            t = t.set_column(t.schema.get_field_index(fld), fld, pa.array(vals, pa.string()))
        w.write_table(t)
    w.close()
    del pf
    os.replace(tmp, path)


def main(argv=None):
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls", type=int, nargs="*", default=LOK_SABHAS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many pages")
    ap.add_argument("--out", default=os.path.join(CLEAN_DIR, "questions.parquet"))
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # ---- pass A: pick the best raw record per qid (dedupe) and find which reply documents we need
    pages = list_pages(args.ls)
    if args.limit:
        pages = pages[: args.limit]
    best, n_read, n_bad_pages = {}, 0, 0
    for ls, path in pages:
        qs = read_page(path)
        if qs is None:
            n_bad_pages += 1
            continue
        for i, q in enumerate(qs):
            n_read += 1
            score = (len((q.get("answerText") or "")), len((q.get("questionText") or "")))
            qid = api_qid(ls, q)
            if qid not in best or score > best[qid][0]:
                best[qid] = (score, path, i)
    print(f"[clean] {len(pages)} pages, {n_read} API records, {len(best)} unique qids, {time.time() - t0:.0f}s", flush=True)
    by_page = defaultdict(list)
    for qid, (_, path, i) in best.items():
        by_page[path].append(i)

    need = {}  # path of reply docs to extract (only for records lacking API text)
    for ls, path in pages:
        if path not in by_page:
            continue
        qs = read_page(path) or []
        for i in by_page[path]:
            if i >= len(qs):
                continue
            q = qs[i]
            if (q.get("answerText") or "").strip() and (q.get("questionText") or "").strip():
                continue
            p = find_doc(ls, doc_key(ls, q))
            if p:
                need[p] = 1
    fb_idx = fallback_index()
    for qid in best:  # replacement files are extracted for every qid that has one
        if qid in fb_idx:
            need[fb_idx[qid]] = 1
    paths = list(need)
    print(f"[clean] {len(paths)} reply documents to extract ({len(fb_idx)} fallback files indexed)", flush=True)
    if paths:
        if args.workers > 1 and len(paths) > 40:
            with ProcessPoolExecutor(args.workers) as ex:
                for i, _ in enumerate(ex.map(_worker, paths, chunksize=16)):
                    if i and i % 5000 == 0:
                        print(f"[clean]   extracted {i}/{len(paths)}", flush=True)
        # (results are cached on disk; the main loop below re-reads them from the cache)

    # ---- pass B: build rows, stream into parquet
    sch = schema()
    tmp_out = args.out + ".tmp"
    writer = pq.ParquetWriter(tmp_out, sch, compression="zstd")
    stats = Stats()
    names, titles = defaultdict(Counter), defaultdict(Counter)
    buf, n_rows = [], 0
    for ls, path in pages:
        if path not in by_page:
            continue
        qs = read_page(path)
        if qs is None:
            continue
        for i in sorted(by_page[path]):
            if i >= len(qs):
                continue
            q = qs[i]
            p = None
            if not ((q.get("answerText") or "").strip() and (q.get("questionText") or "").strip()):
                p = find_doc(ls, doc_key(ls, q))
            d = None
            if p:
                lines, st, err = load_doc_lines(p)
                d = (p, lines, st, err)
            fbp = fb_idx.get(api_qid(ls, q))
            fb = None
            if fbp:
                flines, fst, ferr = load_doc_lines(fbp)
                fb = (fbp, flines, fst, ferr)
            rec = build_record(ls, q, d, fb)
            stats.add(rec)
            for fld, store in (("minister_name", names), ("minister_title", titles)):
                if rec[fld]:
                    store[re.sub(r"[^a-z]", "", rec[fld].lower())][rec[fld]] += 1
            buf.append(rec)
            n_rows += 1
            if len(buf) >= 4000:
                writer.write_table(_batch_table(buf, sch))
                buf = []
    if buf:
        writer.write_table(_batch_table(buf, sch))
    writer.close()
    os.replace(tmp_out, args.out)

    # ---- canonical minister spellings (docx conversion glues words: 'ANUPRIYAPATEL')
    name_map = fuzzy_canonical(names, cutoff=95, rare=5, frequent=30)
    title_map = fuzzy_canonical(titles, cutoff=94, rare=20, frequent=40)
    apply_canonical(args.out, name_map, title_map)

    rep = {"generated": datetime.now().isoformat(timespec="seconds"), "api_pages": len(pages),
           "unreadable_pages": n_bad_pages, "api_records_read": n_read, "rows": n_rows,
           "duplicates_dropped": n_read - n_rows, "seconds": round(time.time() - t0, 1),
           "per_ls": stats.report(),
           "ministry_unmapped": dict(UNMAPPED_MINISTRIES),
           "ministry_map": {"file": os.path.relpath(MINISTRY_MAP_CSV, ROOT).replace("\\", "/"),
                            "n_labels": len(_MINISTRY_MAP[0]) if _MINISTRY_MAP else 0,
                            "n_canonical": len({v[0] for v in _MINISTRY_MAP[0].values()}) if _MINISTRY_MAP else 0},
           "fallback_files_indexed": len(fb_idx)}
    with open(os.path.join(CLEAN_DIR, "clean_report.json"), "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_ls"}))
    for ls, v in rep["per_ls"].items():
        print(f"[clean] LS{ls}: {v['rows']} rows; answer text {v['pct_answer_text']}%; "
              f"qparts>=1 {v['pct_ge1_question_part']}%; minister {v['pct_minister_name_parsed_(of_rows_with_answer)']}%")
    if rep["ministry_unmapped"]:
        print("!" * 78, file=sys.stderr)
        print(f"[clean] ERROR: {len(rep['ministry_unmapped'])} ministry label(s) missing from "
              f"{MINISTRY_MAP_CSV}: {rep['ministry_unmapped']}", file=sys.stderr)
        print("[clean] rows were written with ministry_canonical/ministry_group = null; add the labels to "
              "pipeline/ministry_build.py RULES and re-run it.", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
    return rep


if __name__ == "__main__":
    _rep = main()
    sys.exit(2 if _rep["ministry_unmapped"] else 0)
