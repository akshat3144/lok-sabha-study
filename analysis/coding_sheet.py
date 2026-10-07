"""M3 human-coding sheets: build one Excel workbook per coder and import the filled sheets.

    python -I -X utf8 analysis/coding_sheet.py make   --stage pilot --coders coder1 coder2
    python -I -X utf8 analysis/coding_sheet.py import --stage pilot            # all filled sheets of the stage
    python -I -X utf8 analysis/coding_sheet.py import --allow-incomplete ...   # keep only fully valid rows

make   -> data/coding/sheet_<coder>_<stage>.xlsx   (sheets: Instructions, Coding, Meta[hidden])
          Each coder's sheet contains only the questions assigned to that coder in <stage>_sample.csv (`coders`
          column: "coder1;coder2" = double-coded, "coder1" / "coder2" = single-coded). Re-running `make` after a
          codebook change refreshes the Instructions text and sheet_version without touching the sample.
import -> data/coding/human_labels.csv with columns
          part_id, qid, coder, label, evasion_type, request_type, note, sheet_version, stage
          (rows of the imported (coder, stage) pairs are replaced; other rows are kept)

Blindness: coders see only Lok Sabha number and ministry. NOT shown: asker/party/government status, date, session,
qtype, boilerplate flag, M1/M2 outputs, minister's name is only present if it is inside the answer text itself.
Every coder gets an independently randomised question order (parts of a question stay together, in order).
"""
import argparse
import hashlib
import math
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import IllegalCharacterError  # noqa: F401  (kept for clarity)
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import codebook as cb  # noqa: E402

SEED = 20261017
CELL_MAX = 32000
NONE_APPLICABLE = "NONE_APPLICABLE"
EVASION_CHOICES = cb.EVASION_CHOICES
ANNEXURE_CHARS = cb.ANNEXURE_CHARS
TRUNC_MARKER = "[... truncated ...]"  # appended verbatim after the cut text; same marker as the M2 prompt
HEADERS = ["part_id", "q_seq", "lok_sabha", "ministry", "QUESTION (part to code in red bold)",
           "PART TO CODE", "ANSWER TEXT FOR THIS PART", "ANNEXURE (truncated)", "label", "evasion_type", "request_type", "note",
           "check", "FULL ANSWER (all parts)"]
COL = {h: i + 1 for i, h in enumerate(HEADERS)}
INPUT_COLS = ["label", "evasion_type", "request_type", "note"]
LABEL_COLS = ["part_id", "qid", "coder", "label", "evasion_type", "request_type", "note", "sheet_version", "stage"]
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ------------------------------------------------------------------------------------------------ helpers
def part_set_hash(part_ids):
    return hashlib.sha1("\n".join(sorted(part_ids)).encode("utf-8")).hexdigest()[:8]


def sheet_version(stage, part_ids, codebook_version=None):
    """<codebook version>-<stage>-<hash of the coder's part_id set>. The importer checks only the stage and the
    hash (see check_sheet_version), so sheets issued under an older codebook version still import."""
    return f"{codebook_version or cb.CODEBOOK_VERSION}-{stage}-{part_set_hash(part_ids)}"


def check_sheet_version(version, stage, part_ids):
    """True if `version` (from the sheet's Meta) was issued for exactly this stage and part_id set, whatever
    codebook version it embeds."""
    m = re.fullmatch(r"(.+)-([A-Za-z0-9_]+)-([0-9a-f]{8})", str(version))
    return bool(m) and m.group(2) == str(stage) and m.group(3) == part_set_hash(part_ids)


def clean_cell(text):
    t = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    t = _ILLEGAL.sub("", t)
    if len(t) > CELL_MAX:
        t = t[:CELL_MAX] + "\n[... TRUNCATED, see source ...]"
    return t


def _set_text(cell, text):
    cell.value = clean_cell(text)
    cell.data_type = "s"  # never let '=...' text turn into a formula


def coder_order(qids, coder, stage, seed=SEED):
    """Deterministic coder-specific random permutation of the question ids."""
    rng = np.random.default_rng([seed, zlib.crc32(coder.encode()), zlib.crc32(stage.encode())])
    qids = list(qids)
    perm = rng.permutation(len(qids))
    return [qids[i] for i in perm]


def est_height(texts_widths, line_h=13.0):
    lines = 1
    for t, w in texts_widths:
        n = sum(max(1, math.ceil(len(seg) / max(w, 1) * 1.1)) for seg in str(t).split("\n"))
        lines = max(lines, n)
    return min(409.0, line_h * lines + 4)


def highlighted_question(question, part_text, label):
    """Rich-text question with the part text in bold red (plain text if it cannot be located)."""
    q = clean_cell(question)
    pt = (part_text or "").strip()
    if label == "_all" or not pt:
        return q
    i = q.find(pt)
    if i < 0:  # whitespace-insensitive fallback
        pat = r"\s+".join(re.escape(w) for w in pt.split())
        m = re.search(pat, q) if pat else None
        if not m:
            return q
        i, j = m.start(), m.end()
    else:
        j = i + len(pt)
    red = InlineFont(b=True, color="FFC00000")
    blocks = [x for x in (q[:i], TextBlock(red, q[i:j]), q[j:]) if not (isinstance(x, str) and x == "")]
    return CellRichText(*blocks)


# ------------------------------------------------------------------------------------------------ data loading
def load_texts(root, qids):
    """parts rows and question-level text for the sampled qids (full text stays local)."""
    qids = list(qids)
    parts = pd.read_parquet(os.path.join(root, "data", "measures", "parts.parquet"),
                            filters=[("qid", "in", qids)])
    q = pd.read_parquet(os.path.join(root, "data", "final", "questions_analysis.parquet"),
                        columns=["qid", "ls", "ministry", "question_text", "answer_text", "annexure_text", "has_annexure"],
                        filters=[("qid", "in", qids)])
    return parts, q


# ------------------------------------------------------------------------------------------------ workbook
def build_sheet(parts, qtext, sample, coder, stage, out_path, seed=SEED):
    """parts: parts.parquet rows for the sampled qids; qtext: qid, ls, ministry, question_text, answer_text;
    sample: the sample csv (only qid is used). Returns (path, sheet_version, n_parts)."""
    qtext = qtext.set_index("qid")
    order = coder_order(sample["qid"].tolist(), coder, stage, seed)
    pos = {q: i for i, q in enumerate(order)}
    parts = parts[parts["qid"].isin(pos)].copy()
    parts["_ord"] = parts["qid"].map(pos)
    parts["_row"] = np.arange(len(parts))
    parts = parts.sort_values(["_ord", "_row"], kind="stable")
    version = sheet_version(stage, parts["part_id"])

    wb = Workbook()
    wi = wb.active
    wi.title = "Instructions"
    ws = wb.create_sheet("Coding")
    wm = wb.create_sheet("Meta")

    # --- Instructions
    wi.column_dimensions["A"].width = 130
    intro = [f"HUMAN CODING SHEET - coder: {coder} - stage: {stage} - sheet version: {version}", "",
             "HOW TO USE", "1. Go to the 'Coding' sheet. Work top to bottom; questions are in a random order. Each row is "
             "one question part.",
             "2. Read the PART TO CODE and the ANSWER TEXT FOR THIS PART (the full question is shown with the part in red "
             "bold; any annexure text, shortened, is next to the answer; the full answer is in the far-right column). Click a cell to read long text in the formula bar.",
             "3. Fill the yellow columns: label, evasion_type, request_type (drop-downs) and, optionally, a short note. "
             "Use the note for anything ambiguous.",
             "4. If label is FULL, choose evasion_type = NONE_APPLICABLE. If PARTIAL or NONE, pick the best evasion_type.",
             "5. Do not edit grey columns or reorder/delete rows; save often. Return the file when every row is complete.",
             "6. You are deliberately blind to who asked the question and when. Judge only the text. Do not search for "
             "the asker, the party or the date, and do not discuss items with the other coder before adjudication.",
             "", "=" * 100, ""]
    for i, line in enumerate(intro + cb.instructions_text().split("\n") +
                             ["", "=" * 100, "WORKED EXAMPLES",
                              f"See CODEBOOK.md section 6 (codebook {cb.CODEBOOK_VERSION}): 13 worked examples "
                              "with real answers."], start=1):
        c = wi.cell(row=i, column=1)
        _set_text(c, line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if i == 1 or line in ("HOW TO USE", "WORKED EXAMPLES"):
            c.font = Font(bold=True, size=12)

    # --- Coding
    widths = {"part_id": 26, "q_seq": 8, "lok_sabha": 7, "ministry": 20, "QUESTION (part to code in red bold)": 55,
              "PART TO CODE": 38, "ANSWER TEXT FOR THIS PART": 55, "ANNEXURE (truncated)": 45, "label": 11, "evasion_type": 19,
              "request_type": 14, "note": 30, "check": 20, "FULL ANSWER (all parts)": 70}
    hdr_fill = PatternFill("solid", fgColor="FF305496")
    for h, i in COL.items():
        c = ws.cell(row=1, column=i, value=h)
        c.font = Font(bold=True, color="FFFFFFFF")
        c.fill = hdr_fill
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = widths[h]
    ws.row_dimensions[1].height = 32
    grey = PatternFill("solid", fgColor="FFEDEDED")
    band = [PatternFill("solid", fgColor="FFFFFFFF"), PatternFill("solid", fgColor="FFF2F7FC")]
    entry = PatternFill("solid", fgColor="FFFFF9D6")
    thin = Side(style="thin", color="FFBFBFBF")
    top_border = Border(top=Side(style="medium", color="FF305496"))
    wrap_top = Alignment(wrap_text=True, vertical="top")

    r = 1
    last_q, seq, parity = None, 0, 0
    for row in parts.itertuples(index=False):
        r += 1
        qid = row.qid
        if qid != last_q:
            seq += 1
            parity ^= 1
            last_q = qid
            first_of_q = True
        else:
            first_of_q = False
        qt = qtext.loc[qid]
        label = row.part_label
        part_disp = ("[whole question]" if label == "_all" else f"({label}) ") + clean_cell(row.question_part or "")
        covers = [str(x) for x in (list(row.answer_labels) if row.answer_labels is not None else [])]
        if getattr(row, "shared_answer", False) and len(covers) > 1:
            part_disp += f"\n[answer text is shared with parts: {', '.join(covers)}]"
        if getattr(row, "answer_missing_for_part", False) or not str(row.answer_part or "").strip():
            ans = "[NO ANSWER TEXT FOR THIS PART]"
        else:
            ans = row.answer_part
        full_a = qt["answer_text"] if isinstance(qt["answer_text"], str) and qt["answer_text"].strip() \
            else "[NO ANSWER TEXT]"
        ann = qt["annexure_text"] if isinstance(qt["annexure_text"], str) else ""
        if ann.strip():
            ann = ann.strip()
            if len(ann) > ANNEXURE_CHARS:
                ann = ann[:ANNEXURE_CHARS] + TRUNC_MARKER  # identical to the M2 prompt
        elif bool(qt["has_annexure"]):
            ann = "[annexure referred to, text not available]"
        vals = {"part_id": row.part_id, "q_seq": f"Q{seq:03d}", "lok_sabha": int(qt["ls"]),
                "ministry": qt["ministry"], "PART TO CODE": part_disp, "ANSWER TEXT FOR THIS PART": ans,
                "ANNEXURE (truncated)": ann, "FULL ANSWER (all parts)": full_a}
        for h, v in vals.items():
            c = ws.cell(row=r, column=COL[h])
            if isinstance(v, str):
                _set_text(c, v)
            else:
                c.value = v
        qc = ws.cell(row=r, column=COL["QUESTION (part to code in red bold)"])
        hq = highlighted_question(qt["question_text"], row.question_part, label)
        qc.value = hq
        for h, i in COL.items():
            c = ws.cell(row=r, column=i)
            c.alignment = wrap_top
            c.fill = entry if h in INPUT_COLS else (grey if h in ("part_id", "q_seq", "lok_sabha", "ministry", "check")
                                                     else band[parity])
            c.protection = Protection(locked=h not in INPUT_COLS)
            c.border = Border(top=top_border.top, bottom=thin) if first_of_q else Border(bottom=thin)
        chk = ws.cell(row=r, column=COL["check"])
        L, E, Rq = (f"{get_column_letter(COL[k])}{r}" for k in ("label", "evasion_type", "request_type"))
        chk.value = (f'=IF(AND({L}="",{E}="",{Rq}=""),"",IF(OR({L}="",{Rq}=""),"incomplete",'
                     f'IF(AND({L}="FULL",{E}<>"",{E}<>"{NONE_APPLICABLE}"),"FULL needs NONE_APPLICABLE",'
                     f'IF(AND({L}<>"FULL",OR({E}="",{E}="{NONE_APPLICABLE}")),"give an evasion_type","ok"))))')
        ws.row_dimensions[r].height = est_height([
            (vals["PART TO CODE"], widths["PART TO CODE"]), (ans, widths["ANSWER TEXT FOR THIS PART"]),
            (qt["question_text"], widths["QUESTION (part to code in red bold)"])])
    n = r - 1

    def dv(col, items, title):
        d = DataValidation(type="list", formula1='"' + ",".join(items) + '"', allow_blank=True,
                           showErrorMessage=True, errorTitle="Invalid value", error=f"Choose one of: {', '.join(items)}",
                           showInputMessage=True, promptTitle=title, prompt="Pick from the list")
        d.add(f"{get_column_letter(COL[col])}2:{get_column_letter(COL[col])}{max(n + 1, 2)}")
        ws.add_data_validation(d)
    dv("label", cb.LABELS, "label")
    dv("evasion_type", EVASION_CHOICES, "evasion_type")
    dv("request_type", cb.REQUEST_TYPES, "request_type")
    lc = get_column_letter(COL["label"])
    rng = f"{lc}2:{lc}{max(n + 1, 2)}"
    for val, color in (("FULL", "FFC6EFCE"), ("PARTIAL", "FFFFEB9C"), ("NONE", "FFFFC7CE")):
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{val}"'],
                                                      fill=PatternFill("solid", bgColor=color, fgColor=color)))
    ws.freeze_panes = "C2"
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    ws.protection.formatCells = False

    # --- Meta (hidden)
    meta = [("sheet_version", version), ("coder", coder), ("stage", stage), ("codebook_version", cb.CODEBOOK_VERSION),
            ("n_parts", len(parts)), ("n_questions", parts["qid"].nunique()), ("seed", seed),
            ("headers", "|".join(HEADERS))]
    for i, (k, v) in enumerate(meta, start=1):
        wm.cell(row=i, column=1, value=k)
        wm.cell(row=i, column=2, value=v)
    wm.sheet_state = "hidden"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)
    return out_path, version, len(parts)


def coder_sample(sample, coder):
    """Rows of the sample csv assigned to `coder` (the `coders` column, ';'-separated; absent = everyone)."""
    if "coders" not in sample.columns:
        return sample
    mask = sample["coders"].astype(str).map(lambda v: coder in [x.strip() for x in v.split(";")])
    return sample[mask]


def make(root=ROOT, stage="pilot", coders=("coder1", "coder2"), seed=SEED, out_dir=None):
    out_dir = out_dir or os.path.join(root, "data", "coding")
    sample = pd.read_csv(os.path.join(out_dir, f"{stage}_sample.csv"))
    parts, qtext = load_texts(root, sample["qid"])
    res = []
    for c in coders:
        mine = coder_sample(sample, c)
        res.append(build_sheet(parts, qtext, mine, c, stage, os.path.join(out_dir, f"sheet_{c}_{stage}.xlsx"), seed))
    return res


# ------------------------------------------------------------------------------------------------ importer
class ImportProblems(Exception):
    def __init__(self, problems):
        super().__init__(f"{len(problems)} problem(s) in the coding sheets")
        self.problems = problems


def _norm(v):
    if v is None:
        return ""
    return re.sub(r"\s+", "_", str(v).strip().upper())


def read_sheet(path):
    """Return (meta dict, DataFrame of raw rows incl. part_id, label, evasion_type, request_type, note)."""
    wb = load_workbook(path, data_only=True)
    if "Coding" not in wb.sheetnames or "Meta" not in wb.sheetnames:
        raise ImportProblems([(path, "", "workbook lacks Coding/Meta sheet")])
    meta = {str(k): v for k, v in wb["Meta"].iter_rows(min_row=1, max_col=2, values_only=True) if k}
    rows = list(wb["Coding"].iter_rows(values_only=True))
    hdr = [str(h) if h is not None else "" for h in rows[0]]
    if hdr[:len(HEADERS)] != HEADERS:
        raise ImportProblems([(path, "", f"unexpected header row {hdr[:len(HEADERS)]}")])
    df = pd.DataFrame([r[:len(HEADERS)] for r in rows[1:] if r and r[0] is not None], columns=HEADERS)
    return meta, df


def import_sheets(paths, root=ROOT, out_path=None, allow_incomplete=False, sample_dir=None):
    """Validate and merge filled sheets into human_labels.csv. Returns (labels_df, problems).
    Each sheet is validated against its own issued part set (Meta sheet_version hash, stage) and, when
    <sample_dir>/<stage>_sample.csv exists (default: the folder of out_path), against that coder's assigned
    questions in the sample file (the `coders` column)."""
    out_path = out_path or os.path.join(root, "data", "coding", "human_labels.csv")
    sample_dir = sample_dir or os.path.dirname(out_path)
    problems, frames = [], []
    for p in paths:
        meta, df = read_sheet(p)
        fn = os.path.basename(p)
        ver, coder, stage = meta.get("sheet_version"), meta.get("coder"), meta.get("stage")
        if not (ver and coder and stage):
            problems.append((fn, "", "Meta sheet incomplete"))
            continue
        if not check_sheet_version(ver, stage, df["part_id"].astype(str)):
            problems.append((fn, "", "part_id set differs from the issued sheet (rows deleted, added or edited)"))
        sp = os.path.join(sample_dir, f"{stage}_sample.csv")
        if os.path.exists(sp):
            assigned = set(coder_sample(pd.read_csv(sp), coder)["qid"])
            have = set(df["part_id"].astype(str).str.rsplit("#", n=1).str[0])
            if have != assigned:
                problems.append((fn, "", f"question set differs from the questions assigned to {coder} in "
                                         f"{stage}_sample.csv ({len(have - assigned)} not assigned, "
                                         f"{len(assigned - have)} missing)"))
        if df["part_id"].duplicated().any():
            problems.append((fn, "", f"duplicated part_id: {df.loc[df['part_id'].duplicated(), 'part_id'].tolist()[:5]}"))
        if int(meta.get("n_parts", len(df))) != len(df):
            problems.append((fn, "", f"expected {meta.get('n_parts')} rows, found {len(df)}"))
        recs = []
        for row in df.itertuples(index=False):
            pid = str(row.part_id)
            lab, ev, rq = _norm(row.label), _norm(row.evasion_type), _norm(row.request_type)
            note = "" if row.note is None else str(row.note).strip()
            errs = []
            if not lab:
                errs.append("label missing")
            elif lab not in cb.LABELS:
                errs.append(f"invalid label {row.label!r}")
            if not rq:
                errs.append("request_type missing")
            elif rq not in cb.REQUEST_TYPES:
                errs.append(f"invalid request_type {row.request_type!r}")
            if ev and ev not in EVASION_CHOICES:
                errs.append(f"invalid evasion_type {row.evasion_type!r}")
            elif lab == "FULL":
                if ev and ev != NONE_APPLICABLE:
                    errs.append(f"FULL requires evasion_type NONE_APPLICABLE (got {ev})")
                ev = NONE_APPLICABLE
            elif lab in ("PARTIAL", "NONE") and ev in ("", NONE_APPLICABLE):
                errs.append(f"{lab} requires an evasion_type")
            for e in errs:
                problems.append((fn, pid, e))
            recs.append({"part_id": pid, "qid": pid.rsplit("#", 1)[0], "coder": coder, "label": lab,
                         "evasion_type": ev, "request_type": rq, "note": note, "sheet_version": ver,
                         "stage": stage, "_ok": not errs})
        frames.append(pd.DataFrame(recs))
    if problems and not allow_incomplete:
        raise ImportProblems(problems)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLS + ["_ok"])
    new = new[new["_ok"]].drop(columns="_ok")
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, keep_default_na=False)
        key = set(zip(new["coder"], new["stage"]))
        old = old[[(c, s) not in key for c, s in zip(old["coder"], old["stage"])]]
        new = pd.concat([old, new], ignore_index=True)
    new = new[LABEL_COLS].sort_values(["stage", "coder", "part_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new.to_csv(out_path, index=False)
    return new, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--stage", choices=["pilot", "main"], default="pilot")
    m.add_argument("--coders", nargs="+", default=["coder1", "coder2"])
    m.add_argument("--root", default=ROOT)
    i = sub.add_parser("import")
    i.add_argument("paths", nargs="*")
    i.add_argument("--stage", choices=["pilot", "main"], default=None)
    i.add_argument("--allow-incomplete", action="store_true")
    i.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        for path, ver, n in make(a.root, a.stage, a.coders):
            print(f"{path}  version={ver}  parts={n}")
        return 0
    d = os.path.join(a.root, "data", "coding")
    paths = a.paths or sorted(os.path.join(d, f) for f in os.listdir(d)
                              if f.startswith("sheet_") and f.endswith(".xlsx")
                              and (a.stage is None or f.endswith(f"_{a.stage}.xlsx")))
    try:
        df, probs = import_sheets(paths, a.root, allow_incomplete=a.allow_incomplete)
    except ImportProblems as e:
        for p in e.problems[:40]:
            print("PROBLEM:", *p)
        print(f"{len(e.problems)} problem(s); nothing written. Fix the sheets or use --allow-incomplete.")
        return 1
    print(f"human_labels.csv: {len(df)} rows; skipped problem rows: {len(probs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
