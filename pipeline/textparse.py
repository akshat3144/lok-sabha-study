"""Pure text-parsing helpers for Lok Sabha question/answer text (no file I/O).

Everything here works on plain strings so it can be unit-tested with real snippets.
"""
import html
import re

# --------------------------------------------------------------------------------------
# 1. HTML / whitespace / Hindi cleaning
# --------------------------------------------------------------------------------------
DEVA = re.compile(r"[ऀ-ॿ]")
DEVA_RUN = re.compile(r"[ऀ-ॿ‌‍Ā-ʯ]+")
GARBLED_HINDI_RUN = re.compile(r"\?{3,}")  # the API loses Devanagari to runs of '?'
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

_BLOCK_TAGS = re.compile(r"</?(?:br|p|div|tr|li|ul|ol|h[1-6]|table|tbody|thead)\b[^>]*>", re.I)
_CELL_TAGS = re.compile(r"</t[dh]\s*>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


def _unescape_all(s):
    for _ in range(3):
        n = html.unescape(s)
        if n == s:
            break
        s = n
    return s


_TYPO = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"',
         0xF0B7: "•", 0xF0A7: "•", 0xF0D8: ">", 0xF020: " ", 0x2022: "•"}


def html_to_text(s):
    """HTML or escaped HTML (possibly double escaped) -> plain text with newlines for block tags."""
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # escaped markup -> real markup, but protect a stray literal '&' (e.g. "Commerce & Industry")
    if "&lt;" in s or "&gt;" in s or "&amp;lt;" in s:
        s = _unescape_all(s)
    if _BLOCK_TAGS.search(s):
        # in HTML source, raw CR/LF are only hard-wrap formatting (LS16 answers wrap at ~85 chars);
        # real line breaks are <br>/<p> tags
        s = re.sub(r"\s*\n\s*", " ", s)
    s = _CELL_TAGS.sub(" | ", s)
    s = _BLOCK_TAGS.sub("\n", s)
    s = _ANY_TAG.sub("", s)
    s = html.unescape(s)
    return s


def normalize_line(l):
    l = l.replace(" ", " ").replace(" ", " ").replace(" ", " ").translate(_ZW)
    l = l.translate(_TYPO)
    l = l.replace("\t", " ")
    # glyph-mapped Hindi fonts leave isolated Latin-Extended letters ('Ʞ Ũ') behind
    l = re.sub(r"(?:(?<=\s)|^)[Ā-ʯ]{1,3}(?=\s|$)", "", l)
    l = re.sub(r"[ ]{2,}", " ", l)
    l = re.sub(r"(?<=\d) (?=(?:st|nd|rd|th)\b)", "", l)  # superscript ordinals arrive as separate boxes: '1 st'
    l = re.sub(r"\s*\|\s*$", "", l)  # trailing cell separator
    return l.strip()


def is_hindi_line(l):
    """Mostly Devanagari (docx / PDF)."""
    letters = sum(1 for c in l if c.isalpha())
    dev = len(DEVA.findall(l))
    return bool(dev and dev >= 3 and dev / max(letters, 1) > 0.5)


_GARBLE = re.compile(r"[?�][?�\s]*[?�][?�\s]*[?�][?�\s]*")


def strip_hindi(lines):
    """Remove Hindi lines; return (kept_lines, had_hindi).

    Devanagari-majority lines are dropped.  The API turns Devanagari into runs of '?', sometimes on the
    same line as English (e.g. '???? ??? ( SHRI X )'), so garbled runs are cut out of the line and the
    rest is kept."""
    out, had = [], False
    for l in lines:
        if DEVA.search(l):
            had = True
            # drop Devanagari (incl. stray combining marks); keep any English left on the line
            l = normalize_line(DEVA_RUN.sub(" ", l))
            l = re.sub(r"\(\s*\)|\[\s*\]", "", l).strip()
            if len(re.findall(r"[A-Za-z]", l)) < 2 and not re.search(r"\d", l):
                continue
        if GARBLED_HINDI_RUN.search(l) or re.search(r"[?�]\s[?�]\s[?�]|��", l):
            m = _GARBLE.search(l)
            if m:
                had = True
                l = normalize_line(_GARBLE.sub(" ", l))
                l = re.sub(r"\(\s*\)|\[\s*\]", "", l).strip()
                if not re.search(r"[A-Za-z0-9]{2}", l):
                    continue
        out.append(l)
    return out, had


# page furniture of the converted docx / PDFs
_JUNK_LINE = [
    re.compile(r"^[-–]{1,3}\s*\d{1,3}\s*[-–]{1,3}$"),  # '-3-', '--2--'
    re.compile(r"^page\s*\d+\s*(of\s*\d+)?$", re.I),
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^L\.?\s?S\.?\s?(?:U\.?\s?S\.?|S\.?|UNS\.?)?\s?Q\.?\s?NO\.?\s*[\d*]+.*$", re.I),
    re.compile(r"^\(?\s*contd\b[^\n]{0,12}$", re.I),
    re.compile(r"^[-_=~.]{3,}$"),
    re.compile(r"^[.…]{2,}\s*\d{1,2}\s*/?-?$"),  # '…..2/' continuation marker
    re.compile(r"^[-–—]$"),
]


def is_junk_line(l, strip_bare_numbers=True):
    for i, p in enumerate(_JUNK_LINE):
        if i == 2 and not strip_bare_numbers:
            continue
        if p.match(l):
            return True
    return False


_END_PUNCT = re.compile(r"[.?!:;\"”]$")
_STARTS_NEW = re.compile(
    r"^(?:\(\s*(?:[a-z]|[ivx]{1,4}|\d{1,2})\s*\)|[a-z]\)|\d{1,2}[.)]\s|\d{1,2}\.\d|[•●→*>-]\s|annexure|statement|note\b|\(?[ivx]{1,4}\)|part\b)",
    re.I,
)


def reflow(text):
    """Re-join hard-wrapped lines (docx/PDF) into paragraphs.  Conservative."""
    lines = text.split("\n")
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        if out and out[-1] and not _END_PUNCT.search(out[-1]) and not _STARTS_NEW.match(ln) and not (
            ln.isupper() and len(ln) > 3
        ) and not out[-1].isupper():
            sep = "" if out[-1].endswith("-") and not out[-1].endswith(" -") and len(out[-1]) > 1 and out[-1][-2].isalpha() and ln[:1].islower() else " "
            out[-1] = out[-1] + sep + ln
        else:
            out.append(ln)
    return "\n".join(l for l in out if l != "" or True).strip()


def clean_text(raw_html=None, lines=None, drop_hindi=True, wrap=False, strip_numbers=True):
    """Return (clean_text, had_hindi).  Provide raw_html (API) or lines (docx/pdf)."""
    if lines is None:
        lines = html_to_text(raw_html or "").split("\n")
    lines = [normalize_line(l) for l in lines]
    lines = [l for l in lines if l and not is_junk_line(l, strip_numbers)]
    had = False
    if drop_hindi:
        lines, had = strip_hindi(lines)
    text = "\n".join(lines)
    if wrap:
        text = reflow(text)
    return text.strip(), had


# --------------------------------------------------------------------------------------
# 2. Part (a)/(b)/... splitter
# --------------------------------------------------------------------------------------
_TOK = r"(?:[a-z]|[ivx]{2,4}|\d{1,2})"
_ONE = rf"\(\s*({_TOK})\s*\)"
_SEP = r"(?:\s*(?:,|&|\band\b|\bto\b|\bor\b|-|–|—|/)\s*|\s+)"
_GRP = re.compile(
    rf"(?:\b(?:parts?|paras?|clauses?)\s*)?(?P<g>{_ONE}(?:{_SEP}{_ONE})*)(?P<tail>\s*[:.\-–—]*)",
    re.I,
)
# 'a)' style (no opening paren) -- only honoured at line start
_GRP_NOPAREN = re.compile(
    r"(?:^|(?<=[;:?]\s)|(?<=\band\s))(?P<t>[a-z]|\d{1,2})(?P<d>[).])\s+(?=[A-Za-z(\"'])", re.I | re.M)

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
          "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15}
_ROMAN_REV = {v: k for k, v in _ROMAN.items()}
AMBIG = {"i", "v", "x"}


def _tok_kind(t):
    t = t.lower()
    if t.isdigit():
        return "num"
    if len(t) == 1 and t not in AMBIG:
        return "alpha"
    if len(t) == 1:
        return "ambig"
    return "roman"


def _idx(t, scheme):
    t = t.lower()
    if scheme == "num":
        return int(t) if t.isdigit() else None
    if scheme == "alpha":
        return ord(t) - 96 if len(t) == 1 and t.isalpha() else None
    if scheme == "roman":
        return _ROMAN.get(t)
    return None


def _label(i, scheme):
    if scheme == "num":
        return str(i)
    if scheme == "alpha":
        return chr(96 + i) if 1 <= i <= 26 else None
    return _ROMAN_REV.get(i)


def label_scheme(label):
    """'alpha' / 'roman' / 'num' for an already-parsed label (single letters are alpha)."""
    if label.isdigit():
        return "num"
    return "roman" if len(label) > 1 else "alpha"


def _expand(g, scheme):
    """Expand a label-group string (e.g. '(a) to (c)', '(a) & (b)') to the list of labels."""
    toks = re.findall(_ONE, g, flags=re.I)
    seps = re.split(_ONE, g, flags=re.I)[2::2]  # separators between tokens
    idxs = [_idx(t, scheme) for t in toks]
    if any(i is None for i in idxs):
        return None
    out = [idxs[0]]
    for k in range(1, len(idxs)):
        sep = seps[k - 1].strip().lower() if k - 1 < len(seps) else ","
        if sep in ("to", "-", "–", "—") and idxs[k] > idxs[k - 1]:
            out.extend(range(idxs[k - 1] + 1, idxs[k] + 1))
        else:
            out.append(idxs[k])
    labs = [_label(i, scheme) for i in out]
    return None if any(l is None for l in labs) else labs


def _boundary_ok(text, start):
    """A label group counts only at line start or right after sentence/clause punctuation or 'and'."""
    pre = text[:start]
    stripped = pre.rstrip(" \t")
    if not stripped or stripped.endswith("\n"):
        return True
    ch = stripped[-1]
    if ch in ";:.?!,”\"'":
        return True
    low = stripped.lower()
    if low.endswith(" and") or low.endswith(" or") or low.endswith("&"):
        return True
    return False


def find_parts(text, mode="answer", scheme_hint=None, allowed=None):
    """Locate part markers.  Returns list of dicts {labels, scheme, start, end} in text order.

    mode='question': first marker must start the sequence (a / i / 1).
    mode='answer'  : first marker may be any label (answers can start at (c)), but subsequent markers
                     must be strictly increasing in the same scheme (gap <= 3), which rejects nested
                     (i)/(ii) sub-lists and '(a)' repeats inside a statement.
    """
    cands = []
    for m in _GRP.finditer(text):
        if not _boundary_ok(text, m.start()):
            continue
        cands.append((m.start(), m.end(), m.group("g")))
    # 'a)' markers at line start
    for m in _GRP_NOPAREN.finditer(text):
        t = m.group("t")
        if t.isdigit() and m.group("d") == ".":
            continue  # '1.' belongs to enumerations inside an answer, never a part label
        inline = m.start() > 0 and text[m.start() - 1] != "\n"
        if inline and not t.isalpha():
            continue
        cands.append((m.start(), m.end() - 1, "(" + t + ")"))
    cands.sort()

    def run(strict_start):
        parts, last, scheme, seen = [], None, scheme_hint, set()
        for s, e, g in cands:
            toks = re.findall(_ONE, g, flags=re.I)
            if not toks:
                continue
            kind0 = _tok_kind(toks[0])
            if scheme is None:
                if kind0 == "alpha":
                    sch = "alpha"
                elif kind0 == "num":
                    sch = "num"
                elif kind0 in ("roman", "ambig"):
                    sch = "roman"
                else:
                    continue
                if strict_start and _idx(toks[0], sch) != 1:
                    continue
            else:
                sch = scheme
                # token kinds must be compatible with the scheme
                ok = True
                for t in toks:
                    k = _tok_kind(t)
                    if sch == "alpha" and k in ("roman", "num"):
                        ok = False
                    if sch == "roman" and k in ("alpha", "num"):
                        ok = False
                    if sch == "num" and k != "num":
                        ok = False
                if not ok:
                    continue
            labs = _expand(g, sch)
            if labs and allowed:
                labs = [l for l in labs if l in allowed]  # an answer cannot address parts the question lacks
            if not labs:
                continue
            first = _idx(labs[0], sch)
            if mode == "question" and last is not None and first == last and len(labs) == 1:
                # typo in the source: "(a) ...; (a) ..." -> second one is really (b)
                first = last + 1
                labs = [_label(first, sch)]
                if labs[0] is None:
                    continue
            if last is not None:
                in_order = last < first <= last + 3
                # out-of-order but brand-new labels that the question really has: '(a) & (d): ...' then
                # '(b) & (c): ...' (answers do group non-adjacent parts)
                out_of_order = (allowed is not None and not (set(labs) & seen) and set(labs) <= allowed)
                if not (in_order or out_of_order):
                    continue
            elif strict_start and first != 1:
                continue
            if scheme is None:
                scheme = sch
            last = _idx(labs[-1], sch) if last is None or _idx(labs[-1], sch) > last else last
            seen.update(labs)
            parts.append({"labels": labs, "scheme": sch, "start": s, "end": e})
        return parts

    if mode == "question":
        return run(True)
    p = run(True)
    if not p:
        p = run(False)
    return p


_COMPACT = re.compile(r"(?m)^(\s*)\(\s*([a-z])\s*(&|,|and|to|-)\s*([a-z])\s*\)", re.I)


def split_parts(text, mode="answer", scheme_hint=None, allowed=None):
    """Split text into (preamble, parts).  parts: list of {label, labels, grouped, text}.

    allowed: labels the question actually has (lets an answer use out-of-order groups)."""
    text = _COMPACT.sub(lambda m: f"{m.group(1)}({m.group(2)}) {m.group(3)} ({m.group(4)})", text or "")
    marks = find_parts(text, mode, scheme_hint, set(allowed) if allowed else None)
    if not marks:
        return text.strip(), []
    pre = text[: marks[0]["start"]].strip()
    parts = []
    for i, m in enumerate(marks):
        end = marks[i + 1]["start"] if i + 1 < len(marks) else len(text)
        body = text[m["end"]:end].strip()
        parts.append({"label": m["labels"][0], "labels": m["labels"], "grouped": len(m["labels"]) > 1,
                      "text": body})
    return pre, parts


# --------------------------------------------------------------------------------------
# 3. Question preamble, minister header, statement-laid
# --------------------------------------------------------------------------------------
_PLEASED = re.compile(r"Will\s+the\s+.{0,250}?be\s+pleased\s+to\s+(?:state|inform|reply)\s*:?", re.I | re.S)


def strip_question_preamble(text):
    """Remove 'Will the Minister of X be pleased to state:' (+ any member/number line before it)."""
    m = _PLEASED.search(text[:600])
    if m:
        return text[m.end():].strip(), True
    return text, False


_HONORIFIC = re.compile(
    r"^(?:(?:SHRIMATI|SHRI|SMT|SUSHRI|KUMARI|KU|SARDAR|DR|PROF|ADV|CAPT|COL|GEN|MAJ|LT|RAJA|MAHARAJA|SWAMI|"
    r"ACHARYA|JUSTICE|BABU|THIRU|THIRUMATI|MS|MR|MRS|PANDIT|PT|SHREE|SHRIMANT|CH)\b\.?\s*)+", re.I)
_SMALL = {"of", "and", "the", "in", "for", "to", "&", "on", "with", "or", "ministry", "state"}


def _title_name(s):
    out = []
    for tok in re.split(r"(\s+)", s.strip()):
        if not tok.strip():
            out.append(" ")
            continue
        if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?\.?", tok):
            out.append(tok.upper())
        else:
            out.append("-".join(p.capitalize() for p in tok.lower().split("-")))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def title_case_title(s):
    words = re.sub(r"\s+", " ", s.strip()).split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in ("of", "and", "the", "in", "for", "to", "on", "&"):
            out.append(lw if lw != "&" else "&")
        elif re.fullmatch(r"\(?(ic|i/c)\)?", lw):
            out.append(w.upper())
        else:  # capitalise each alphabetic run, also after '(' etc.
            out.append(re.sub(r"[A-Za-z][A-Za-z']*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), w))
    return " ".join(out)


_LABELISH = re.compile(r"^\s*(?:[a-z]|[ivx]{2,4}|\d{1,2})\s*$", re.I)
_CHARGE = re.compile(r"independent\s+charge|^\s*(?:ic|i/c)\s*$", re.I)


_BRACKETS = str.maketrans("[]{}", "()()")
_MINWORD = r"MIN[IS]{0,2}TE?R"


def _looks_like_name(inner):
    """A person's name inside the header parentheses (not an acronym such as 'PSBs')."""
    inner = inner.strip()
    if _HONORIFIC.match(inner):
        return True
    words = [w for w in re.split(r"[\s.]+", inner) if w]
    return len(words) >= 2 and all(w[0].isupper() for w in words if w.isalpha()) and len(inner) <= 60


def parse_minister_header(text):
    """Parse the header at the start of an answer.

    Returns (info, rest) where info has minister_title, minister_name, minister_name_raw, minister_rank
    (all None when absent) and rest is the answer text with the header removed.
    """
    info = {"minister_title": None, "minister_name": None, "minister_name_raw": None, "minister_rank": None}
    t = text.lstrip()
    t = re.sub(r"^ANSWERS?\b\s*[:\-]?\s*", "", t, flags=re.I)
    head = t[:600].translate(_BRACKETS)  # some headers use [SHRI X] or {SHRI X}
    mm = re.search(_MINWORD + r"|\bMINISTRY\b", head[:200], re.I)
    if not mm:
        # name-only header: "(SHRI RATTAN LAL KATARIA)"
        nm0 = re.match(r"^\(\s*((?:SHRIMATI|SHRI|SMT|SUSHRI|DR|PROF|KUMARI|COL|GEN|CAPT|LT|MAJ|ADV)\b(?:[^()]|\([^()]{0,20}\)){2,80})\)",
                       head, re.I)
        if nm0:
            info["minister_name_raw"] = re.sub(r"\s+", " ", nm0.group(1)).strip()
            nm_clean = re.sub(r"\s*\([^()]*\)", "", nm0.group(1)).strip() or nm0.group(1)  # drop '(Retd.)'
            info["minister_name"] = _title_name(_HONORIFIC.sub("", nm_clean).strip() or nm_clean)
            return info, t[nm0.end():].lstrip(" \n:.-")
        nm1 = re.match(r"^((?:SHRIMATI|SHRI|SMT|SUSHRI|DR|PROF|KUMARI)\b\.?[^()\n]{2,60})\n(?=\s*\(?\s*(?:[a-z]|[ivx]{2,4})\s*\)|\s*(?:[A-Z]|\())", t, re.I)
        if nm1 and not re.search(r"[:;?]$", nm1.group(1)):
            info["minister_name_raw"] = re.sub(r"\s+", " ", nm1.group(1)).strip()
            info["minister_name"] = _title_name(_HONORIFIC.sub("", nm1.group(1)).strip() or nm1.group(1))
            return info, t[nm1.end():].strip()
        return info, t.strip()
    name = None
    end = None
    for m in re.finditer(r"\(((?:[^()]|\([^()]{0,20}\)){2,90})\)", head):  # one nesting level: (COL X (Retd.))
        inner = m.group(1).strip()
        before = head[: m.start()]
        if _CHARGE.search(inner):
            continue
        if _LABELISH.match(inner):
            break  # a part label before any name: the header has no name paren
        has_title = bool(re.search(_MINWORD, before, re.I))
        if not has_title and not (_HONORIFIC.match(inner) and len(before) <= 120):
            break
        if (len(before) > 330 or re.search(r"[a-z]\.\s+[A-Z]", before) or before.count("\n") > 4
                or re.search(r"\(\s*(?:[a-z]|[ivx]{2,4}|\d{1,2})\s*\)", before, re.I)):
            break
        if not _looks_like_name(inner):
            continue
        name, end, name_start = inner, m.end(), m.start()
        break
    if name is not None:
        title_raw = head[:name_start].strip()
        info["minister_name_raw"] = re.sub(r"\s+", " ", name)
        name = re.sub(r"\s*\([^()]*\)", "", name).strip() or name  # drop '(Retd.)'
        nm = _HONORIFIC.sub("", name).strip()
        info["minister_name"] = _title_name(nm or name)
        rest = t[end:].lstrip(" \n:.-")
        # 'THE MINISTER OF STATE IN THE (SHRI X)' + next line 'MINISTRY OF STEEL'
        m2 = re.match(r"^((?:MINISTRY|DEPARTMENT)\b[^\n]{0,80})(?:\n|$)", rest)
        if m2 and re.search(r"\b(?:in|of|for)(?:\s+the)?$", title_raw, re.I):
            title_raw = title_raw + " " + m2.group(1).strip()
            rest = rest[m2.end():].lstrip()
    else:
        # title only: leading upper-case lines / a short 'The Finance Minister' line
        ls = t.split("\n")
        k = 0
        buf = []
        while k < len(ls) and k < 4 and not _STARTS_NEW.match(ls[k]) and (
                ls[k].isupper() or re.match(r"^(THE\s+)?" + _MINWORD + r"\b", ls[k], re.I)
                or (k == 0 and len(ls[k]) <= 90 and re.search(r"\bminister\b", ls[k], re.I)
                    and not re.search(r"[.?:;,]$", ls[k]))):
            buf.append(ls[k])
            k += 1
        if not buf or any(re.search(r"GOVERNMENT OF INDIA|LOK SABHA|QUESTION NO|ANSWERED ON", b, re.I) for b in buf):
            return info, t.strip()
        title_raw = " ".join(buf)
        ms = re.search(r"\s(?:SHRIMATI|SHRI|SMT\.?|SUSHRI|DR\.?|PROF\.?)\s+[A-Za-z]", title_raw, re.I)
        if ms and ms.start() > 8:  # 'MINISTER OF FINANCE SHRI ARUN JAITLEY' on one line
            info["minister_name_raw"] = title_raw[ms.start():].strip()
            info["minister_name"] = _title_name(_HONORIFIC.sub("", info["minister_name_raw"]).strip())
            title_raw = title_raw[: ms.start()]
        # a bare 'Shri Ashwini Kumar Choubey' line right after the title
        if k < len(ls) and _HONORIFIC.match(ls[k]) and len(ls[k]) <= 70 and not re.search(r"[:;?.]$", ls[k]) \
                and not _STARTS_NEW.match(ls[k]):
            info["minister_name_raw"] = re.sub(r"\s+", " ", ls[k]).strip()
            info["minister_name"] = _title_name(_HONORIFIC.sub("", ls[k]).strip() or ls[k])
            k += 1
        rest = "\n".join(ls[k:]).strip()
    title_raw = re.sub(r"^(THE\s+)", "", re.sub(r"\s+", " ", title_raw).strip(" ,-"), flags=re.I)
    title_raw = re.sub(r"(?<=[A-Za-z])(?=\()", " ", title_raw)
    title_raw = re.sub(r"\b(?:MINSTER|MINISTR|MINISER|MINISTE|MINITER)\b", "MINISTER", title_raw,
                       flags=re.I)  # typos seen in the source
    title_raw = re.sub(r"\b(?:MINSTRY|MINISTY|MINISRTY|MINITRY|MINSITRY|MINISTERY)\b", "MINISTRY", title_raw, flags=re.I)
    title_raw = title_raw.strip(" ,;:.-")
    info["minister_title"] = title_case_title(title_raw) if title_raw else None
    tl = title_raw.lower()
    if re.search(r"prime\s+minister", tl):
        info["minister_rank"] = "pm"
    elif re.search(r"minister\s+of\s+state.*(independent|\(ic\)|\(i/c\))|\(ic\)|independent charge", tl):
        info["minister_rank"] = "mos_ic"
    elif re.search(r"minister\s+of\s+state", tl) or re.search(r"\bmos\b", tl):
        info["minister_rank"] = "mos"
    elif re.search(_MINWORD, tl, re.I):
        info["minister_rank"] = "cabinet"
    return info, rest.strip()


STATEMENT_LAID = re.compile(
    r"statement\s*(?:of\s+(?:the\s+)?reply\s*)?(?:is|was|has\s+been)?\s*(?:being\s+)?"
    r"(?:(?:laid|placed)\s+(?:down\s+)?(?:on|upon)\s+the\s+table|enclosed|attached|annexed)",
    re.I)
_STATEMENT_HEAD = re.compile(
    r"^\s*(?:the\s+)?stat\w{0,3}ment\s+(?:as\s+)?(?:ref\w+|for|in\s+reply|in\s+respect|on|relating|regarding|to\s+be)\b", re.I)
_LAY_SENTENCE = re.compile(r"[^\n]*?\bstatement\s*(?:is|was|has\s+been)?\s*(?:being\s+)?laid\s+on\s+the\s+table[^\n.]*\.?", re.I)


def _starts_with_label(l):
    return bool(re.match(r"^\s*(?:parts?\s*)?\(\s*(?:[a-z]|[ivx]{2,4}|\d{1,2})\s*\)", l, re.I))


def _mostly_upper(l):
    letters = [c for c in l if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.7


_HEAD_JUNK = re.compile(
    r"^(?:[\[\]()*.\-\s]*|position\s+no\.?\s*\d+|\(?\d+(?:st|nd|rd|th)\s+position\)?)$", re.I)
_HEAD_END = re.compile(r"(?:19|20)\d\d\s*\.?$|[\"”]\s*\.?$|:\s*$")


def strip_statement_heading(text):
    """Remove the 'STATEMENT REFERRED TO IN REPLY TO ... REGARDING "TITLE"' heading (1-6 lines).

    Variants seen: 'STATEMENT REFERRED TO IN REPLY TO', 'STATEMENT FOR LOK SABHA STARRED QUESTION',
    'Statement in reply to Lok Sabha ...', 'Statement referred to in reply to ...'."""
    ls = text.split("\n")
    i = 0
    while i < len(ls) and (not ls[i].strip() or _HEAD_JUNK.match(ls[i].strip())):
        i += 1
    if i >= len(ls) or not _STATEMENT_HEAD.match(ls[i]) or not re.search(
            r"question|reply|lok\s*sabha", " ".join(ls[i:i + 3]), re.I):
        return text, False
    end = i
    for j in range(i, min(len(ls), i + 6)):
        if j > i and _starts_with_label(ls[j]):
            break
        end = j
        l = ls[j].rstrip()
        if _HEAD_END.search(l):
            break
        if j + 1 < len(ls):
            nxt = ls[j + 1]
            if _starts_with_label(nxt) or (_mostly_upper(ls[i]) and not _mostly_upper(nxt)):
                break
    rest = ls[end + 1:]
    if len(ls[end]) > 200:
        # heading and body flattened onto one long line:  '... asked by Shri X.  (a) to (d) Grant of ...'
        m = re.search(r"(?<=[.\"”])\s+(?=\(\s*[a-z]\s*\)|[a-z]\)\s)", ls[end][60:], re.I)
        if m:
            rest = [ls[end][60 + m.end():]] + rest
    return "\n".join(rest).strip(), True


def split_statement(text):
    """If answer text says 'A statement is laid on the Table', return
    (laid_labels, statement_body) else (None, text)."""
    head = text[:400]
    m = STATEMENT_LAID.search(head)
    if not m:
        # header lost / no lay sentence, but the text opens with the 'STATEMENT REFERRED TO ...' heading
        body, had = strip_statement_heading(re.sub(r"^\s*stat\w{0,3}ment\s*\n", "", text, flags=re.I))
        if had:
            hm = re.search(r"parts?\s*(\(.*?\)(?:\s*(?:to|and|&|,|-)\s*\(.*?\))*)", text[:300], re.I)
            marks = find_parts(hm.group(1) + " ", "answer") if hm else []
            return [l for mk in marks for l in mk["labels"]], body.strip()
        return None, text
    # labels preceding the sentence: '(a) to (f): A statement ...'
    pre = text[: m.start()]
    # 'enclosed/attached/annexed' (as opposed to 'laid on the table') after a substantive answer is just a
    # reference ("... is given in the statement enclosed"), not a lay sentence: keep the whole answer
    if not re.search(r"laid|placed", m.group(0), re.I) and len(re.sub(r"[\W_]+", "", pre)) >= 40:
        return None, text
    marks = find_parts(pre + " ", "answer")
    labels = [l for mk in marks for l in mk["labels"]]
    after = text[m.end():]
    after = re.sub(r"^\s*(?:of|on|in)?\s*(?:the\s+)?(?:table\s+of\s+)?(?:the\s+)?(?:house|lok\s*sabha|sabha)\b\.?", "",
                   after, count=1, flags=re.I)
    after = re.sub(r"^[\s.*]+", "", after)
    after = re.sub(r"^\s*stat\w{0,3}ment\s*\n", "", after, flags=re.I)  # lone 'STATEMENT' / 'STATMENT' line
    body, _ = strip_statement_heading(after)
    return labels, body.strip()


# --------------------------------------------------------------------------------------
# 4. Boilerplate patterns / annexure
# --------------------------------------------------------------------------------------
BOILERPLATE = {
    "info_being_collected": r"(?:information|data|details|reply)\b[^.]{0,80}\b(?:is|are|being)\b[^.]{0,30}\bcollect(?:ed|ing)\b|being collected and will be laid",
    "does_not_arise": r"\bdo(?:es)?\s+not\s+arise\b",
    "no_such_proposal": r"\bno\s+such\s+(?:proposal|scheme|plan|provision|case|instance|incident|report)s?\b|\bthere\s+is\s+no\s+(?:such\s+)?proposal\b|\bno\s+proposal\b",
    "not_maintained_centrally": r"\bnot\s+(?:being\s+)?(?:maintained|available|compiled|collected|kept)\s+(?:centrally|at\s+(?:the\s+)?(?:central|national)\b)|\b(?:data|information|record\w*)\b[^.]{0,60}\bnot\s+(?:being\s+)?(?:maintained|compiled)\b",
    "under_consideration": r"\bunder\s+(?:active\s+)?(?:consideration|examination|process|review|finali[sz]ation)\b|\bbeing\s+(?:examined|considered)\b",
    "state_subject": r"\bstate\s+subject\b|\bsubject\s+matter\s+of\s+(?:the\s+)?states?\b|\bresponsibility\s+of\s+(?:the\s+)?state\s+government",
    "not_applicable": r"\bnot\s+applicable\b|\bnot\s+relevant\b",
    "sub_judice": r"\bsub[\s-]?judice\b",
    "public_interest_security": r"\bnot\s+in\s+(?:the\s+)?public\s+interest\b|\bsecurity\s+reasons\b|\bnational\s+security\b[^.]{0,40}\bnot\s+(?:be\s+)?(?:disclosed|shared)",
    "refers_elsewhere": r"\bmay\s+be\s+(?:addressed|referred)\s+to\b|\bconcerned\s+ministry\b|\bpertains\s+to\s+(?:the\s+)?ministry\b|\btransferred\s+to\b",
    "no_information": r"\bno\s+(?:such\s+)?(?:information|data|record)\b[^.]{0,40}\b(?:available|maintained|received|reported)\b|\bdoes\s+not\s+(?:maintain|collect|keep)\b",
}
_BP_RE = {k: re.compile(v, re.I) for k, v in BOILERPLATE.items()}


def boilerplate_flags(text):
    return [k for k, rx in _BP_RE.items() if rx.search(text or "")]


_END_STARS = re.compile(r"[ \t]*(?:\*[ \t]*){3,}$", re.M)  # '*****', '* *******' or trailing stars on a text line
_ANNEX_HEAD = re.compile(
    r"^\s*(?:annexure|annexures|annex|appendix|enclosure)\b(?:[ \-–:.]*(?:[ivx]{1,4}|[a-z]|\d{1,2}))?\b[^\n]{0,60}$|"
    r"^\s*(?:annexure|appendix)\s+referred\s+to\b[^\n]*$",
    re.I | re.M,
)


def split_annexure(text):
    """Split main answer text from trailing '*****' / ANNEXURE material.
    Returns (main, annexure_text, annexure_referenced)."""
    cuts = []
    for m in _END_STARS.finditer(text):
        # a '*****' line right after the minister header precedes the answer; only a star line that
        # follows real answer text ends it
        if len(re.sub(r"[\s*]", "", text[: m.start()])) >= 25:
            cuts.append(m.start())
            break
    for m in _ANNEX_HEAD.finditer(text):
        line = m.group(0).strip()
        # a heading is a short line, not prose like "Annexure-I gives the list of..."
        referred = re.match(r"(?i)(annexure|appendix)\s+referred", line)
        if referred or (len(line) <= 70 and not re.search(r"[.;,]$", line)):
            if m.start() > 0:
                cuts.append(m.start())
            else:
                cuts.append(0)
            break
    referenced = bool(re.search(r"\bannexure\b|\bappendix\b|\bannex\b", text, re.I))
    if not cuts:
        return _END_STARS.sub("", text).strip(), "", referenced
    c = min(cuts)
    tail = text[c:]
    tail = _END_STARS.sub("", tail).strip()
    return _END_STARS.sub("", text[:c]).strip(), tail, referenced


def repair_truncated_start(text):
    """The API sometimes chops the first characters of answerText (header and the opening '(a'),
    leaving '): No, Sir.' or ') to (c): ...' or 'to (c): ...'.  Rebuild a leading '(a)' marker.
    Returns (text, repaired_bool)."""
    t = text.lstrip()
    m = re.match(r"^\)\s*(?:(?P<sep>to|and|&)\s*\(\s*(?P<last>[a-z])\s*\))?\s*[:.\-]?", t, re.I)
    if m:
        lab = "(a)" + (f" {m.group('sep')} ({m.group('last')})" if m.group("last") else "")
        return lab + ": " + t[m.end():].lstrip(), True
    m = re.match(r"^(?P<sep>to|and|&)\s*\(\s*(?P<last>[a-z])\s*\)\s*[:.\-]?", t, re.I)
    if m:
        return f"(a) {m.group('sep')} ({m.group('last')}): " + t[m.end():].lstrip(), True
    m = re.match(r"^[:.]\s+(?=\S)", t)
    if m:
        return "(a): " + t[m.end():], True
    return text, False
