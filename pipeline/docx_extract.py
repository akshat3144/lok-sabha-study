"""Layout-aware text extraction from Lok Sabha .docx files.

The Lok Sabha Word files are mostly PDF-to-Word conversions: text lives in absolutely positioned
text boxes (wp:anchor -> wps:txbx -> w:txbxContent) that python-docx does not see and whose XML order
is NOT reading order.  We read document.xml directly, split it into pages (section breaks / page
breaks), and per page emit flowing paragraphs first, then anchored text boxes sorted by (y, x) with
boxes on the same visual row merged into one line.  A minority of files are natively authored Word
documents whose "(a) (b) (c)" part labels are automatic list numbering (numbering.xml); those labels
are re-materialised so the text parser sees them.
"""
import re
import zipfile

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
EXTRACT_VERSION = 3  # bump when extraction logic changes (invalidates data/clean/cache)
ROW_TOL = 90000  # EMU; boxes whose top offsets differ by less are on the same visual row


class BadDocx(Exception):
    pass


# ---------------------------------------------------------------- automatic list numbering
def _roman(n, upper=False):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
            (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out.upper() if upper else out


def _fmt(n, fmt):
    if fmt == "lowerLetter":
        return chr(96 + (n - 1) % 26 + 1) * ((n - 1) // 26 + 1)
    if fmt == "upperLetter":
        return chr(64 + (n - 1) % 26 + 1) * ((n - 1) // 26 + 1)
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n, True)
    if fmt == "bullet":
        return ""
    return str(n)  # decimal and everything exotic


class Numbering:
    """Resolve w:numPr to a visible label like '(a)' (best effort)."""

    def __init__(self, num_xml, styles_xml):
        self.abstract, self.nums, self.style_num = {}, {}, {}
        self.counters = {}
        if num_xml is not None:
            root = etree.fromstring(num_xml)
            for an in root.findall(W + "abstractNum"):
                lv = {}
                for l in an.findall(W + "lvl"):
                    il = int(l.get(W + "ilvl"))
                    nf = l.find(W + "numFmt")
                    lt = l.find(W + "lvlText")
                    st = l.find(W + "start")
                    lv[il] = (nf.get(W + "val") if nf is not None else "decimal",
                              lt.get(W + "val") if lt is not None else "%1.",
                              int(st.get(W + "val")) if st is not None else 1)
                self.abstract[an.get(W + "abstractNumId")] = lv
            for n in root.findall(W + "num"):
                aid = n.find(W + "abstractNumId")
                ov = {}
                for o in n.findall(W + "lvlOverride"):
                    so = o.find(W + "startOverride")
                    if so is not None:
                        ov[int(o.get(W + "ilvl"))] = int(so.get(W + "val"))
                self.nums[n.get(W + "numId")] = (aid.get(W + "val") if aid is not None else None, ov)
        if styles_xml is not None and self.nums:
            sroot = etree.fromstring(styles_xml)
            for st in sroot.findall(W + "style"):
                npr = st.find(W + "pPr/" + W + "numPr")
                if npr is not None:
                    nid = npr.find(W + "numId")
                    il = npr.find(W + "ilvl")
                    if nid is not None:
                        self.style_num[st.get(W + "styleId")] = (nid.get(W + "val"), int(il.get(W + "val")) if il is not None else 0)

    def label(self, p):
        if not self.nums:
            return ""
        ppr = p.find(W + "pPr")
        if ppr is None:
            return ""
        npr = ppr.find(W + "numPr")
        numid, ilvl = None, 0
        if npr is not None:
            nid, il = npr.find(W + "numId"), npr.find(W + "ilvl")
            numid = nid.get(W + "val") if nid is not None else None
            ilvl = int(il.get(W + "val")) if il is not None else 0
        else:
            ps = ppr.find(W + "pStyle")
            if ps is not None and ps.get(W + "val") in self.style_num:
                numid, ilvl = self.style_num[ps.get(W + "val")]
        if not numid or numid == "0" or numid not in self.nums:
            return ""
        aid, ov = self.nums[numid]
        lv = self.abstract.get(aid)
        if not lv or ilvl not in lv:
            return ""
        c = self.counters.setdefault(numid, {})
        c[ilvl] = c.get(ilvl, (ov.get(ilvl, lv[ilvl][2]) - 1)) + 1
        for k in [k for k in c if k > ilvl]:
            del c[k]
        fmt, text, _ = lv[ilvl]
        if fmt == "bullet":
            return ""

        def sub(m):
            j = int(m.group(1)) - 1
            f = lv.get(j, ("decimal", "", 1))[0]
            return _fmt(c.get(j, lv.get(j, (0, 0, 1))[2]), f)

        return re.sub(r"%(\d)", sub, text)


# ---------------------------------------------------------------- text helpers
def _para_text(p, numbering=None):
    """Text directly in paragraph p, excluding nested text-box content."""
    parts = []

    def walk(e):
        for c in e:
            t = c.tag
            if t == W + "t":
                parts.append(c.text or "")
            elif t in (W + "tab", W + "noBreakHyphen"):
                parts.append(" " if t == W + "tab" else "-")
            elif t in (W + "br", W + "cr"):
                parts.append("\n")
            elif t in (W + "p", W + "txbxContent", MC + "Fallback", W + "delText", W + "instrText"):
                continue
            else:
                walk(c)

    walk(p)
    txt = "".join(parts)
    if numbering is not None:
        lab = numbering.label(p)
        if lab and txt.strip():
            txt = lab + " " + txt
    return txt


def _box_lines(txbx, numbering):
    out = []
    for p in txbx.iter(W + "p"):
        t = _para_text(p, numbering)
        if t.strip():
            out.append(t)
    return out


def _anchor_info(a):
    pv, ph = a.find(WP + "positionV"), a.find(WP + "positionH")

    def off(e):
        try:
            return int(e.findtext(WP + "posOffset"))
        except Exception:
            return None

    return off(pv), off(ph)


def extract_lines(path):
    """Return a list of text lines (reading order, page by page) and a dict of layout stats."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        with open(path, "rb") as f:
            magic = f.read(8)
        if magic.startswith(b"\xd0\xcf\x11\xe0"):
            raise BadDocx("legacy_ole_doc")
        raise BadDocx("not_a_zip")
    with z:
        names = z.namelist()
        if "word/document.xml" not in names:
            raise BadDocx("no_document_xml")
        root = etree.fromstring(z.read("word/document.xml"))
        n_media = sum(1 for n in names if n.startswith("word/media/"))
        numbering = None
        if "word/numbering.xml" in names:
            numbering = Numbering(z.read("word/numbering.xml"),
                                  z.read("word/styles.xml") if "word/styles.xml" in names else None)
    body = root.find(W + "body")
    # Each page collects "items" = (kind, payload) in XML order; the y position of flowing paragraphs is
    # reconstructed from the paragraph spacing (pgMar top + cumulative spacing-before / exact line heights),
    # because the converter places loose lines (continuations, titles, dates) in flow, not in text boxes.
    pages = [{"items": [], "margin": None}]
    state = {"seq": 0}

    def cur():
        return pages[-1]

    def para_geom(p):
        before = after = 0
        line = 280
        ind = 0
        ppr = p.find(W + "pPr")
        if ppr is not None:
            sp = ppr.find(W + "spacing")
            if sp is not None:
                try:
                    before = int(sp.get(W + "before") or 0)
                    after = int(sp.get(W + "after") or 0)
                    ln = sp.get(W + "line")
                    if ln:
                        line = int(ln) if sp.get(W + "lineRule") in ("exact", "atLeast") else int(int(ln) * 280 / 240)
                except ValueError:
                    pass
            ie = ppr.find(W + "ind")
            if ie is not None:
                try:
                    ind = int(ie.get(W + "left") or ie.get(W + "start") or 0)
                except ValueError:
                    pass
        return before, after, max(line, 1), ind

    def sect_margin(sp):
        pm = sp.find(W + "pgMar")
        if pm is None:
            return None
        try:
            return int(pm.get(W + "top") or 0), int(pm.get(W + "left") or 0)
        except ValueError:
            return None

    def handle_p(p):
        before, after, line, ind = para_geom(p)
        t = _para_text(p, numbering)
        lines = [ln for ln in t.split("\n") if ln.strip()] if t.strip() else []
        state["seq"] += 1
        cur()["items"].append(("flow", (state["seq"], before, after, line, ind, lines)))
        for a in p.iter():
            if a.tag in (WP + "anchor", WP + "inline"):
                if any(x.tag == MC + "Fallback" for x in a.iterancestors()):
                    continue
                y, x = _anchor_info(a) if a.tag == WP + "anchor" else (None, None)
                for tb in a.iter(W + "txbxContent"):
                    ls = _box_lines(tb, numbering)
                    if ls:
                        state["seq"] += 1
                        cur()["items"].append(("box", (y, x, state["seq"], ls)))
        sp = p.find(".//" + W + "sectPr")
        if sp is not None:
            m = sect_margin(sp)
            for pg in reversed(pages):
                if pg["margin"] is not None:
                    break
                pg["margin"] = m or (1440, 1440)
            pages.append({"items": [], "margin": None})
        elif any(b.get(W + "type") == "page" for b in p.iter(W + "br")):
            pages.append({"items": [], "margin": None})

    def walk(e):
        for c in e:
            if c.tag == W + "p":
                handle_p(c)
            elif c.tag == W + "tbl":
                for tr in c.iter(W + "tr"):
                    cells = []
                    for tc in tr.findall(W + "tc"):
                        ct = " ".join(_para_text(p, numbering).strip() for p in tc.iter(W + "p")
                                      if _para_text(p, numbering).strip())
                        cells.append(ct)
                    row = " | ".join(x for x in cells if x)
                    state["seq"] += 1
                    cur()["items"].append(("flow", (state["seq"], 0, 0, 300, 0, [row] if row else [])))
            elif c.tag in (W + "sdt", W + "sdtContent", W + "customXml"):
                walk(c)
            elif c.tag == W + "sectPr":
                m = sect_margin(c)
                for pg in reversed(pages):
                    if pg["margin"] is not None:
                        break
                    pg["margin"] = m or (1440, 1440)

    if body is not None:
        walk(body)
    for pg in pages:
        if pg["margin"] is None:
            pg["margin"] = (1440, 1440)

    EMU = 635  # EMU per twip
    lines = []
    for pg in pages:
        top, left = pg["margin"]
        cursor = top
        entries = []  # (y_emu, x_emu, seq, [lines])
        for kind, payload in pg["items"]:
            if kind == "box":
                entries.append(payload)
                continue
            seq, before, after, line, ind, ls = payload
            cursor += before
            for k, txt in enumerate(ls):
                entries.append(((cursor + k * line) * EMU, (left + ind) * EMU, seq, [txt]))
            cursor += line * max(len(ls), 1) + after
        pos = [b for b in entries if b[0] is not None]
        nopos = [b for b in entries if b[0] is None]
        pos.sort(key=lambda b: (b[0], b[1] if b[1] is not None else 0, b[2]))
        rows, cur_row, row_y = [], [], None
        for b in pos:
            if row_y is not None and abs(b[0] - row_y) <= ROW_TOL:
                cur_row.append(b)
            else:
                if cur_row:
                    rows.append(cur_row)
                cur_row, row_y = [b], b[0]
        if cur_row:
            rows.append(cur_row)
        for r in rows:
            r.sort(key=lambda b: (b[1] if b[1] is not None else 0, b[2]))
            if all(len(b[3]) == 1 for b in r):
                lines.append(" ".join(b[3][0].strip() for b in r))
            else:
                for b in r:
                    lines.extend(b[3])
        for b in sorted(nopos, key=lambda b: b[2]):
            lines.extend(b[3])
    return lines, {"n_pages": len(pages), "n_media": n_media}
