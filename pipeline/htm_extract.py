"""Robust HTML -> text lines for the legacy .htm statement/annexure files (data/raw/docs_fallback).

The files are 2001-2017 era exports: Word/FrontPage "Save as web page" output (cp1252 / iso-8859-1 /
utf-8 with BOM, hard-wrapped source lines, <o:p> tags, conditional comments, huge inline styles) and,
for LS13, scanned-image OCR dumps.  We decode with charset detection, drop <head>/<script>/<style>/
comments, keep block structure as line breaks and table rows as ' | ' separated text lines.
"""
import re
from html.parser import HTMLParser

EXTRACT_VERSION = 2

_META_CHARSET = re.compile(rb"charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-:.]+)", re.I)
_ALIAS = {"iso-8859-1": "cp1252", "latin1": "cp1252", "latin-1": "cp1252", "us-ascii": "cp1252",
          "windows-1252": "cp1252", "ascii": "cp1252", "utf8": "utf-8", "unicode": "utf-16",
          "x-unknown": "cp1252"}


def decode_html(b):
    """bytes -> (text, encoding_used).  BOM > valid utf-8 > declared charset > cp1252."""
    if b.startswith(b"\xef\xbb\xbf"):
        return b[3:].decode("utf-8", "replace"), "utf-8-sig"
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace"), "utf-16"
    try:  # strictly valid UTF-8 (multi-byte sequences are essentially never valid by accident)
        return b.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    m = _META_CHARSET.search(b[:4096])
    enc = m.group(1).decode("ascii", "ignore").lower() if m else "cp1252"
    enc = _ALIAS.get(enc, enc)
    try:
        import codecs
        codecs.lookup(enc)
    except LookupError:
        enc = "cp1252"
    if enc == "utf-8":  # declared utf-8 but not valid -> it is really cp1252
        enc = "cp1252"
    return b.decode(enc, "replace"), enc


_BLOCK = {"p", "div", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "tbody", "thead",
          "tfoot", "blockquote", "center", "pre", "hr", "br", "dd", "dt", "dl", "form", "address"}
_SKIP = {"script", "style", "head", "title", "xml", "noscript", "object", "applet"}
NL = chr(10)


class _P(HTMLParser):
    """Block-aware text collector.  A table row with >= 2 non-empty cells becomes ONE ' | ' joined line
    (cell paragraphs are merged); a one-cell row (layout wrapper, e.g. FrontPage page tables) keeps its
    internal line breaks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = []
        self.frames = []  # ["row", depth, cells] / ["cell", depth, buf]
        self.depth = 0
        self.skip = 0
        self.pre = 0

    @property
    def buf(self):
        return self.frames[-1][2] if self.frames and self.frames[-1][0] == "cell" else self.root

    def _close_cell(self):
        f = self.frames.pop()
        self.frames[-1][2].append("".join(f[2]))

    def _close_row(self):
        f = self.frames.pop()
        cells = [c for c in f[2] if c.strip()]
        if len(cells) >= 2:
            txt = " | ".join(re.sub(r"\s+", " ", c).strip() for c in cells)
        else:
            txt = NL.join(cells)
        self.buf.append(NL + txt + NL)

    def _top(self, kind):
        return bool(self.frames) and self.frames[-1][0] == kind and self.frames[-1][1] == self.depth

    def _close_open(self):
        while self._top("cell") or self._top("row"):
            if self._top("cell"):
                self._close_cell()
            else:
                self._close_row()

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
            return
        if tag == "table":
            self.depth += 1
        elif tag == "tr":
            self._close_open()
            self.frames.append(["row", self.depth, []])
        elif tag in ("td", "th"):
            if self._top("cell"):
                self._close_cell()
            if not self._top("row"):
                self.frames.append(["row", self.depth, []])
            self.frames.append(["cell", self.depth, []])
        elif tag == "pre":
            self.pre += 1
        if tag in _BLOCK:
            self.buf.append(NL)

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK and tag not in _SKIP:
            self.buf.append(NL)

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag in ("td", "th"):
            if self._top("cell"):
                self._close_cell()
        elif tag == "tr":
            if self._top("cell"):
                self._close_cell()
            if self._top("row"):
                self._close_row()
        elif tag == "table":
            self._close_open()
            self.depth = max(0, self.depth - 1)
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
        if tag in _BLOCK:
            self.buf.append(NL)

    def handle_data(self, data):
        if self.skip:
            return
        if not self.pre:
            data = re.sub(r"[ \t\r\n\f\v]+", " ", data)  # HTML whitespace collapsing (nbsp is kept)
        self.buf.append(data)

    def result(self):
        for f in self.frames:  # close everything still open (sloppy markup)
            f[1] = 0
        self.depth = 0
        self._close_open()
        return "".join(self.root)


_LEG = re.compile("[¡-ÿ]")


def _mostly_legacy(t):
    """Legacy Hindi fonts (Kruti/Shusha...) arrive as Latin-1 supplement gibberish such as '¸êü»Ö ´ÖÓ¡Öß'."""
    n = len(re.sub(r"\s", "", t))
    k = len(_LEG.findall(t))
    return k >= 3 and k / max(n, 1) >= 0.4


def strip_legacy_hindi(ln):
    """Return (line, removed_flag): drop '(...)' groups and whole lines made of legacy-font glyph soup."""
    hit = [False]

    def grp(m):
        if _mostly_legacy(m.group(1)):
            hit[0] = True
            return " "
        return m.group(0)

    ln2 = re.sub(r"\(([^()]*)\)", grp, ln)
    if _mostly_legacy(ln2):
        return "", True
    return re.sub(r"[ \t]+", " ", ln2).strip(), hit[0]


def html_bytes_to_lines(b):
    """Return (lines, stats).  Lines are stripped, empty lines dropped, table rows joined with ' | '."""
    text, enc = decode_html(b)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)           # incl. <!--[if gte mso 9]><xml>..
    text = re.sub(r"<!\[if[^\]]*\]>|<!\[endif\]>", " ", text, flags=re.I)
    text = re.sub(r"<(script|style)\b.*?</\1\s*>", " ", text, flags=re.S | re.I)
    p = _P()
    try:
        p.feed(text)
        p.close()
        raw = p.result()
    except Exception:  # pathological markup: fall back to tag stripping
        raw = re.sub(r"<(?:br|/p|/div|/tr)\b[^>]*>", NL, text, flags=re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
    lines, legacy = [], False
    for ln in raw.split(NL):
        ln = ln.replace("\xa0", " ")
        ln, hit = strip_legacy_hindi(ln)
        legacy = legacy or hit
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        ln = re.sub(r"^(?:\|\s*)+|(?:\s*\|)+$", "", ln).strip()  # empty cells at row edges
        ln = re.sub(r"(?:\s*\|\s*){2,}", " | ", ln)             # runs of empty cells
        if ln:
            lines.append(ln)
    return lines, {"n_pages": 1, "n_media": len(re.findall(rb"<img\b", b, re.I)), "encoding": enc, "legacy_hindi": legacy}


def htm_lines(path):
    with open(path, "rb") as f:
        return html_bytes_to_lines(f.read())
