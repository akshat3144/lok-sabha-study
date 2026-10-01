"""Image-only answer .docx files (scanned pages embedded as JPGs) -> one PDF per question, for Gemini OCR.

Images are taken in document order (the order of their r:embed references in word/document.xml), so pages stay
in sequence. Raw files are only read; PDFs go to data/derived/ocr_pdf/lsNN/<qid>.pdf.

    python -I analysis/docx_to_pdf.py          # all needs_ocr questions whose answer file is a .docx
"""
import io
import os
import re
import sys
import zipfile

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")
FALLBACK = os.path.join(ROOT, "data", "raw", "docs_fallback")
OUT = os.path.join(ROOT, "data", "derived", "ocr_pdf")


def ordered_images(z):
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8", "ignore")
    rid2t = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    rid2t.update({k: v for v, k in re.findall(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)})
    doc = z.read("word/document.xml").decode("utf-8", "ignore")
    seen, out = set(), []
    for rid in re.findall(r'r:(?:embed|id)="([^"]+)"', doc):
        t = rid2t.get(rid, "")
        name = "word/" + t.lstrip("/").removeprefix("word/") if t else ""
        if name.startswith("word/media/") and name not in seen:
            seen.add(name)
            out.append(name)
    rest = sorted(n for n in z.namelist() if n.startswith("word/media/") and n not in seen)
    return out + rest


def convert(qid, ls):
    src = os.path.join(FALLBACK, f"ls{ls}", f"{qid}.docx")
    if not os.path.exists(src):
        return "no_docx"
    dst = os.path.join(OUT, f"ls{ls}", f"{qid}.pdf")
    if os.path.exists(dst):
        return "exists"
    with zipfile.ZipFile(src) as z:
        imgs = [Image.open(io.BytesIO(z.read(n))).convert("RGB") for n in ordered_images(z)]
    if not imgs:
        return "no_images"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    imgs[0].save(dst + ".part", format="PDF", save_all=True, append_images=imgs[1:], resolution=200)
    os.replace(dst + ".part", dst)
    return "ok"


def main():
    q = pd.read_parquet(QUESTIONS, columns=["qid", "ls", "needs_ocr"])
    q = q[q["needs_ocr"].fillna(False).astype(bool)]
    res = pd.Series([convert(a, b) for a, b in q[["qid", "ls"]].values]).value_counts()
    print(res.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
