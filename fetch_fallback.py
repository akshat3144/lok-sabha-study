"""Follow-up fetch for answers the main jobs could not give us text for.

Targets (read from data/clean/questions.parquet):
  * any Lok Sabha: answer missing or "statement laid on the Table" with no statement body
  * 18th LS: the downloaded .docx was unusable (not a Word file, old .doc, image-only scan)
Files go to data/raw/docs_fallback/ls<N>/<qid>.<ext>. Same safety as collect.py: atomic writes,
skip what exists, Ctrl+C / `python collect.py stop` to pause, re-run to resume.

    python fetch_fallback.py            # fetch
    python fetch_fallback.py --list     # only write data/clean/fallback_targets.csv and show counts
"""
import os, sys, time, random, json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works under `python -I`
import collect as C

OUT = os.path.join(C.DATA, "raw", "docs_fallback")
TARGETS = os.path.join(C.DATA, "clean", "fallback_targets.csv")
BAD_DOC_HINTS = ("baddocx", "no_document_xml", "legacy_ole", "image_only")  # values written by pipeline/clean.py


def build_targets():
    cols = ["qid", "ls", "answer_missing", "statement_empty", "source", "doc_note", "extract_error", "pdf_url",
            "docx_url", "doc_file"]
    d = pd.read_parquet(os.path.join(C.DATA, "clean", "questions.parquet"), columns=cols)
    for c in ("pdf_url", "docx_url"):  # some records carry a blank " " link: treat as no link
        s = d[c].astype("string").str.strip()
        d[c] = s.where(s.str.startswith("http", na=False))
    note = (d["doc_note"].fillna("").astype(str) + " " + d["extract_error"].fillna("").astype(str)).str.lower()
    bad_doc = (d["ls"] == 18) & note.apply(lambda s: any(h in s for h in BAD_DOC_HINTS))
    no_text = (d["answer_missing"].fillna(False) | d["statement_empty"].fillna(False))
    # 18th-LS rows whose docx simply hasn't been downloaded yet are the main docs job's business, not ours
    # only rows whose main 18th-LS doc hasn't been downloaded at all belong to the main docs job
    pending18 = (d["ls"] == 18) & d["doc_file"].isna()
    t = d[(bad_doc | (no_text & ~pending18)) & (d["pdf_url"].notna() | d["docx_url"].notna())].copy()
    t["reason"] = ["bad_docx" if b else "no_answer_text" for b in bad_doc.loc[t.index]]
    # 18th LS: the docx is what we already have (and it failed), so go to the PDF;
    # earlier LS: try the (smaller) docx first
    t["order"] = ["pdf" if (r == "bad_docx" or ls == 18) else "docx,pdf" for r, ls in zip(t["reason"], t["ls"])]
    t[["qid", "ls", "reason", "order", "docx_url", "pdf_url"]].to_csv(TARGETS, index=False)
    return t


def done(qid, ls):
    base = os.path.join(OUT, f"ls{ls}", qid)
    return any(os.path.exists(base + e) for e in (".docx", ".pdf", ".htm", ".bin"))


def main():
    t = build_targets()
    print(t.groupby(["ls", "reason"]).size().to_string())
    if "--list" in sys.argv:
        return
    import signal
    signal.signal(signal.SIGINT, C.on_sigint)
    if os.path.exists(C.STOP_FILE):
        os.remove(C.STOP_FILE)
    if "--workers" in sys.argv:  # e.g. `python fetch_fallback.py --workers 5`
        C.WORKERS = max(1, min(6, int(sys.argv[sys.argv.index("--workers") + 1])))
        print(f"using {C.WORKERS} workers")
    todo = [r for r in t.itertuples() if not done(r.qid, r.ls)]
    stats = {"job": "fallback", "files_total": len(t), "files_done": len(t) - len(todo), "failed": 0, "state": "running"}
    C.log("fallback", f"{len(t)} targets, {stats['files_done']} on disk, {len(todo)} to fetch")
    t0, n = time.time(), [0]

    def work(r):
        if C.should_stop() or done(r.qid, r.ls):
            return
        os.makedirs(os.path.join(OUT, f"ls{r.ls}"), exist_ok=True)
        for kind in r.order.split(","):
            url = r.docx_url if kind == "docx" else r.pdf_url
            if not isinstance(url, str) or not url:
                continue
            resp = C.fetch("fallback", url, timeout=90, max_tries=5)
            if resp is not None and resp.status_code == 200 and len(resp.content) > 500:
                head = resp.content[:512].lstrip().lower()
                ext = (".pdf" if head.startswith(b"%pdf") else ".docx" if head.startswith(b"pk")
                       else ".htm" if head.startswith((b"<", b"\xef\xbb\xbf<")) or b"<html" in head else ".bin")
                C.atomic_write(os.path.join(OUT, f"ls{r.ls}", r.qid + ext), resp.content)
                with C.lock:
                    stats["files_done"] += 1
                    n[0] += 1
                    if n[0] % 100 == 0:
                        rate = n[0] / (time.time() - t0)
                        stats["eta_hours"] = round((stats["files_total"] - stats["files_done"]) / rate / 3600, 2)
                        C.log("fallback", f"progress {stats['files_done']}/{stats['files_total']} files, ETA {stats['eta_hours']} h, failures {stats['failed']}")
                        C.write_status("fallback", stats)
                C.polite_sleep(C.BASE_DELAY * 0.5 + random.uniform(0, 0.5))
                return
            C.polite_sleep(C.BASE_DELAY)
        if not C.should_stop():
            with C.lock:
                stats["failed"] += 1
                with open(os.path.join(C.DATA, "fallback_failed.jsonl"), "a", encoding="utf-8") as f:
                    clean = lambda v: v if isinstance(v, str) else None  # pandas NA isn't JSON-serialisable
                    f.write(json.dumps({"qid": r.qid, "docx": clean(r.docx_url), "pdf": clean(r.pdf_url)}) + "\n")

    with ThreadPoolExecutor(C.WORKERS) as ex:
        list(ex.map(work, todo))
    left = sum(1 for r in t.itertuples() if not done(r.qid, r.ls))
    stats["state"] = "stopped" if C.should_stop() else ("complete" if left == 0 else "incomplete")
    C.write_status("fallback", stats)
    C.log("fallback", f"FINISHED state={stats['state']} done={stats['files_done']}/{stats['files_total']} missing={left}")


if __name__ == "__main__":
    main()
