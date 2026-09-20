"""Source-parity check for M1 (and any text-based measure).

18th-LS answers come from Word/PDF files; 13th-17th LS answers come from the API text.  To show the measurement
does not depend on the source, take 17th-LS questions that have BOTH (API answer text AND a docx/pdf link), push the
file through the SAME extraction code the pipeline uses for 18th-LS documents, and compare the two versions.

    python -I -X utf8 analysis/parity_check.py --stage list                 # DRY RUN: choose + print the sample, no network
    python -I -X utf8 analysis/parity_check.py --stage download             # polite, resumable (collect.fetch)
    python -I -X utf8 analysis/parity_check.py --stage extract              # text / part-count parity (local, CPU)
    python -I -X utf8 analysis/parity_check.py --stage score                # M1 on both versions (GPU box; needs fine-tuned model)
    python -I -X utf8 analysis/parity_check.py --stage report               # writes results/parity_report.md

`report` works with whatever is available (text parity always; M1 parity once `score` has run).
The report contains only identifiers and statistics, never answer text (copyright).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "analysis")
for p in (ROOT, os.path.join(ROOT, "pipeline"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CLEAN_PATH = os.path.join(ROOT, "data", "clean", "questions.parquet")
PARITY_DIR = os.path.join(ROOT, "data", "raw", "parity", "ls17")
CACHE_DIR = os.path.join(ROOT, "data", "raw", "parity", "_cache")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_CSV = os.path.join(RESULTS_DIR, "parity_sample.csv")
PAIRS_CSV = os.path.join(RESULTS_DIR, "parity_pairs.csv")
REPORT_MD = os.path.join(RESULTS_DIR, "parity_report.md")
PARITY_PARTS = os.path.join(ROOT, "data", "measures", "parity_parts.parquet")
PARITY_QUESTIONS = os.path.join(ROOT, "data", "measures", "parity_questions.parquet")
PARITY_M1_DIR = os.path.join(ROOT, "data", "measures", "parity_m1")

LS = 17
MIN_API_ANSWER_CHARS = 30


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ======================================================================================
# 1. Sample selection (stratified by ministry)
# ======================================================================================
def _http(u):
    u = str(u or "").strip()
    return u if u.startswith("http") else None


def links_of(row):
    """[(url, ext)] in the order collect.run_docs tries them: docx first, then pdf."""
    out = []
    for col, ext in (("docx_url", ".docx"), ("pdf_url", ".pdf")):
        u = _http(row.get(col) if isinstance(row, dict) else getattr(row, col, None))
        if u:
            out.append((u, ext))
    return out


def allocate(counts, n, min_each=2):
    """Allocate n picks over strata (dict stratum -> available). Proportional (largest remainder), at least
    min_each (or all that exist) per stratum, never more than available; total == min(n, sum(counts))."""
    total = sum(counts.values())
    n = min(n, total)
    alloc = {k: min(v, min_each) for k, v in counts.items()}
    if sum(alloc.values()) > n:  # more strata than picks: keep the largest strata
        keep = sorted(counts, key=lambda k: (-counts[k], k))[:n]
        return {k: (1 if k in keep else 0) for k in counts}
    rest = n - sum(alloc.values())
    room = {k: counts[k] - alloc[k] for k in counts}
    share = {k: rest * room[k] / max(sum(room.values()), 1) for k in counts}
    base = {k: min(room[k], int(math.floor(share[k]))) for k in counts}
    for k in counts:
        alloc[k] += base[k]
    left = n - sum(alloc.values())
    for k in sorted(counts, key=lambda k: (-(share[k] - math.floor(share[k])), k)):
        if left <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            left -= 1
    return alloc


def _h(seed, qid):
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def select_sample(df, n=300, seed=17, min_each=2):
    """df: clean questions of LS17 with columns qid, ministry, answer_text, docx_url, pdf_url, answer_missing,
    qa_mismatch, question_missing. Returns the sample (qid, ministry, docx_url, pdf_url), deterministic."""
    d = df[(df["answer_text"].fillna("").str.len() >= MIN_API_ANSWER_CHARS)]
    for c in ("answer_missing", "qa_mismatch", "question_missing"):
        if c in d:
            d = d[~d[c].fillna(False).astype(bool)]
    d = d[d["docx_url"].map(_http).notna() | d["pdf_url"].map(_http).notna()]
    counts = d.groupby("ministry").size().to_dict()
    alloc = allocate(counts, n, min_each)
    picks = []
    for m, k in alloc.items():
        if k <= 0:
            continue
        g = d[d["ministry"] == m]
        order = sorted(g["qid"], key=lambda q: _h(seed, q))[:k]
        picks.extend(order)
    out = d[d["qid"].isin(picks)][["qid", "ministry", "docx_url", "pdf_url"]].copy()
    out["_h"] = out["qid"].map(lambda q: _h(seed, q))
    return out.sort_values("_h").drop(columns="_h").reset_index(drop=True)


def load_ls17(path=CLEAN_PATH, columns=None):
    cols = columns or ["qid", "ls", "ministry", "answer_text", "docx_url", "pdf_url", "answer_missing", "qa_mismatch",
                       "question_missing"]
    return pd.read_parquet(path, columns=cols, filters=[("ls", "==", LS)])


def on_disk(qid, outdir=PARITY_DIR):
    for ext in (".docx", ".pdf"):
        p = os.path.join(outdir, qid + ext)
        if os.path.exists(p):
            return p
    return None


def run_list(args):
    df = load_ls17(args.clean)
    sample = select_sample(df, args.n, args.seed, args.min_each)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample.to_csv(args.sample_csv, index=False)
    have = sum(on_disk(q, args.outdir) is not None for q in sample["qid"])
    print(f"LS{LS}: {len(df):,} questions, {int(df['docx_url'].map(_http).notna().sum()):,} with docx link")
    print(f"sample: {len(sample)} questions over {sample['ministry'].nunique()} ministries; "
          f"{have} already on disk in {args.outdir}; {len(sample) - have} to download "
          f"(~{(len(sample) - have) * 1.5 / 60:.0f} min at the polite rate); manifest -> {args.sample_csv}")
    print(sample.groupby("ministry").size().sort_values(ascending=False).head(args.show).to_string())
    print("first rows:")
    print(sample.head(5)[["qid", "ministry", "docx_url"]].to_string(index=False))
    return sample


# ======================================================================================
# 2. Downloader (reuses collect.fetch / collect.atomic_write; polite, resumable)
# ======================================================================================
def run_download(args):
    sample = pd.read_csv(args.sample_csv) if os.path.exists(args.sample_csv) and not args.reselect else None
    if sample is None:
        sample = select_sample(load_ls17(args.clean), args.n, args.seed, args.min_each)
    import collect  # lazy: importing collect creates its data folders / log helpers

    import signal
    signal.signal(signal.SIGINT, collect.on_sigint)
    os.makedirs(args.outdir, exist_ok=True)
    fail_log = os.path.join(os.path.dirname(args.outdir), "failed.jsonl")
    todo = [r for r in sample.to_dict("records") if on_disk(r["qid"], args.outdir) is None]
    if args.limit:
        todo = todo[:args.limit]
    log(f"{len(sample) - len([r for r in sample.to_dict('records') if on_disk(r['qid'], args.outdir) is None])} on disk, "
        f"{len(todo)} to fetch (one at a time, ~{collect.BASE_DELAY * 0.75:.1f} s apart)")
    ok = failed = 0
    for i, r in enumerate(todo):
        if collect.should_stop():
            log("stop requested; exiting (re-run to resume)")
            break
        got = False
        for url, ext in links_of(r):
            resp = collect.fetch("parity", url, timeout=90, max_tries=5)
            if resp is not None and resp.status_code == 200 and len(resp.content) > 500:
                collect.atomic_write(os.path.join(args.outdir, r["qid"] + ext), resp.content)
                got = True
                collect.polite_sleep(collect.BASE_DELAY * 0.5 + random.uniform(0, 0.5))
                break
            collect.polite_sleep(collect.BASE_DELAY)
        if got:
            ok += 1
        else:
            failed += 1
            if not collect.should_stop():
                with open(fail_log, "a", encoding="utf8") as f:
                    f.write(json.dumps({"qid": r["qid"], "docx": r.get("docx_url"), "pdf": r.get("pdf_url")}) + "\n")
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(todo)} (ok {ok}, failed {failed})")
    log(f"download finished: ok {ok}, failed {failed} (failures in {fail_log}; re-run to retry)")


# ======================================================================================
# 3. Extraction with the pipeline's own functions + side-by-side parts
# ======================================================================================
def _plain(parts):
    out = []
    for p in (list(parts) if parts is not None else []):
        p = dict(p)
        p["labels"] = [str(x) for x in (list(p.get("labels")) if p.get("labels") is not None else [])]
        out.append(p)
    return out


def extract_file_answer(path, question_parts, cache_dir=CACHE_DIR):
    """Run an answer file through the SAME code path clean.build_record uses for 18th-LS documents:
    load_doc_lines -> normalize_line -> split_doc_lines -> tp.clean_text(strip_numbers) -> process_answer(wrap=True).
    The question side is the API question (identical in both versions, as in production where only the
    answer is missing). Returns dict(answer_text, answer_parts, statement_laid, doc_note, found_heading, ...)."""
    import clean as cl
    import textparse as tp
    cl.CACHE_DIR = cache_dir  # keep the parity files out of the pipeline's own doc-line cache
    lines, stats, err = cl.load_doc_lines(path)
    res = {"answer_text": "", "answer_parts": [], "statement_laid": False, "doc_note": None, "found_heading": None,
           "extract_error": err, "doc_pages": stats.get("n_pages") if stats else None, "file_q_body": ""}
    nl = [tp.normalize_line(l) for l in lines]
    nl = [l for l in nl if l]
    ql, al, found = cl.split_doc_lines(nl)
    res["found_heading"] = bool(found)
    qtxt, _ = tp.clean_text(lines=ql, wrap=False, strip_numbers=True)
    atxt, _ = tp.clean_text(lines=al, wrap=False, strip_numbers=True)
    body, _how = cl.extract_question_body(qtxt)
    res["file_q_body"] = tp.reflow(body) if body else ""
    if err:
        res["doc_note"] = "bad_" + err.split(":")[-1][:30]
    elif not nl:
        res["doc_note"] = "image_only" if (stats or {}).get("n_media") else "empty_doc"
    elif not found:
        res["doc_note"] = "no_answer_heading"
    if not atxt:
        return res
    qp = _plain(question_parts)
    qlabels = [l for p in qp for l in p["labels"]]
    q_scheme = (qp[-1]["labels"] and tp.label_scheme(qp[-1]["labels"][0])) if qp else None
    ans = cl.process_answer(atxt, q_scheme=q_scheme, wrap=True, q_labels=qlabels)
    res.update(answer_text=ans["answer_text"], answer_parts=ans["answer_parts"], statement_laid=ans["statement_laid"])
    return res


def build_part_rows(qid, question_text, question_parts, answer_text, answer_parts, version):
    """Rows in the exact schema of analysis/parts.py (re-uses parts.build) with qid/part_id suffixed by version."""
    import parts as pt
    df = pd.DataFrame([{"qid": qid, "question_parts": question_parts, "answer_parts": answer_parts,
                        "question_text": question_text, "answer_text": answer_text}])
    p = pt.build(df)
    p["version"] = version
    p["qid_base"] = qid
    p["qid"] = f"{qid}@{version}"
    p["part_id"] = [f"{qid}@{version}#{l}" for l in p["part_label"]]
    return p


# ---------- comparison metrics ----------
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[\W_]+")


def norm_ws(s):
    return _WS.sub(" ", (s or "").lower()).strip()


def alnum(s):
    return _NONALNUM.sub("", (s or "").lower())


def char_similarity(a, b):
    """Normalised Indel (Levenshtein-without-substitution) similarity in [0, 1]; both empty -> 1."""
    from rapidfuzz import fuzz
    a, b = norm_ws(a), norm_ws(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def content_similarity(a, b):
    """Same, after dropping everything but letters and digits (insensitive to spacing / bullets / punctuation)."""
    from rapidfuzz import fuzz
    a, b = alnum(a), alnum(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def n_labelled(answer_parts):
    return len([p for p in _plain(answer_parts) if p["label"] not in ("_pre", "_all")])


def label_set(answer_parts):
    return {l for p in _plain(answer_parts) for l in p["labels"]}


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare_one(api, file):
    """api / file: dict(answer_text, answer_parts, statement_laid[, q body]). Returns the metric row."""
    a, f = api["answer_text"] or "", file["answer_text"] or ""
    return {"api_len": len(a), "file_len": len(f),
            "len_ratio": (len(f) / len(a)) if a else np.nan,
            "char_sim": char_similarity(a, f), "content_sim": content_similarity(a, f),
            "n_parts_api": n_labelled(api["answer_parts"]), "n_parts_file": n_labelled(file["answer_parts"]),
            "parts_equal": n_labelled(api["answer_parts"]) == n_labelled(file["answer_parts"]),
            "label_jaccard": jaccard(label_set(api["answer_parts"]), label_set(file["answer_parts"])),
            "statement_laid_api": bool(api.get("statement_laid")), "statement_laid_file": bool(file.get("statement_laid"))}


def run_extract(args):
    sample = pd.read_csv(args.sample_csv)
    cols = ["qid", "ministry", "question_text", "answer_text", "question_parts", "answer_parts", "statement_laid"]
    df = pd.read_parquet(args.clean, columns=cols, filters=[("qid", "in", sample["qid"].tolist())]).set_index("qid")
    rows, part_rows, q_rows = [], [], []
    n_missing = 0
    for r in sample.itertuples(index=False):
        path = on_disk(r.qid, args.outdir)
        if path is None:
            n_missing += 1
            continue
        a = df.loc[r.qid]
        api = {"answer_text": a["answer_text"], "answer_parts": _plain(a["answer_parts"]),
               "statement_laid": bool(a["statement_laid"])}
        fx = extract_file_answer(path, a["question_parts"], args.cache_dir)
        file = {"answer_text": fx["answer_text"], "answer_parts": fx["answer_parts"], "statement_laid": fx["statement_laid"]}
        m = {"qid": r.qid, "ministry": r.ministry, "ext": os.path.splitext(path)[1].lstrip("."), "doc_note": fx["doc_note"],
             "found_heading": fx["found_heading"], "extract_error": fx["extract_error"], "doc_pages": fx["doc_pages"],
             "q_char_sim": char_similarity(a["question_text"], fx["file_q_body"]) if fx["file_q_body"] else np.nan,
             "file_answer_empty": not bool(fx["answer_text"].strip())}
        m.update(compare_one(api, file))
        rows.append(m)
        qp = _plain(a["question_parts"])
        for ver, ans_text, ans_parts in (("api", api["answer_text"], api["answer_parts"]),
                                         ("file", file["answer_text"], file["answer_parts"])):
            part_rows.append(build_part_rows(r.qid, a["question_text"], qp, ans_text, ans_parts, ver))
            q_rows.append({"qid": f"{r.qid}@{ver}", "question_text": a["question_text"], "answer_text": ans_text,
                           "version": ver, "qid_base": r.qid})
    if not rows:
        log(f"no downloaded files found in {args.outdir} ({n_missing} sampled questions); run --stage download first")
        return None
    pairs = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs.to_csv(args.pairs_csv, index=False)
    os.makedirs(os.path.dirname(PARITY_PARTS), exist_ok=True)
    pd.concat(part_rows, ignore_index=True).to_parquet(args.parity_parts, index=False)
    pd.DataFrame(q_rows).to_parquet(args.parity_questions, index=False)
    log(f"extracted {len(pairs)} files ({n_missing} sampled questions not downloaded) -> {args.pairs_csv}, "
        f"{args.parity_parts}")
    return pairs


# ======================================================================================
# 4. M1 on both versions (GPU box) -- re-uses m1_embed's scoring pipeline unchanged
# ======================================================================================
def run_score(args):
    import m1_embed as m1
    if not os.path.exists(args.parity_parts):
        raise SystemExit("run --stage extract first")
    argv = ["--stage", "score", "--parts", args.parity_parts, "--questions", args.parity_questions,
            "--out-dir", args.m1_dir, "--model", args.model, "--batch-size", str(args.batch_size),
            "--max-len", str(args.max_len)]
    if args.model_dir:
        argv += ["--model-dir", args.model_dir]
    if args.untuned_only:
        argv += ["--untuned-only"]
    m1.main(argv)


# ======================================================================================
# 5. Statistics: paired test, Bland-Altman, equivalence, concordance
# ======================================================================================
def lins_ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.cov(x, y, bias=True)[0, 1]
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def bland_altman(a, b, level=0.95):
    """a = reference (API), b = file. diff = b - a. Returns bias, SD, 95 % limits of agreement and their CIs,
    and the proportional-bias slope (regression of diff on mean)."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d, m = b - a, (a + b) / 2
    n = len(d)
    bias, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    t = stats.t.ppf(0.5 + level / 2, n - 1) if n > 1 else float("nan")
    se_bias = sd / math.sqrt(n) if n > 1 else float("nan")
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    se_loa = sd * math.sqrt(1 / n + 1.96 ** 2 / (2 * (n - 1))) if n > 1 else float("nan")
    out = {"n": int(n), "bias": bias, "sd_diff": sd, "bias_ci": [bias - t * se_bias, bias + t * se_bias],
           "loa": [lo, hi], "loa_lower_ci": [lo - t * se_loa, lo + t * se_loa],
           "loa_upper_ci": [hi - t * se_loa, hi + t * se_loa]}
    if n > 3 and m.std() > 0:
        lr = stats.linregress(m, d)
        out["prop_bias_slope"], out["prop_bias_p"] = float(lr.slope), float(lr.pvalue)
    return out


def paired_tests(a, b, margin=0.03):
    """Paired t, Wilcoxon signed-rank, Cohen's dz, TOST equivalence on the mean difference (b - a) with +/- margin,
    Pearson / Spearman / Lin's CCC."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    n = len(d)
    out = {"n": int(n), "mean_diff": float(d.mean()) if n else float("nan")}
    if n < 3:
        return out
    t = stats.ttest_rel(b, a)
    out.update(t_stat=float(t.statistic), t_p=float(t.pvalue), cohens_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0)
    if np.any(d != 0):
        w = stats.wilcoxon(d)
        out.update(wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue))
    else:
        out.update(wilcoxon_stat=0.0, wilcoxon_p=1.0)
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:  # identical versions (degenerate): equivalent iff the (zero) difference is inside the margin
        out.update(tost_margin=margin, tost_p=0.0 if abs(d.mean()) < margin else 1.0, equivalent=bool(abs(d.mean()) < margin))
    if se > 0:
        p_lo = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)   # H0: diff <= -margin
        p_hi = stats.t.cdf((d.mean() - margin) / se, n - 1)       # H0: diff >= +margin
        out.update(tost_margin=margin, tost_p=float(max(p_lo, p_hi)), equivalent=bool(max(p_lo, p_hi) < 0.05))
    out.update(pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(stats.spearmanr(a, b).statistic), ccc=lins_ccc(a, b))
    return out


def cluster_bootstrap_mean_diff(df, a_col, b_col, cluster="qid_base", n_boot=2000, seed=0):
    """Part-level mean(b - a) with a percentile CI from resampling questions (parts within a question are dependent)."""
    d = df.dropna(subset=[a_col, b_col])
    g = d.assign(_d=d[b_col] - d[a_col]).groupby(cluster)["_d"].agg(["sum", "count"])
    if len(g) < 2:
        return {"mean_diff": float("nan"), "ci": [float("nan")] * 2, "n_parts": int(len(d)), "n_clusters": int(len(g))}
    rng = np.random.default_rng(seed)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return {"mean_diff": float(s.sum() / c.sum()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_parts": int(len(d)), "n_clusters": int(len(g))}


def cohen_kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def m1_parity(m1_dir, margin=0.03, threshold=None, model_id=None):
    """Pair API vs file versions from the scored parity parts. Returns dict of results (or None if not scored)."""
    import m1_embed as m1
    sp, qp = m1.out_paths(model_id or m1.DEFAULT_MODEL, m1_dir)
    if not os.path.exists(sp):
        return None
    s = pd.read_parquet(sp)
    q = pd.read_parquet(qp)
    for df in (s, q):
        df["qid_base"] = df["qid"].str.rsplit("@", n=1).str[0]
        df["version"] = df["qid"].str.rsplit("@", n=1).str[1]
    s["part_label"] = s["part_label"].astype(str)
    pv = s.pivot_table(index=["qid_base", "part_label"], columns="version", values=["m1_cos", "m1_cos_untuned"], aggfunc="first")
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv.reset_index()
    qv = q.pivot_table(index="qid_base", columns="version",
                       values=["m1_q_cos", "m1_part_mean", "m1_part_min", "m1_q_cos_untuned", "m1_part_mean_untuned"], aggfunc="first")
    qv.columns = [f"{a}_{b}" for a, b in qv.columns]
    qv = qv.reset_index()
    res = {"n_questions_scored": int(len(qv)), "n_parts_paired": int(len(pv))}
    for name, tab, cols in (("question_part_mean", qv, ("m1_part_mean_api", "m1_part_mean_file")),
                            ("question_whole_text", qv, ("m1_q_cos_api", "m1_q_cos_file")),
                            ("question_part_min", qv, ("m1_part_min_api", "m1_part_min_file")),
                            ("question_part_mean_untuned", qv, ("m1_part_mean_untuned_api", "m1_part_mean_untuned_file")),
                            ("part_level", pv, ("m1_cos_api", "m1_cos_file")),
                            ("part_level_untuned", pv, ("m1_cos_untuned_api", "m1_cos_untuned_file"))):
        if cols[0] not in tab or cols[1] not in tab:
            continue
        a, b = tab[cols[0]].to_numpy(), tab[cols[1]].to_numpy()
        r = {"tests": paired_tests(a, b, margin), "bland_altman": bland_altman(a, b)}
        if name.startswith("part_level"):
            r["cluster_bootstrap"] = cluster_bootstrap_mean_diff(tab, cols[0], cols[1])
        res[name] = r
    if threshold is not None and "m1_cos_api" in pv:
        ok = pv.dropna(subset=["m1_cos_api", "m1_cos_file"])
        x, y = ok["m1_cos_api"] >= threshold, ok["m1_cos_file"] >= threshold
        res["coverage_classification"] = {"threshold": threshold, "agreement": float((x == y).mean()), "kappa": cohen_kappa(x, y),
                                          "n": int(len(ok))}
    # by file type
    ext = pd.read_csv(PAIRS_CSV)[["qid", "ext"]].rename(columns={"qid": "qid_base"}) if os.path.exists(PAIRS_CSV) else None
    if ext is not None:
        qe = qv.merge(ext, on="qid_base")
        res["by_ext"] = {e: {"n": int(len(g)), "mean_diff_part_mean": float((g["m1_part_mean_file"] - g["m1_part_mean_api"]).mean())}
                         for e, g in qe.groupby("ext")}
    return res


# ======================================================================================
# 6. Report
# ======================================================================================
def _q(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {q: float(np.nanquantile(x, q)) for q in qs} if len(x) else {}


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ba_row(name, r):
    t, b = r["tests"], r["bland_altman"]
    lo, hi = b["loa"]
    eq = t.get("equivalent")
    return (f"| {name} | {t['n']} | {_f(b['bias'], 4)} [{_f(b['bias_ci'][0], 4)}, {_f(b['bias_ci'][1], 4)}] | "
            f"[{_f(lo)}, {_f(hi)}] | {_f(t.get('t_p'), 4)} | {_f(t.get('wilcoxon_p'), 4)} | "
            f"{'yes' if eq else 'no' if eq is not None else 'n/a'} | {_f(t.get('ccc'))} |")


def write_report(pairs, m1res, path, n_sample, margin, sample=None, failures=0):
    L = []
    w = L.append
    w("# Source-parity check: API text vs Word/PDF file extraction (17th Lok Sabha)")
    w("")
    w(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `analysis/parity_check.py`. "
      "Contains identifiers and statistics only; no answer text.")
    w("")
    w("## Design")
    w(f"- Sample: {n_sample} 17th-LS questions stratified by ministry that have API answer text AND a docx/pdf link "
      f"(selection is deterministic, hash-ordered; manifest in `results/parity_sample.csv`).")
    w("- The file is processed by the pipeline's own 18th-LS document path (`clean.load_doc_lines` -> `split_doc_lines` -> "
      "`tp.clean_text` -> `process_answer(wrap=True)`); the question side is the API question in both versions, as in "
      "production where only the answer is missing. Parts are built with `analysis/parts.py`.")
    w("- Each question therefore has two answers: `api` (reference) and `file`. Differences = extraction / formatting effects.")
    w("")
    n = len(pairs)
    w(f"## 1. Coverage and extraction diagnostics (n = {n} downloaded files)")
    w("")
    w(f"- Downloaded/extracted: {n} of {n_sample} sampled" + (f" ({failures} download failures logged)" if failures else "") + ".")
    w("- File type: " + ", ".join(f"{k} {v}" for k, v in pairs["ext"].value_counts().items()))
    w(f"- File answer empty after extraction: {pairs['file_answer_empty'].sum()} ({pairs['file_answer_empty'].mean():.1%})")
    w(f"- 'ANSWER' heading found: {pairs['found_heading'].mean():.1%}")
    notes = pairs["doc_note"].fillna("none").value_counts()
    w("- Extraction notes: " + ", ".join(f"{k} {v}" for k, v in notes.items()))
    w("")
    ok = pairs[~pairs["file_answer_empty"]]
    w("## 2. Text agreement (API answer vs file answer)")
    w("")
    w("| Metric | n | mean | p5 | p25 | median | p75 | p95 | share >= 0.95 | share >= 0.90 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for name, col in (("char similarity (Indel, whitespace-normalised)", "char_sim"),
                      ("content similarity (letters/digits only)", "content_sim"),
                      ("answer length ratio file/api", "len_ratio"),
                      ("question-body similarity (file vs API question)", "q_char_sim")):
        x = ok[col].dropna().to_numpy()
        if not len(x):
            continue
        qq = _q(x)
        sh = (lambda t: f"{(x >= t).mean():.1%}") if col != "len_ratio" else (lambda t: "")
        w(f"| {name} | {len(x)} | {x.mean():.3f} | {qq[0.05]:.3f} | {qq[0.25]:.3f} | {qq[0.5]:.3f} | {qq[0.75]:.3f} | {qq[0.95]:.3f} | {sh(0.95)} | {sh(0.90)} |")
    w("")
    w("By file type (median content similarity / median length ratio):")
    w("")
    for e, g in ok.groupby("ext"):
        w(f"- {e}: n={len(g)}, content sim {g['content_sim'].median():.3f}, length ratio {g['len_ratio'].median():.3f}")
    w("")
    w("## 3. Part structure agreement")
    w("")
    w(f"- Same number of labelled answer parts: {pairs['parts_equal'].mean():.1%} "
      f"(mean |difference| {np.abs(pairs['n_parts_api'] - pairs['n_parts_file']).mean():.3f})")
    w(f"- Mean Jaccard of covered part labels: {pairs['label_jaccard'].mean():.3f}")
    w(f"- 'Statement laid' flag agreement: {(pairs['statement_laid_api'] == pairs['statement_laid_file']).mean():.1%}")
    low = ok.nsmallest(10, "content_sim")[["qid", "ext", "content_sim", "n_parts_api", "n_parts_file", "doc_note"]]
    w("")
    w("Ten least similar pairs (for manual inspection; identifiers only):")
    w("")
    w("| qid | ext | content sim | parts api | parts file | note |")
    w("|---|---|---|---|---|---|")
    for r in low.itertuples(index=False):
        w(f"| {r.qid} | {r.ext} | {r.content_sim:.3f} | {r.n_parts_api} | {r.n_parts_file} | {'' if pd.isna(r.doc_note) else r.doc_note} |")
    w("")
    w("## 4. M1 score agreement")
    w("")
    if not m1res:
        w("M1 scores for the parity set are not available yet. Run on the GPU box:")
        w("")
        w("```\npython -I -X utf8 analysis/parity_check.py --stage score\npython -I -X utf8 analysis/parity_check.py --stage report\n```")
    else:
        w(f"Both versions scored by the same fine-tuned bi-encoder in the same run ({m1res['n_questions_scored']} questions, "
          f"{m1res['n_parts_paired']} paired parts). Difference = file - api. Tests are on question-level means (parts within a "
          f"question are not independent); part-level rows use a question-cluster bootstrap. Equivalence (TOST) margin "
          f"= +/-{margin} cosine.")
        w("")
        w("| Level / measure | n | bias [95% CI] | 95% limits of agreement | paired t p | Wilcoxon p | TOST equiv. | Lin's CCC |")
        w("|---|---|---|---|---|---|---|---|")
        for k in ("question_part_mean", "question_whole_text", "question_part_min", "question_part_mean_untuned", "part_level",
                  "part_level_untuned"):
            if k in m1res:
                w(_ba_row(k, m1res[k]))
        w("")
        for k in ("part_level", "part_level_untuned"):
            if k in m1res and "cluster_bootstrap" in m1res[k]:
                c = m1res[k]["cluster_bootstrap"]
                w(f"- {k}: mean diff {c['mean_diff']:.4f}, question-cluster bootstrap 95% CI [{c['ci'][0]:.4f}, {c['ci'][1]:.4f}] "
                  f"({c['n_parts']} parts, {c['n_clusters']} questions)")
        if "coverage_classification" in m1res:
            c = m1res["coverage_classification"]
            w(f"- Part coverage classification at threshold {c['threshold']:.3f}: agreement {c['agreement']:.1%}, kappa {c['kappa']:.3f} (n={c['n']})")
        for k in ("question_part_mean", "part_level"):
            b = m1res.get(k, {}).get("bland_altman", {})
            if "prop_bias_slope" in b:
                w(f"- {k}: proportional-bias slope (diff on mean) {b['prop_bias_slope']:.4f}, p={b['prop_bias_p']:.3g}")
        if "by_ext" in m1res:
            w("- By file type (question-level mean diff in part-mean cosine): " +
              ", ".join(f"{e} {v['mean_diff_part_mean']:.4f} (n={v['n']})" for e, v in m1res["by_ext"].items()))
        w("")
        w("Suggested pass criteria (set BEFORE looking at the numbers; adjust in the paper's pre-analysis note): "
          f"|bias| < {margin}, TOST equivalence at +/-{margin}, and 95% limits of agreement narrower than the spread of M1 "
          "between FULL and NONE human-coded parts.")
    w("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(L) + "\n")
    return "\n".join(L)


def run_report(args):
    if not os.path.exists(args.pairs_csv):
        raise SystemExit("no results/parity_pairs.csv: run --stage extract first")
    pairs = pd.read_csv(args.pairs_csv)
    n_sample = len(pd.read_csv(args.sample_csv)) if os.path.exists(args.sample_csv) else len(pairs)
    fail = os.path.join(os.path.dirname(args.outdir), "failed.jsonl")
    nfail = sum(1 for _ in open(fail, encoding="utf8")) if os.path.exists(fail) else 0
    m1res = m1_parity(args.m1_dir, args.margin, args.threshold, args.model)
    write_report(pairs, m1res, args.report, n_sample, args.margin, failures=nfail)
    if m1res:
        with open(args.report.replace(".md", ".json"), "w", encoding="utf8") as f:
            json.dump(m1res, f, indent=1)
    log(f"wrote {args.report}")


# ======================================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["list", "download", "extract", "score", "report"], default="list")
    ap.add_argument("--list", dest="stage", action="store_const", const="list", help="alias of --stage list (dry run)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-each", type=int, default=2, help="minimum questions per ministry (if available)")
    ap.add_argument("--limit", type=int, default=None, help="download at most N files this run")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--reselect", action="store_true", help="ignore the saved manifest and re-run the selection")
    ap.add_argument("--clean", default=CLEAN_PATH)
    ap.add_argument("--outdir", default=PARITY_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--sample-csv", default=SAMPLE_CSV)
    ap.add_argument("--pairs-csv", default=PAIRS_CSV)
    ap.add_argument("--parity-parts", default=PARITY_PARTS)
    ap.add_argument("--parity-questions", default=PARITY_QUESTIONS)
    ap.add_argument("--m1-dir", default=PARITY_M1_DIR)
    ap.add_argument("--report", default=REPORT_MD)
    ap.add_argument("--model", default="sentence-transformers/multi-qa-mpnet-base-cos-v1")
    ap.add_argument("--model-dir", default=None, help="fine-tuned model dir (default: m1_embed's models/m1/<slug>)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--untuned-only", action="store_true")
    ap.add_argument("--margin", type=float, default=0.03, help="equivalence margin (cosine units) for TOST")
    ap.add_argument("--threshold", type=float, default=None, help="M1 coverage threshold (from calibration) for classification agreement")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"list": run_list, "download": run_download, "extract": run_extract, "score": run_score, "report": run_report}[args.stage](args)


if __name__ == "__main__":
    main()
