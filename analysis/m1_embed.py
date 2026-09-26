"""M1 -- bi-encoder relevance of an answer to the question part it addresses.

Method follows Alvarez & Morrier (2025, Political Analysis, "Measuring the Quality of Answers in Political
Q&As with LLMs"): fine-tune a sentence-transformers bi-encoder (default multi-qa-mpnet-base-cos-v1) with
MultipleNegativesRankingLoss on in-domain (question_part, answer_part) pairs from a 5 % train / 1 % validation
sample of QUESTIONS (split by qid), then score cosine(question_part, answer_part) for ALL parts.

    python analysis/m1_embed.py --stage pack                  # slim question table for upload (qid, question_text, answer_text)
    python analysis/m1_embed.py --stage estimate              # corpus statistics + time/cost estimate only
    python analysis/m1_embed.py --stage train                 # fine-tune (resumable)
    python analysis/m1_embed.py --stage score                 # tuned + untuned scores (resumable)
    python analysis/m1_embed.py --stage all                   # train + score
    python analysis/m1_embed.py --stage all --model BAAI/bge-base-en-v1.5   # robustness encoder
    python analysis/m1_embed.py --stage calibrate --labels data/coding/human_labels.csv

Inputs : data/measures/parts.parquet (analysis/parts.py), data/final/questions_analysis.parquet
Outputs: data/measures/m1_scores.parquet   part_id, qid, part_label, m1_cos, m1_cos_untuned, ...
         data/measures/m1_question.parquet qid, m1_q_cos, m1_part_mean, m1_part_min, ...
         (a non-default --model writes m1_scores__<slug>.parquet / m1_question__<slug>.parquet)

The raw text is copyright of the Lok Sabha Secretariat: run this only on your own private machine.
Heavy imports (torch, sentence-transformers) are lazy so the data-prep / splitting / aggregation /
calibration functions import and test without a GPU stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS_PATH = os.path.join(ROOT, "data", "measures", "parts.parquet")
QUESTIONS_PATH = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")
MEASURES_DIR = os.path.join(ROOT, "data", "measures")
DEFAULT_LABELS = os.path.join(ROOT, "data", "coding", "human_labels.csv")

DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
A100_USD_PER_HOUR = 0.89          # Jarvis Labs A100 40GB
ASSUMED_TOK_PER_S = 150_000       # conservative fp16 inference throughput, 110M-param encoder, length-sorted
MIN_ANSWER_CHARS = 20             # training-only filter
MIN_QUESTION_CHARS = 5            # training-only filter (nothing to match against)
CLIP_CHARS_PER_TOKEN = 8          # texts are clipped to max_len * 8 chars before tokenising (always >> max_len tokens)


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ======================================================================================
# 1. Text pre-processing
# ======================================================================================
# "A statement is laid on the Table of the House." / "Statement `A` is laid on the Table ..." is a pure lay
# sentence (the statement content follows and is kept).  NOT matched on purpose: "Information is being collected
# and will be laid on the Table of the House" -- that is a substantive DEFERRAL and must stay in the text.
_LAID = re.compile(
    r"(?i)\b(?:an?\s+|the\s+)?statements?(?:\s*[`'\"‘’“”]?\s*[A-Za-z0-9]{1,3}\s*[`'\"‘’“”]?)?"
    r"\s+(?:is|are|has\s+been|have\s+been)\s+laid\s*,?\s+(?:on|in|of|upon)\s+(?:the\s+|then\s+)?table\s+of\s+(?:the\s+)?(?:house|lok\s*sabha|sabha)\s*[.:]?")
_STARS = re.compile(r"(?:\*[ \t]*){2,}\.?")
_MIN_HDR = re.compile(
    r"^\W*(?:ANSWERS?\W+)?(?:THE\s+)?(?:PRIME\s+MINISTER|(?:DEPUTY\s+|UNION\s+|CABINET\s+)?MINISTER(?:\s+OF\s+STATE)?)\b", re.I)
_HDR_LINE = re.compile(
    r"^\W*(?:(?:STATEMENT|ANNEXURE|ANNEX)\W+)?(?:REFERRED\s+TO\b|ANSWER\s+TO\s+PARTS?\b|ANSWERS?\W*$).*", re.I)
_LEAD_LABEL = re.compile(
    r"^\s*(?:answers?\s+to\s+(?:parts?\s+)?)?\(\s*[A-Za-z]{1,3}\s*\)(?:\s*(?:,|and|&|to|-|–)\s*\(\s*[A-Za-z]{1,3}\s*\))*\s*[:.\-–]*\s*",
    re.I)
_WS = re.compile(r"\s+")


def _mostly_upper(s, thresh=0.6):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= thresh


def clean_answer(text):
    """Strip what is not answer content: minister/'ANSWER' header lines, the 'statement is laid' lay sentence,
    '*****' rules, 'STATEMENT REFERRED TO IN REPLY OF PARTS ..' headings, leading '(a) to (c):' labels;
    collapse whitespace.  Idempotent.  Returns '' for empty/None."""
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = text.replace("\r", "\n").split("\n")
    kept = []
    for i, l in enumerate(lines):
        s = l.strip()
        if not s:
            continue
        if i < 3 and len(s) < 400 and _MIN_HDR.match(s) and (_mostly_upper(s) or re.search(r"\(\s*(?:shri|dr|smt|sushri|prof)", s, re.I)):
            continue  # minister header (ordinary prose like "The Prime Minister`s 15 Point ..." is mixed-case: kept)
        if len(s) < 400 and _HDR_LINE.match(s) and (re.fullmatch(r"\W*answers?\W*", s, re.I)
                                                   or re.search(r"question\s*no|parts?\s*\(", s, re.I)):
            continue  # 'STATEMENT REFERRED TO IN REPLY TO PARTS (b) TO (g) OF ... QUESTION NO. 529 ...' (any case)
        kept.append(s)
    t = " ".join(kept)
    t = _LAID.sub(" ", t)
    t = _STARS.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _LEAD_LABEL.sub("", t, count=1)
    t = t.lstrip(" :.-–").strip()
    return t


def clean_question(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return _WS.sub(" ", text).strip()


# ======================================================================================
# 2. Split by qid (deterministic hash => stable when the corpus is rebuilt with more answers)
# ======================================================================================
def _qid_u(qid, seed):
    h = hashlib.sha1(f"{seed}:{qid}".encode("utf8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def assign_split(qids, train_frac=0.05, val_frac=0.01, seed=13):
    """'train' / 'val' / 'heldout' for each qid (array-like). All parts of one question share a split."""
    qids = pd.Series(qids)
    u = {q: _qid_u(q, seed) for q in qids.unique()}
    uu = qids.map(u).to_numpy()
    out = np.where(uu < train_frac, "train", np.where(uu < train_frac + val_frac, "val", "heldout"))
    return out


# ======================================================================================
# 3. Training pairs and leak-free batches
# ======================================================================================
def build_training_pairs(df, split_name, min_answer_chars=MIN_ANSWER_CHARS, min_question_chars=MIN_QUESTION_CHARS):
    """df needs columns qid, split, q (cleaned question part), a (cleaned answer part).
    Keeps rows of `split_name`, drops missing / very short answers (TRAINING ONLY) and exact duplicate pairs."""
    d = df[df["split"] == split_name]
    d = d[(d["a"].str.len() >= min_answer_chars) & (d["q"].str.len() >= min_question_chars)]
    d = d.drop_duplicates(subset=["q", "a"])
    return d[["qid", "q", "a"]].reset_index(drop=True)


def make_batches(qids, answers, batch_size, seed, min_last=8):
    """Index batches for MultipleNegativesRankingLoss with NO false in-batch negatives: no two pairs in a batch
    share a qid (sibling parts' answers are plausible matches) or an identical answer text.
    Deterministic given (seed). The undersized tail batch is dropped when smaller than `min_last`."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(qids)).tolist()
    batches = []
    while order:
        cur, seen_q, seen_a, defer = [], set(), set(), []
        for i in order:
            if len(cur) < batch_size and qids[i] not in seen_q and answers[i] not in seen_a:
                cur.append(i)
                seen_q.add(qids[i])
                seen_a.add(answers[i])
            else:
                defer.append(i)
        if len(cur) >= min_last or (len(cur) >= 2 and len(cur) == len(order)):
            batches.append(cur)
        elif len(cur) < min_last and not batches:
            batches.append(cur)  # tiny corpora (smoke tests): keep what we have
        if len(defer) == len(order):  # no progress
            break
        order = defer
    return batches


def latest_signature_ok(path, signature):
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f).get("signature") == signature
    except Exception:
        return False


# ======================================================================================
# 4. Evaluation of retrieval quality on held-out pairs (sanity check that tuning helped)
# ======================================================================================
def retrieval_metrics(q_emb, a_emb, groups):
    """q_emb[i] should retrieve a_emb[i] (or any candidate with the same group id) among all candidates.
    Returns acc@1, MRR, mean cosine of true pairs and of all other pairs."""
    q = np.asarray(q_emb, dtype=np.float32)
    a = np.asarray(a_emb, dtype=np.float32)
    g = np.asarray(groups)
    sims = q @ a.T
    correct = g[:, None] == g[None, :]
    best = np.where(correct, sims, -np.inf).max(1)
    rank = 1 + ((sims > best[:, None]) & ~correct).sum(1)
    pos = np.diag(sims)
    n = len(q)
    neg_mean = (sims.sum() - pos.sum()) / max(n * n - n, 1)
    return {"n": int(n), "acc1": float((rank == 1).mean()), "mrr": float((1.0 / rank).mean()),
            "cos_pos": float(pos.mean()), "cos_neg": float(neg_mean)}


# ======================================================================================
# 5. Universes of unique texts (dedup => shared answers are encoded once) and indices
# ======================================================================================
def index_texts(raw_series, clean_fn, clip_chars, prefix=""):
    """raw_series -> (idx int64 array [-1 where cleaned text is empty], unique texts WITH prefix,
    clipped bool per unique text).  Cleaning runs on unique raw strings only."""
    codes, uniq_raw = pd.factorize(pd.Series(raw_series).fillna(""), sort=False)
    cleaned = [clean_fn(u) for u in uniq_raw]
    ser = pd.Series(cleaned)
    codes2, uniq_clean = pd.factorize(ser.where(ser != "", None), sort=False)  # '' -> -1
    mapping = codes2  # raw-unique code -> clean-unique code
    idx = np.where(codes >= 0, mapping[np.clip(codes, 0, None)], -1).astype(np.int64)
    clipped = np.array([len(t) > clip_chars for t in uniq_clean], dtype=bool)
    texts = [prefix + t[:clip_chars] for t in uniq_clean]
    return idx, texts, clipped


def build_universes(parts, questions, max_len, q_prefix="", d_prefix=""):
    """parts: DataFrame(question_part, answer_part, ...), questions: DataFrame(qid, question_text, answer_text)
    -> dict with query-side / doc-side unique text lists and per-row indices (part-level and question-level)."""
    clip = max_len * CLIP_CHARS_PER_TOKEN
    n = len(parts)
    q_idx, q_texts, q_clip = index_texts(pd.concat([parts["question_part"], questions["question_text"]], ignore_index=True),
                                         clean_question, clip, q_prefix)
    a_idx, a_texts, a_clip = index_texts(pd.concat([parts["answer_part"], questions["answer_text"]], ignore_index=True),
                                         clean_answer, clip, d_prefix)
    return {"q_texts": q_texts, "a_texts": a_texts, "q_clipped": q_clip, "a_clipped": a_clip,
            "part_q": q_idx[:n], "part_a": a_idx[:n], "ques_q": q_idx[n:], "ques_a": a_idx[n:]}


def token_lengths(tokenizer, texts, max_len, batch=20000):
    """Token count per text, capped at max_len + 1 (so '> max_len' means truncated)."""
    out = np.empty(len(texts), dtype=np.int32)
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], add_special_tokens=True, truncation=True, max_length=max_len + 1,
                        return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        out[i:i + len(enc)] = [len(x) for x in enc]
    return out


def truncation_report(universes, q_len, a_len, max_len):
    """% of PART-level rows (valid ones) whose question / answer exceeded the model max length, and same at
    question level.  A row is truncated if its text was clipped or its token count > max_len."""
    q_tr = (q_len > max_len) | universes["q_clipped"]
    a_tr = (a_len > max_len) | universes["a_clipped"]

    def pct(idx, tr):
        v = idx[idx >= 0]
        return float(tr[v].mean() * 100) if len(v) else float("nan")
    return {"max_len": max_len,
            "part_question_pct_truncated": pct(universes["part_q"], q_tr),
            "part_answer_pct_truncated": pct(universes["part_a"], a_tr),
            "question_level_question_pct_truncated": pct(universes["ques_q"], q_tr),
            "question_level_answer_pct_truncated": pct(universes["ques_a"], a_tr)}


# ======================================================================================
# 6. Resumable embedding store
# ======================================================================================
def texts_digest(texts):
    h = hashlib.sha1()
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode("utf8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class EmbedStore:
    """fp16 embeddings of a text universe, stored in length-descending order in a .npy memmap, written in
    chunks; chunk completion is recorded in a json marker after the memmap is flushed, so an interrupted run
    resumes at the first unfinished chunk.  A different signature (texts / model / max_len) starts afresh."""

    def __init__(self, directory, name, n, dim, signature, chunk):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"emb_{name}.npy")
        self.marker = os.path.join(directory, f"emb_{name}.done.json")
        self.n, self.dim, self.chunk, self.signature = n, dim, chunk, signature
        st = None
        try:
            with open(self.marker, encoding="utf8") as f:
                st = json.load(f)
        except Exception:
            pass
        ok = (st is not None and st.get("signature") == signature and st.get("n") == n and st.get("dim") == dim
              and st.get("chunk") == chunk and os.path.exists(self.path))
        self.done = set(st["done"]) if ok else set()
        if not ok:
            if os.path.exists(self.marker):
                os.remove(self.marker)
            self.mm = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float16, shape=(n, dim))
        else:
            self.mm = np.load(self.path, mmap_mode="r+")

    @property
    def n_chunks(self):
        return math.ceil(self.n / self.chunk)

    def write(self, ci, emb):
        s = ci * self.chunk
        self.mm[s:s + len(emb)] = emb.astype(np.float16)
        self.mm.flush()
        self.done.add(ci)
        _atomic_json(self.marker, {"signature": self.signature, "n": self.n, "dim": self.dim, "chunk": self.chunk,
                                   "done": sorted(self.done)})

    def complete(self):
        return len(self.done) == self.n_chunks


def encode_universe(encode_fn, texts, store, order, batch_size, log_every=1, label="", cost_per_s=None,
                    tok_lens=None, on_chunk=None):
    """Encode `texts` in `order` (longest first), chunk by chunk, skipping chunks already in `store`.
    encode_fn(list[str]) -> float array [len, dim], L2-normalised.  Rows land at positions in `order`."""
    n = len(texts)
    t0, done_items, skipped = time.time(), 0, 0
    for ci in range(store.n_chunks):
        s, e = ci * store.chunk, min((ci + 1) * store.chunk, n)
        if ci in store.done:
            skipped += e - s
            continue
        emb = encode_fn([texts[i] for i in order[s:e]])
        store.write(ci, np.asarray(emb))
        done_items += e - s
        if on_chunk:
            on_chunk(ci)
        if (ci % log_every == 0) or e == n:
            el = time.time() - t0
            rate = done_items / max(el, 1e-9)
            left = (n - skipped - done_items) / max(rate, 1e-9)
            usd = f", ~${(left / 3600) * A100_USD_PER_HOUR:.2f} left" if cost_per_s else ""
            log(f"  [{label}] {s + (e - s)}/{n} texts ({rate:,.0f}/s), ETA {left / 60:.1f} min{usd}")
    return store


def pair_cosines(q_mm, a_mm, q_rows, a_rows, block=200_000):
    """cosine for pair rows (positions into the two stores; -1 => NaN). Embeddings are already L2-normalised."""
    out = np.full(len(q_rows), np.nan, dtype=np.float32)
    ok = np.where((q_rows >= 0) & (a_rows >= 0))[0]
    for s in range(0, len(ok), block):
        sel = ok[s:s + block]
        qe = np.asarray(q_mm[q_rows[sel]], dtype=np.float32)
        ae = np.asarray(a_mm[a_rows[sel]], dtype=np.float32)
        out[sel] = (qe * ae).sum(1)
    return out


# ======================================================================================
# 7. Aggregation
# ======================================================================================
def aggregate_question(scores, questions_qids, q_cos, q_cos_untuned=None):
    """Question-level table from part scores + whole-question cosines (aligned with questions_qids)."""
    g = scores.groupby("qid")["m1_cos"]
    agg = pd.DataFrame({"m1_part_mean": g.mean(), "m1_part_min": g.min(), "n_parts_scored": g.count()})
    if "m1_cos_untuned" in scores:
        agg["m1_part_mean_untuned"] = scores.groupby("qid")["m1_cos_untuned"].mean()
    out = pd.DataFrame({"qid": np.asarray(questions_qids), "m1_q_cos": np.asarray(q_cos, dtype=np.float32)})
    if q_cos_untuned is not None:
        out["m1_q_cos_untuned"] = np.asarray(q_cos_untuned, dtype=np.float32)
    out = out.merge(agg, left_on="qid", right_index=True, how="left")
    out["n_parts_scored"] = out["n_parts_scored"].fillna(0).astype(int)
    return out


# ======================================================================================
# 8. Coverage-threshold calibration against human labels (Youden's J, FULL vs NONE)
# ======================================================================================
def youden_threshold(scores, is_full):
    """Threshold maximising J = TPR - FPR for FULL (1) vs NONE (0), 'positive' = score >= threshold. Returns dict."""
    from sklearn.metrics import roc_auc_score, roc_curve
    s = np.asarray(scores, dtype=float)
    y = np.asarray(is_full, dtype=int)
    if y.min() == y.max():
        raise ValueError("need both FULL and NONE examples to calibrate a threshold")
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k]) if np.isfinite(thr[k]) else float(s.max() + 1e-6)
    return {"threshold": t, "J": float(j[k]), "sensitivity": float(tpr[k]), "specificity": float(1 - fpr[k]),
            "auc": float(roc_auc_score(y, s))}


def calibrate_threshold(scores_df, labels_df, score_col="m1_cos", n_boot=500, seed=0):
    """Join part scores to human labels (columns part_id,label with FULL/PARTIAL/NONE), pick the threshold
    separating FULL from NONE (Youden's J), report AUC (+ bootstrap 95 % CIs), and where PARTIAL falls.
    Multiple coders per part: majority label, ties dropped."""
    lab = labels_df[["part_id", "label"]].copy()
    lab["label"] = lab["label"].astype(str).str.strip().str.upper()
    lab = lab[lab["label"].isin(["FULL", "PARTIAL", "NONE"])]
    dup_dropped = 0
    if lab["part_id"].duplicated().any():
        def maj(x):
            vc = x.value_counts()
            return vc.index[0] if (len(vc) == 1 or vc.iloc[0] > vc.iloc[1]) else None
        lab = lab.groupby("part_id")["label"].apply(maj).dropna().reset_index()
        dup_dropped = 1
    d = scores_df[["part_id", score_col]].merge(lab, on="part_id", how="inner").dropna(subset=[score_col])
    fn = d[d["label"].isin(["FULL", "NONE"])]
    res = youden_threshold(fn[score_col], (fn["label"] == "FULL").astype(int))
    res.update(score_col=score_col, n_labeled=int(len(d)), n_full=int((d["label"] == "FULL").sum()),
               n_none=int((d["label"] == "NONE").sum()), n_partial=int((d["label"] == "PARTIAL").sum()),
               multi_coder_majority=bool(dup_dropped))
    res["mean_score_by_label"] = {k: float(v) for k, v in d.groupby("label")[score_col].mean().items()}
    res["share_above_threshold_by_label"] = {k: float(v) for k, v in
                                             d.assign(a=d[score_col] >= res["threshold"]).groupby("label")["a"].mean().items()}
    if n_boot:
        rng = np.random.default_rng(seed)
        s, y = fn[score_col].to_numpy(), (fn["label"] == "FULL").to_numpy().astype(int)
        aucs, ths = [], []
        for _ in range(n_boot):
            b = rng.integers(0, len(s), len(s))
            if y[b].min() == y[b].max():
                continue
            r = youden_threshold(s[b], y[b])
            aucs.append(r["auc"])
            ths.append(r["threshold"])
        if aucs:
            res["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
            res["threshold_ci95"] = [float(np.percentile(ths, 2.5)), float(np.percentile(ths, 97.5))]
    return res


def part_coverage(scores_df, threshold, score_col="m1_cos"):
    """part_coverage per question = share of its SCORED parts whose cosine >= threshold."""
    d = scores_df.dropna(subset=[score_col])
    cov = (d[score_col] >= threshold).groupby(d["qid"]).mean().rename("m1_part_coverage")
    return cov.reset_index()


# ======================================================================================
# 9. Model-dependent code (lazy torch / sentence-transformers)
# ======================================================================================
def slug_of(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.rstrip("/").split("/")[-1])


def out_paths(model_id, out_dir):
    suffix = "" if model_id == DEFAULT_MODEL else f"__{slug_of(model_id)}"
    return (os.path.join(out_dir, f"m1_scores{suffix}.parquet"), os.path.join(out_dir, f"m1_question{suffix}.parquet"))


def pick_device(requested=None):
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_sha_of(model):
    try:
        sha = getattr(model[0].auto_model.config, "_commit_hash", None)
        if sha:
            return sha
    except Exception:
        pass
    return "unknown"


def load_st(model_path, device, max_len, half=False):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_path, device=device)
    m.max_seq_length = max_len
    if half and device.startswith("cuda"):
        m.half()
    m.eval()
    return m


class STEncoder:
    """encode(list[str]) -> L2-normalised float32 array. Sorting by length happens inside model.encode."""

    def __init__(self, model, batch_size):
        self.model, self.batch_size = model, batch_size

    def __call__(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


def estimate_run(n_tok_enc, n_tok_train, epochs, n_models=2, tok_per_s=ASSUMED_TOK_PER_S):
    """Seconds / USD. n_tok_enc: tokens in ONE pass over all unique texts (capped at max_len); each pass is run
    for the tuned and the untuned model (n_models).  Training ~ 3x inference cost per token (fwd+bwd) at ~60 %
    efficiency of inference."""
    enc_s = n_models * n_tok_enc / tok_per_s
    train_s = epochs * n_tok_train * 3 / (0.6 * tok_per_s)
    return {"encode_s": enc_s, "train_s": train_s, "total_s": enc_s + train_s,
            "usd": (enc_s + train_s) / 3600 * A100_USD_PER_HOUR}


def fmt_estimate(e, tok_per_s):
    return (f"ESTIMATE (A100 40GB, fp16, assumes {tok_per_s:,} tok/s inference): train ~{e['train_s'] / 60:.0f} min, "
            f"encode (tuned + untuned) ~{e['encode_s'] / 60:.0f} min, total ~{e['total_s'] / 3600:.2f} h "
            f"=> ~${e['usd']:.2f} at ${A100_USD_PER_HOUR}/h.  (Plan for 1.5-2x for model download, I/O, slower seqs.)")


# ---------- data loading ----------
def load_data(parts_path, questions_path, limit_parts=None, split_seed=13):
    parts = pd.read_parquet(parts_path, columns=["part_id", "qid", "part_label", "question_part", "answer_part",
                                                 "answer_missing_for_part"])
    q = pd.read_parquet(questions_path, columns=["qid", "question_text", "answer_text"])
    if limit_parts:
        # deterministic, qid-wise subsample: take qids in hash order until >= limit_parts parts
        qids = parts["qid"].drop_duplicates()
        order = sorted(qids, key=lambda x: _qid_u(x, split_seed + 1))
        counts = parts.groupby("qid").size()
        keep, tot = set(), 0
        for x in order:
            keep.add(x)
            tot += counts[x]
            if tot >= limit_parts:
                break
        parts = parts[parts["qid"].isin(keep)].reset_index(drop=True)
        q = q[q["qid"].isin(keep)]
    # one whole-question row per qid, in the order of first appearance in parts
    q = q.drop_duplicates("qid").set_index("qid").reindex(parts["qid"].drop_duplicates()).reset_index()
    q[["question_text", "answer_text"]] = q[["question_text", "answer_text"]].fillna("")
    return parts.reset_index(drop=True), q


# ---------- training ----------
def train_config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_train(args, parts, split, log=log):
    import torch
    from sentence_transformers import losses

    device = pick_device(args.device)
    model_dir = args.model_dir
    work = os.path.join(args.work_dir, "train")
    os.makedirs(work, exist_ok=True)

    # cleaned pairs (unique raw strings are cleaned once)
    qi, qt, _ = index_texts(parts["question_part"], clean_question, 10 ** 9)
    ai, at, _ = index_texts(parts["answer_part"], clean_answer, 10 ** 9)
    q_arr = np.array(qt + [""], dtype=object)[np.where(qi < 0, len(qt), qi)]
    a_arr = np.array(at + [""], dtype=object)[np.where(ai < 0, len(at), ai)]
    df = pd.DataFrame({"qid": parts["qid"].to_numpy(), "split": split, "q": q_arr, "a": a_arr})
    df.loc[parts["answer_missing_for_part"].to_numpy(), "a"] = ""
    train = build_training_pairs(df, "train", args.min_answer_chars)
    val = build_training_pairs(df, "val", args.min_answer_chars)
    n_all = {"train": int((split == "train").sum()), "val": int((split == "val").sum())}
    log(f"training pairs: {len(train)} of {n_all['train']} train-split parts "
        f"(dropped {n_all['train'] - len(train)}: missing/short answer <{args.min_answer_chars} chars, empty question, dup); "
        f"validation pairs {len(val)} of {n_all['val']}")
    if len(train) < 8:
        raise SystemExit("too few training pairs; raise --train-frac or use more parts")

    cfg = {"model": args.model, "n_train": len(train), "pairs_digest": texts_digest(list(train["q"]) + list(train["a"]))[:16],
           "epochs": args.epochs, "lr": args.lr, "bs": args.train_batch_size, "max_len": args.train_max_len,
           "seed": args.seed, "scale": args.scale, "train_frac": args.train_frac, "val_frac": args.val_frac,
           "q_prefix": args.q_prefix, "d_prefix": args.d_prefix, "min_answer_chars": args.min_answer_chars}
    ft_id = train_config_hash(cfg)
    done_marker = os.path.join(model_dir, "m1_train_done.json")
    if not args.retrain and latest_signature_ok(done_marker, ft_id):
        log(f"fine-tuned model {ft_id} already in {model_dir}; skipping training (--retrain to force)")
        return ft_id

    torch.manual_seed(args.seed)
    model = load_st(args.model, device, args.train_max_len)
    base_sha = model_sha_of(model)
    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    loss_fn = losses.MultipleNegativesRankingLoss(model, scale=args.scale)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    qs = [args.q_prefix + x for x in train["q"]]
    as_ = [args.d_prefix + x for x in train["a"]]
    qid_l = train["qid"].tolist()
    batches0 = make_batches(qid_l, as_, args.train_batch_size, args.seed)
    steps_per_epoch = len(batches0)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    def val_eval(m, max_pairs=3000):
        if len(val) < 20:
            return None
        v = val.sample(min(max_pairs, len(val)), random_state=0)
        m.eval()
        with torch.no_grad():
            e = lambda xs: m.encode([x for x in xs], batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
            qe = e([args.q_prefix + x for x in v["q"]])
            ae = e([args.d_prefix + x for x in v["a"]])
        grp = pd.factorize(v["a"])[0]
        return retrieval_metrics(qe, ae, grp)

    ckpt = os.path.join(work, "ckpt.pt")
    state = {"step": 0}
    if os.path.exists(ckpt) and not args.retrain:
        try:
            st = torch.load(ckpt, map_location=device, weights_only=False)
            if st.get("cfg_id") == ft_id:
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                sched.load_state_dict(st["sched"])
                scaler.load_state_dict(st["scaler"])
                state = {"step": st["step"], "losses": st.get("losses", [])}
                log(f"resuming training from step {state['step']}/{total}")
            else:
                log("checkpoint belongs to a different configuration; starting fresh")
        except Exception as e:  # noqa: BLE001
            log(f"could not load checkpoint ({type(e).__name__}); starting fresh")
    base_val = None
    if state["step"] == 0:
        base_val = val_eval(model)
        log(f"validation retrieval BEFORE tuning: {base_val}")
    losses_log = state.get("losses", [])

    step, t0 = state["step"], time.time()
    model.train()
    for ep in range(args.epochs):
        batches = (batches0 if ep == 0 else make_batches(qid_l, as_, args.train_batch_size, args.seed + ep))[:steps_per_epoch]
        for bi, b in enumerate(batches):
            gstep = ep * steps_per_epoch + bi
            if gstep < step:
                continue
            feats = []
            for texts in ([qs[i] for i in b], [as_[i] for i in b]):
                f = model.tokenize(texts)
                # sentence-transformers >=5.x adds non-tensor entries to the features dict; move only tensors
                feats.append({k: (v.to(device) if hasattr(v, "to") else v) for k, v in f.items()})
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(feats, None)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step = gstep + 1
            losses_log.append(float(loss.detach()))
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                run_steps = step - state["step"]
                eta = (total - step) * el / max(run_steps, 1)
                log(f"  train step {step}/{total} loss {np.mean(losses_log[-25:]):.4f} ETA {eta / 60:.1f} min "
                    f"(~${eta / 3600 * A100_USD_PER_HOUR:.2f})")
            if step % args.ckpt_every == 0 and step < total:
                tmp = ckpt + ".tmp"
                torch.save({"cfg_id": ft_id, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "losses": losses_log}, tmp)
                os.replace(tmp, ckpt)
    model.eval()
    after_val = val_eval(model)
    log(f"validation retrieval AFTER tuning: {after_val}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    report = {"signature": ft_id, "cfg": cfg, "base_model": args.model, "base_sha": base_sha, "steps": step,
              "steps_per_epoch": steps_per_epoch, "final_loss_last25": float(np.mean(losses_log[-25:])),
              "val_before": base_val, "val_after": after_val, "n_train_pairs": len(train), "n_val_pairs": len(val),
              "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _atomic_json(done_marker, report)
    _atomic_json(os.path.join(args.work_dir, "train_report.json"), report)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return ft_id


# ---------- corpus preparation (shared by estimate / score) ----------
def cached_token_lengths(tokenizer, texts, max_len, work_dir, name, digest):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, f"toklen_{name}_{max_len}_{digest[:16]}.npy")
    if os.path.exists(path):
        return np.load(path)
    out = token_lengths(tokenizer, texts, max_len)
    np.save(path + ".tmp.npy", out)
    os.replace(path + ".tmp.npy", path)
    return out


def prepare_corpus(args, parts, questions, split):
    from transformers import AutoTokenizer
    log("preparing texts ...")
    U = build_universes(parts, questions, args.max_len, args.q_prefix, args.d_prefix)
    U["q_dig"], U["a_dig"] = texts_digest(U["q_texts"]), texts_digest(U["a_texts"])
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model)
    q_len = cached_token_lengths(tok, U["q_texts"], args.max_len, args.work_dir, "q", U["q_dig"])
    a_len = cached_token_lengths(tok, U["a_texts"], args.max_len, args.work_dir, "a", U["a_dig"])
    trunc = truncation_report(U, q_len, a_len, args.max_len)
    log(f"unique texts: {len(U['q_texts']):,} query-side, {len(U['a_texts']):,} doc-side "
        f"(of {len(parts) + len(questions):,} rows each); % truncated at {args.max_len} tokens: " +
        ", ".join(f"{k.replace('_pct_truncated', '')} {v:.2f}" for k, v in trunc.items() if k != "max_len"))
    n_tok = int(np.minimum(q_len, args.max_len).sum() + np.minimum(a_len, args.max_len).sum())
    tr = (split == "train") & (U["part_q"] >= 0) & (U["part_a"] >= 0)
    n_tok_train = int(np.minimum(q_len[U["part_q"][tr]], args.train_max_len).sum()
                      + np.minimum(a_len[U["part_a"][tr]], args.train_max_len).sum())
    return U, q_len, a_len, trunc, n_tok, n_tok_train


# ---------- scoring ----------
def run_score(args, parts, questions, split, prepared, log=log):
    U, q_len, a_len, trunc, n_tok, _ = prepared
    ft_id = None
    done_marker = os.path.join(args.model_dir, "m1_train_done.json")
    try:
        with open(done_marker, encoding="utf8") as f:
            ft_id = json.load(f)["signature"]
    except Exception:
        if not args.untuned_only:
            raise SystemExit(f"no fine-tuned model in {args.model_dir}: run --stage train first (or --untuned-only)")
    device = pick_device(args.device)
    os.makedirs(args.work_dir, exist_ok=True)
    base = load_st(args.model, device, args.max_len, half=True)
    base_sha = model_sha_of(base)

    q_order = np.argsort([-len(t) for t in U["q_texts"]], kind="stable")
    a_order = np.argsort([-len(t) for t in U["a_texts"]], kind="stable")
    q_rank = np.empty(len(q_order), dtype=np.int64)
    q_rank[q_order] = np.arange(len(q_order))
    a_rank = np.empty(len(a_order), dtype=np.int64)
    a_rank[a_order] = np.arange(len(a_order))

    def to_rows(idx, rank):
        return np.where(idx >= 0, rank[np.clip(idx, 0, None)], -1)
    rows = {"p_q": to_rows(U["part_q"], q_rank), "p_a": to_rows(U["part_a"], a_rank),
            "k_q": to_rows(U["ques_q"], q_rank), "k_a": to_rows(U["ques_a"], a_rank)}
    q_dig, a_dig = U["q_dig"], U["a_dig"]

    cos = {}
    kinds = [] if args.untuned_only else ["tuned"]
    kinds.append("untuned")
    for kind in kinds:
        cos_path = os.path.join(args.work_dir, f"cos_{kind}.npz")
        ident = f"{args.model}@{base_sha}" + (f"+ft:{ft_id}" if kind == "tuned" else "")
        sig = hashlib.sha1(f"{ident}|{args.max_len}|{q_dig}|{a_dig}".encode()).hexdigest()
        if os.path.exists(cos_path):
            z = np.load(cos_path, allow_pickle=False)
            if str(z["sig"]) == sig:
                log(f"[{kind}] cosines already computed ({cos_path})")
                cos[kind] = (z["part"], z["question"])
                continue
        model = base if kind == "untuned" else load_st(args.model_dir, device, args.max_len, half=True)
        enc = STEncoder(model, args.batch_size)
        dim = model.get_sentence_embedding_dimension()
        chunk = args.batch_size * args.ckpt_batches
        stores = {}
        for side, texts, order, dg in (("q", U["q_texts"], q_order, q_dig), ("a", U["a_texts"], a_order, a_dig)):
            store = EmbedStore(args.work_dir, f"{kind}_{side}", len(texts), dim, f"{sig}|{side}", chunk)
            log(f"[{kind}] encoding {side}-side: {len(texts):,} texts, {store.n_chunks - len(store.done)} of "
                f"{store.n_chunks} chunks to do (checkpoint every {chunk} texts)")
            encode_universe(enc, texts, store, order, args.batch_size, label=f"{kind}/{side}", cost_per_s=True)
            stores[side] = store
        part_cos = pair_cosines(stores["q"].mm, stores["a"].mm, rows["p_q"], rows["p_a"])
        ques_cos = pair_cosines(stores["q"].mm, stores["a"].mm, rows["k_q"], rows["k_a"])
        tmp = cos_path + ".tmp.npz"
        np.savez(tmp, part=part_cos, question=ques_cos, sig=np.array(sig))
        os.replace(tmp, cos_path)
        cos[kind] = (part_cos, ques_cos)
        for s in stores.values():
            del s.mm
            if not args.keep_embeddings:
                for p in (s.path, s.marker):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        if kind == "tuned":
            del model
    nan = np.full(len(parts), np.nan, dtype=np.float32)
    nanq = np.full(len(questions), np.nan, dtype=np.float32)
    p_t, k_t = cos.get("tuned", (nan, nanq))
    p_u, k_u = cos["untuned"]

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scores = pd.DataFrame({
        "part_id": parts["part_id"].to_numpy(), "qid": parts["qid"].to_numpy(),
        "part_label": parts["part_label"].to_numpy(),
        "m1_cos": p_t, "m1_cos_untuned": p_u,
        "m1_valid": ~np.isnan(p_u), "split": split,
        "q_truncated": np.where(U["part_q"] >= 0, (q_len > args.max_len)[np.clip(U["part_q"], 0, None)] | U["q_clipped"][np.clip(U["part_q"], 0, None)], False),
        "a_truncated": np.where(U["part_a"] >= 0, (a_len > args.max_len)[np.clip(U["part_a"], 0, None)] | U["a_clipped"][np.clip(U["part_a"], 0, None)], False),
        "model_id": args.model, "model_sha": base_sha, "ft_id": ft_id or "", "max_seq_len": args.max_len,
        "run_date": run_date})
    qtab = aggregate_question(scores, questions["qid"], k_t, k_u)
    qtab["split"] = pd.Series(split, index=parts["qid"]).groupby(level=0).first().reindex(qtab["qid"]).to_numpy()
    qtab["model_id"], qtab["model_sha"], qtab["ft_id"], qtab["run_date"] = args.model, base_sha, ft_id or "", run_date
    os.makedirs(args.out_dir, exist_ok=True)
    sp, qp = out_paths(args.model, args.out_dir)
    for df, p in ((scores, sp), (qtab, qp)):
        tmp = p + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    manifest = {"model": args.model, "model_sha": base_sha, "ft_id": ft_id, "run_date": run_date, "n_parts": len(scores),
                "n_questions": len(qtab), "n_unique_query_texts": len(U["q_texts"]), "n_unique_doc_texts": len(U["a_texts"]),
                "truncation": trunc, "args": {k: v for k, v in vars(args).items()},
                "n_valid_parts": int(scores["m1_valid"].sum())}
    _atomic_json(sp.replace(".parquet", "_run.json"), manifest)
    v = scores[scores["m1_valid"]]
    log(f"wrote {sp} ({len(scores):,} parts, {len(v):,} valid) and {qp} ({len(qtab):,} questions)")
    if not args.untuned_only:
        for name, d in (("heldout", v[v["split"] == "heldout"]), ("train", v[v["split"] == "train"])):
            log(f"  {name}: mean cos tuned {d['m1_cos'].mean():.3f} untuned {d['m1_cos_untuned'].mean():.3f} (n={len(d):,})")


def run_calibrate(args, log=log):
    sp, qp = out_paths(args.model, args.out_dir)
    scores = pd.read_parquet(sp)
    if not os.path.exists(args.labels):
        raise SystemExit(f"labels file not found: {args.labels} (needs columns part_id,label)")
    labels = pd.read_csv(args.labels)
    col = args.score_col
    res = calibrate_threshold(scores, labels, score_col=col)
    outj = sp.replace(".parquet", f"_calibration_{col}.json")
    _atomic_json(outj, res)
    cov = part_coverage(scores, res["threshold"], col)
    cp = sp.replace("m1_scores", "m1_coverage")
    cov.to_parquet(cp, index=False)
    log(f"threshold {res['threshold']:.4f} (J={res['J']:.3f}, AUC={res['auc']:.3f}, n={res['n_labeled']}) -> {outj}; "
        f"coverage -> {cp}")
    return res


# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["estimate", "pack", "train", "score", "all", "calibrate"], default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF id of the base bi-encoder (robustness: BAAI/bge-base-en-v1.5)")
    ap.add_argument("--batch-size", type=int, default=512, help="ENCODING batch size (fp16, length-sorted)")
    ap.add_argument("--ckpt-batches", type=int, default=100, help="checkpoint embeddings every N encoding batches")
    ap.add_argument("--train-batch-size", type=int, default=64, help="MNRL batch size (= in-batch negatives)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="MNRL similarity scale")
    ap.add_argument("--ckpt-every", type=int, default=300, help="training checkpoint every N steps")
    ap.add_argument("--max-len", type=int, default=512, help="max tokens when scoring (truncation)")
    ap.add_argument("--train-max-len", type=int, default=256, help="max tokens when training")
    ap.add_argument("--train-frac", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-answer-chars", type=int, default=MIN_ANSWER_CHARS)
    ap.add_argument("--q-prefix", default="", help="text prepended to questions (e.g. BGE query instruction)")
    ap.add_argument("--d-prefix", default="", help="text prepended to answers")
    ap.add_argument("--tokenizer-id", default=None, help="tokenizer for the estimate/truncation stats if different from --model "
                    "(e.g. a cached bert-uncased tokenizer when offline; slightly different token counts)")
    ap.add_argument("--tok-per-s", type=int, default=ASSUMED_TOK_PER_S, help="assumed GPU inference tokens/s for the estimate")
    ap.add_argument("--limit-parts", type=int, default=None, help="smoke test: subsample whole qids until >= N parts")
    ap.add_argument("--parts", default=PARTS_PATH)
    ap.add_argument("--questions", default=QUESTIONS_PATH)
    ap.add_argument("--out-dir", default=MEASURES_DIR)
    ap.add_argument("--work-dir", default=None, help="checkpoints / embeddings (default data/measures/m1_work/<slug>)")
    ap.add_argument("--model-dir", default=None, help="fine-tuned model dir (default models/m1/<slug>)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--untuned-only", action="store_true")
    ap.add_argument("--keep-embeddings", action="store_true")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--score-col", default="m1_cos")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    slug = slug_of(args.model)
    args.work_dir = args.work_dir or os.path.join(args.out_dir, "m1_work", slug)
    args.model_dir = args.model_dir or os.path.join(ROOT, "models", "m1", slug)
    if args.stage == "calibrate":
        run_calibrate(args)
        return
    if args.stage == "pack":  # slim input for the GPU box: only the 3 question-level columns M1 reads
        dst = os.path.join(args.out_dir, "m1_input_questions.parquet")
        pd.read_parquet(args.questions, columns=["qid", "question_text", "answer_text"]).to_parquet(dst, index=False, compression="zstd")
        log(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.0f} MB); upload it with data/measures/parts.parquet and "
            f"analysis/m1_embed.py to your PRIVATE box (copyright text)")
        return
    parts, questions = load_data(args.parts, args.questions, args.limit_parts, args.seed)
    split = assign_split(parts["qid"], args.train_frac, args.val_frac, args.seed)
    vc = pd.Series(split).value_counts().to_dict()
    log(f"{len(parts):,} parts / {len(questions):,} questions; part split {vc}")
    prepared = None
    if args.stage in ("estimate", "all", "score"):
        prepared = prepare_corpus(args, parts, questions, split)
        _, _, _, _, n_tok, n_tok_train = prepared
        n_models = 1 if args.untuned_only else 2
        if args.stage == "score":
            n_tok_train, ep = 0, 0
        else:
            ep = args.epochs
        log(fmt_estimate(estimate_run(n_tok, n_tok_train, ep, n_models, args.tok_per_s), args.tok_per_s))
    if args.stage == "estimate":
        return
    if args.stage in ("train", "all"):
        run_train(args, parts, split)
    if args.stage in ("score", "all"):
        run_score(args, parts, questions, split, prepared)


if __name__ == "__main__":
    main()
