"""M2 source parity: label the FILE-text version of the 300 parity questions with the production M2 configuration
and compare with the production (API-text) labels.

    python -I -X utf8 analysis/parity_m2.py estimate                  # offline cost estimate
    python -I -X utf8 analysis/parity_m2.py label   [--workers 8]     # Gemini calls (sync), resumable -> m2_parity_file.parquet
    python -I -X utf8 analysis/parity_m2.py retest  [--workers 8]     # OPTIONAL noise floor: relabel the API text, same config
    python -I -X utf8 analysis/parity_m2.py analyze                   # kappa / score difference -> results/parity_m2.json

Production configuration (fixed): gemini-3.8-flash, no rationale, default thinking (thinking='auto'), temperature auto,
seed 0, prompt m2p-3 (no-rationale hash). Inputs for the file version differ from production ONLY in `answer_text`
(file extraction instead of API text); question text, question parts, annexure text and flags are the production ones.
Does not modify m2_gemini.py / m2_prompt.py / codebook.py or any existing data file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_gemini as G  # noqa: E402
import m2_prompt as MP  # noqa: E402

ROOT = os.path.dirname(HERE)
MEAS = os.path.join(ROOT, "data", "measures")
PARITY_PARTS = os.path.join(MEAS, "parity_parts.parquet")
PARITY_Q = os.path.join(MEAS, "parity_questions.parquet")
PROD_LABELS = os.path.join(MEAS, "m2_labels.parquet")
FILE_LABELS = os.path.join(MEAS, "m2_parity_file.parquet")
RETEST_LABELS = os.path.join(MEAS, "m2_parity_api_retest.parquet")
OUT_JSON = os.path.join(ROOT, "results", "parity_m2.json")
COST_JSON = os.path.join(MEAS, "m2_parity_cost.json")          # file-version labelling
RETEST_COST_JSON = os.path.join(MEAS, "m2_parity_retest_cost.json")

MODEL = "gemini-3.8-flash"
PROJECT = "lok-sabha-paper-2026"
HUMAN_KAPPA = 0.708
SCORE = {"FULL": 1.0, "PARTIAL": 0.5, "NONE": 0.0}


def prod_cfg():
    """Exactly the production configuration: no rationale, default thinking, auto temperature, seed 0."""
    return G.GenCfg(rationale=False)


def build_store(version):
    """Store for one version ('api' or 'file') of the parity questions, keyed by the BASE qid (production qids)."""
    pp = pd.read_parquet(PARITY_PARTS)
    pq_ = pd.read_parquet(PARITY_Q)
    pp = pp[pp["version"] == version].copy()
    pq_ = pq_[pq_["version"] == version].copy()
    pp["qid"] = pp["qid_base"]
    pq_["qid"] = pq_["qid_base"]
    base = pd.read_parquet(os.path.join(ROOT, "data", "final", "questions_analysis.parquet"),
                           columns=["qid", "annexure_text", "qa_mismatch", "answer_garbled"],
                           filters=[("qid", "in", pq_["qid"].tolist())])
    qa = pq_[["qid", "question_text", "answer_text"]].merge(base, on="qid", how="left")
    return G.Store.from_frames(pp[["qid", "part_label", "question_part"]], qa)


def est_cost(store):
    cfg = prod_cfg()
    e = G.estimate_store(store, MODEL, cfg)
    return e


def run_labels(store, out_path, workers, cap_usd, cfg, cost_json=COST_JSON):
    """Sync labelling with a hard cost guard. Resumable: qids with all-ok rows already in out_path are skipped."""
    client = G.SyncClient(PROJECT, "global")
    done_rows = pd.read_parquet(out_path).to_dict("records") if os.path.exists(out_path) else []
    ok_q = {}
    for r in done_rows:
        ok_q.setdefault(r["qid"], []).append(r["status"] == "ok")
    skip = {q for q, v in ok_q.items() if all(v)}
    kept = [r for r in done_rows if r["qid"] in skip]
    todo = [q for q in store.qids if q not in skip]
    spent = [json.load(open(cost_json))["usd"] if os.path.exists(cost_json) else 0.0]
    lock = threading.Lock()
    stop = threading.Event()
    print(f"{len(skip)} done, {len(todo)} to label, workers={workers}, cap=${cap_usd}", flush=True)

    def usd_of(rows):
        n = len(rows)
        return G.cost_usd(MODEL, sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows), "sync")

    def one(q):
        if stop.is_set():
            return q, None
        rows, raw, att = G.label_unit(client, store.unit(q), MODEL, cfg, retry=True)
        with lock:
            spent[0] += usd_of(rows)
            if spent[0] > cap_usd:
                stop.set()
        return q, rows

    out = list(kept)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (q, rows) in enumerate(ex.map(one, todo), 1):
            if rows:
                out += rows
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  spent ${spent[0]:.3f}  {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(out).pipe(G.to_frame).to_parquet(out_path, index=False)
    df = G.to_frame(out)
    # one more attempt for rows still invalid (label_unit already retried once with the error appended)
    bad = sorted(set(df.loc[df["status"] != "ok", "qid"]))
    if bad and not stop.is_set():
        print(f"second pass on {len(bad)} invalid questions", flush=True)
        keep = df[~df["qid"].isin(bad)].to_dict("records")
        for q in bad:
            rows, raw, att = G.label_unit(client, store.unit(q), MODEL, cfg, retry=True)
            spent[0] += usd_of(rows)
            keep += rows
        df = G.to_frame(keep)
    df.to_parquet(out_path, index=False)
    json.dump({"usd": spent[0], "stopped_on_cap": stop.is_set(), "model": MODEL, "price_mode": "sync (2x batch list price)",
               "input_tokens": float(df["input_tokens"].sum()), "output_tokens": float(df["output_tokens"].sum())},
              open(cost_json, "w"), indent=1)
    print(f"done: {len(df)} part rows, {int((df['status'] != 'ok').sum())} invalid, cost ${spent[0]:.3f} -> {out_path}")
    return df


# ======================================================================================
# analysis
# ======================================================================================
def kappa(x, y, cats=None):
    x, y = pd.Series(list(x)), pd.Series(list(y))
    cats = cats or sorted(set(x) | set(y))
    idx = {c: i for i, c in enumerate(cats)}
    C = np.zeros((len(cats), len(cats)))
    for a, b in zip(x, y):
        C[idx[a], idx[b]] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(1) * C.sum(0)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def boot_kappa(df, a, b, cluster="qid", B=2000, seed=0):
    g = list(df.groupby(cluster).indices.values())
    rng = np.random.default_rng(seed)
    xs, ys = df[a].to_numpy(), df[b].to_numpy()
    out = []
    for _ in range(B):
        idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
        out.append(kappa(xs[idx], ys[idx]))
    out = np.array(out)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def compare(api, other, mask_qids=None, tag=""):
    """api / other: label frames (status ok). Matched on part_id. Returns dict of statistics."""
    m = api.merge(other, on="part_id", suffixes=("_a", "_f"))
    m = m[(m["status_a"] == "ok") & (m["status_f"] == "ok")].copy()
    m["qid"] = m["qid_a"]
    if mask_qids is not None:
        m = m[m["qid"].isin(mask_qids)]
    res = {"n_parts_matched": int(len(m)), "n_questions": int(m["qid"].nunique())}
    res["label_agree"] = float((m["label_a"] == m["label_f"]).mean())
    res["label_kappa"] = kappa(m["label_a"], m["label_f"])
    res["label_kappa_ci"] = boot_kappa(m, "label_a", "label_f")
    res["evasion_agree"] = float((m["evasion_type_a"] == m["evasion_type_f"]).mean())
    res["evasion_kappa"] = kappa(m["evasion_type_a"], m["evasion_type_f"])
    res["evasion_kappa_ci"] = boot_kappa(m, "evasion_type_a", "evasion_type_f")
    nf = m[(m["label_a"] != "FULL") & (m["label_f"] != "FULL")]
    res["evasion_both_nonfull_n"] = int(len(nf))
    res["evasion_both_nonfull_agree"] = float((nf["evasion_type_a"] == nf["evasion_type_f"]).mean()) if len(nf) else None
    res["evasion_both_nonfull_kappa"] = kappa(nf["evasion_type_a"], nf["evasion_type_f"]) if len(nf) > 5 else None
    res["request_kappa"] = kappa(m["request_type_a"], m["request_type_f"])
    # confusion for the label
    res["label_confusion_api_rows"] = {a: {b: int(((m["label_a"] == a) & (m["label_f"] == b)).sum()) for b in SCORE} for a in SCORE}
    # question score
    m["sa"] = m["label_a"].map(SCORE)
    m["sf"] = m["label_f"].map(SCORE)
    q = m.groupby("qid")[["sa", "sf"]].mean()
    d = (q["sf"] - q["sa"]).to_numpy()
    n = len(d)
    from scipy import stats
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    res["qscore_mean_api"] = float(q["sa"].mean())
    res["qscore_mean_file"] = float(q["sf"].mean())
    res["qscore_diff_mean"] = float(d.mean())
    res["qscore_diff_ci95"] = [float(d.mean() - t * se), float(d.mean() + t * se)]
    res["qscore_diff_sd"] = float(d.std(ddof=1))
    res["qscore_questions_identical"] = float((np.abs(d) < 1e-9).mean())
    res["qscore_pearson"] = float(np.corrcoef(q["sa"], q["sf"])[0, 1])
    # part-level mean difference with question-cluster bootstrap
    pd_ = (m["sf"] - m["sa"]).to_numpy()
    g = list(m.groupby("qid").indices.values())
    rng = np.random.default_rng(1)
    bs = [pd_[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])].mean() for _ in range(2000)]
    res["part_score_diff_mean"] = float(pd_.mean())
    res["part_score_diff_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # TOST on question-score difference, margin 0.05 (supplementary; not a pre-specified criterion)
    return res


def analyze():
    api = pd.read_parquet(PROD_LABELS)
    pairs = pd.read_csv(os.path.join(ROOT, "results", "parity_pairs.csv"))
    qids = pairs["qid"].tolist()
    api = api[api["qid"].isin(qids)]
    out = {"n_api_parts": int(len(api)), "n_api_questions": int(api["qid"].nunique())}
    f = pd.read_parquet(FILE_LABELS)
    out["file_status"] = f["status"].value_counts().to_dict()
    out["model_ids"] = sorted(set(api["model_id"]) | set(f["model_id"]))
    out["prompt_hash"] = sorted(set(api["prompt_hash"]) | set(f["prompt_hash"]))
    nonempty = set(pairs.loc[~pairs["file_answer_empty"], "qid"])
    out["all300"] = compare(api, f)
    out["nonempty274"] = compare(api, f, nonempty)
    out["n_nonempty"] = len(nonempty)
    # split by whether the answer string is byte-identical in both versions (identical => pure sampling noise of the LLM)
    q = pd.read_parquet(PARITY_Q)
    a_t = q[q["version"] == "api"].set_index("qid_base")["answer_text"]
    f_t = q[q["version"] == "file"].set_index("qid_base")["answer_text"]
    ident = {k for k in nonempty if a_t[k] == f_t[k]}
    out["byte_identical_questions"] = len(ident)
    out["nonempty_byte_identical"] = compare(api, f, ident)
    out["nonempty_not_identical"] = compare(api, f, nonempty - ident)
    # the 26 image-only files (file text empty)
    out["empty26"] = compare(api, f, set(qids) - nonempty)
    if os.path.exists(RETEST_LABELS):
        r = pd.read_parquet(RETEST_LABELS)
        out["retest_api_vs_api_nonempty"] = compare(api, r, nonempty)
        out["retest_file_vs_retest"] = compare(r, f, nonempty)
    for k, pth in (("cost_file", COST_JSON), ("cost_retest", RETEST_COST_JSON)):
        if os.path.exists(pth):
            out[k] = json.load(open(pth))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["estimate", "label", "retest", "analyze"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap-usd", type=float, default=4.5)
    ap.add_argument("--only-nonempty", action="store_true", help="skip questions whose file text is empty")
    a = ap.parse_args(argv)
    if a.stage == "analyze":
        analyze()
        return
    version = "api" if a.stage == "retest" else "file"
    store = build_store(version)
    if a.only_nonempty:
        pairs = pd.read_csv(os.path.join(ROOT, "results", "parity_pairs.csv"))
        keep = set(pairs.loc[~pairs["file_answer_empty"], "qid"])
        store.qids = [q for q in store.qids if q in keep]
    print(f"{len(store.qids)} questions ({version} text); prompt hash {MP.PROMPT_HASH_NR}; model {MODEL}")
    if a.stage == "estimate":
        e = est_cost(store)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
        return
    cfg = prod_cfg()
    assert cfg.phash() == MP.PROMPT_HASH_NR
    out = RETEST_LABELS if a.stage == "retest" else FILE_LABELS
    run_labels(store, out, a.workers, a.cap_usd, cfg, RETEST_COST_JSON if a.stage == "retest" else COST_JSON)


if __name__ == "__main__":
    main()
