"""Exploratory: label the 400 in-estimate gold questions with gemini-3.1-flash-lite, production prompt/config
(no rationale, default thinking = 'auto', temperature auto, seed 0). Sync client, <= 8 workers, hard cost cap.
    python -I -X utf8 analysis/explore_label_alt.py [--limit N] [--workers 8] [--cap 3.0]
Output: data/measures/m2_gold_alt_3.1fl.parquet (+ m2_gold_alt_3.1fl_cost.json). Resumable. New files only.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_gemini as G  # noqa: E402

ROOT = os.path.dirname(HERE)
MEAS = os.path.join(ROOT, "data", "measures")
OUT = os.path.join(MEAS, "m2_gold_alt_3.1fl.parquet")
COST = os.path.join(MEAS, "m2_gold_alt_3.1fl_cost.json")
MODEL = "gemini-3.1-flash-lite"
PROJECT = "lok-sabha-paper-2026"


class Sel:  # selection namespace for Store.load
    qids_file = None; ls = None; sample = None; seed = 1; limit = None; skip_no_answer = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=float, default=3.0)
    a = ap.parse_args()
    ms = pd.read_csv(os.path.join(ROOT, "data", "coding", "main_sample.csv"))
    qids = sorted(ms.loc[ms["in_estimate"].astype(bool), "qid"].unique())
    print(len(qids), "in-estimate gold questions")
    tmp = os.path.join(MEAS, "_tmp_gold_alt_qids.csv")
    pd.DataFrame({"qid": qids}).to_csv(tmp, index=False)
    sel = Sel(); sel.qids_file = tmp
    store = G.Store.load(sel)
    os.remove(tmp)
    print(len(store.qids), "in store; dropped", store.n_dropped)
    cfg = G.GenCfg(rationale=False)              # == production (cfg hash 0adca7)
    est = G.estimate_store(store, MODEL, cfg)
    print("offline estimate:", est)
    if a.limit:
        store.qids = store.qids[:a.limit]
    client = G.SyncClient(PROJECT, "global")
    done = pd.read_parquet(OUT).to_dict("records") if os.path.exists(OUT) else []
    okq = {}
    for r in done:
        okq.setdefault(r["qid"], []).append(r["status"] == "ok")
    skip = {q for q, v in okq.items() if all(v)}
    kept = [r for r in done if r["qid"] in skip]
    todo = [q for q in store.qids if q not in skip]
    spent = [json.load(open(COST))["usd"] if os.path.exists(COST) else 0.0]
    lock, stop = threading.Lock(), threading.Event()
    print(f"{len(skip)} done, {len(todo)} todo, workers={a.workers}, cap=${a.cap}, spent so far ${spent[0]:.4f}", flush=True)

    def usd(rows):
        return G.cost_usd(MODEL, sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows), "sync")

    def one(q):
        if stop.is_set():
            return q, None
        rows, raw, att = G.label_unit(client, store.unit(q), MODEL, cfg, retry=True)
        with lock:
            spent[0] += usd(rows)
            if spent[0] > a.cap:
                stop.set()
        return q, rows

    out = list(kept); t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(a.workers, 8)) as ex:
        for i, (q, rows) in enumerate(ex.map(one, todo), 1):
            if rows:
                out += rows
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} spent ${spent[0]:.4f} {time.time()-t0:.0f}s", flush=True)
                G.to_frame(out).to_parquet(OUT, index=False)
    df = G.to_frame(out)
    df.to_parquet(OUT, index=False)
    json.dump({"usd": spent[0], "stopped_on_cap": stop.is_set(), "model": MODEL,
               "price_mode": "sync = 2x batch list price (BATCH_PRICES in m2_gemini.py)",
               "input_tokens": float(df["input_tokens"].sum()), "output_tokens": float(df["output_tokens"].sum()),
               "n_part_rows": int(len(df)), "n_invalid": int((df["status"] != "ok").sum())}, open(COST, "w"), indent=1)
    print(f"done {len(df)} part rows, invalid {(df['status']!='ok').sum()}, cost ${spent[0]:.4f}")


if __name__ == "__main__":
    main()
