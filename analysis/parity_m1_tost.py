"""M1 source parity: TOST (+/-0.03) on the per-question M1 difference (file - api), from the two separately scored
versions (scratch dirs written by m1_embed --stage score). Re-uses parity_check's statistics. Writes results/parity_m1.json."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import parity_check as pc  # noqa: E402

def load(d):
    s = pd.read_parquet(os.path.join(d, "m1_scores.parquet")); q = pd.read_parquet(os.path.join(d, "m1_question.parquet"))
    for df in (s, q):
        df["qid_base"] = df["qid"].str.rsplit("@", n=1).str[0]
    return s, q

def main(d_api, d_file, out_json, margin=0.03):
    sa, qa = load(d_api); sf, qf = load(d_file)
    pairs = pd.read_csv(pc.PAIRS_CSV); nonempty = set(pairs.loc[~pairs["file_answer_empty"], "qid"])
    res = {}
    q = qa.merge(qf, on="qid_base", suffixes=("_api", "_file"))
    p = sa.merge(sf, on=["qid_base", "part_label"], suffixes=("_api", "_file"))
    for scope, keep in (("all300", None), ("nonempty274", nonempty)):
        qq = q if keep is None else q[q["qid_base"].isin(keep)]
        pp = p if keep is None else p[p["qid_base"].isin(keep)]
        r = {}
        for name, tab, c in (("question_part_mean", qq, "m1_part_mean"), ("question_whole_text", qq, "m1_q_cos"),
                             ("question_part_mean_untuned", qq, "m1_part_mean_untuned"), ("part_level", pp, "m1_cos")):
            a, b = tab[c + "_api"].to_numpy(float), tab[c + "_file"].to_numpy(float)
            e = {"tests": pc.paired_tests(a, b, margin), "bland_altman": pc.bland_altman(a, b)}
            if name == "part_level":
                e["cluster_bootstrap"] = pc.cluster_bootstrap_mean_diff(tab, c + "_api", c + "_file")
            r[name] = e
            # 90% CI for TOST (equivalent to two one-sided tests at 5%)
            ok = ~(np.isnan(a) | np.isnan(b)); d = (b - a)[ok]; from scipy import stats
            se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(0.95, len(d) - 1)
            e["ci90_mean_diff"] = [float(d.mean() - t * se), float(d.mean() + t * se)]
            e["sd_pooled_q"] = float(np.nanstd(a, ddof=1))
        res[scope] = r
    # M1 level for context: spread of M1 between FULL / NONE M2 labels is not computed here
    json.dump(res, open(out_json, "w"), indent=1)
    return res

if __name__ == "__main__":
    S = sys.argv[1]
    r = main(os.path.join(S, "out_api"), os.path.join(S, "out_file"), os.path.join(pc.RESULTS_DIR, "parity_m1.json"))
    for sc, v in r.items():
        for k, e in v.items():
            t = e["tests"]
            print(sc, k, "n", t["n"], "diff %.4f" % t["mean_diff"], "ci90", [round(x, 4) for x in e["ci90_mean_diff"]],
                  "tostp %.2g" % t.get("tost_p", float("nan")), "equiv", t.get("equivalent"), "sd_diff %.4f" % e["bland_altman"]["sd_diff"],
                  "loa", [round(x, 3) for x in e["bland_altman"]["loa"]], "ccc %.4f" % t.get("ccc", float("nan")), "pearson %.4f" % t.get("pearson", float("nan")))
