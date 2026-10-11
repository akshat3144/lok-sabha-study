"""Which questions get M2 (Gemini) labels: the budget-fitting design fixed on 2026-10-18 (analysis plan §4.3a, paper/analysis_plan.md).

The chosen M2 configuration (gemini-3.8-flash, no rationale) costs ~$624 for all 393,501 questions, over the
$300 credit. The design is therefore:
  - census     : every question of the 17th and 18th Lok Sabha (H2 / H4 are estimated on LS17 ∪ LS18 only);
  - random     : a stratified random sample of the 13th–16th Lok Sabha, fraction 1/6 within each stratum
                 (stratum = Lok Sabha × session × starred/unstarred; n_h = round(N_h / 6), at least 1);
  - gold       : every pilot and main gold-sample question not already drawn (needed for PPI++; these extra
                 questions get m2_weight = 0, i.e. they never enter the population estimates).
m2_weight = N_h / n_h (census: 1). Strata come from the qid itself (lsNN-sNN-S|U-...), so no outcome is used.

    python -I analysis/m2_subsample.py            # writes data/measures/m2_sample.csv and prints a summary
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")
GOLD_FILES = [os.path.join(ROOT, "data", "coding", f) for f in ("pilot_sample.csv", "main_sample.csv")]
OUT = os.path.join(ROOT, "data", "measures", "m2_sample.csv")

CENSUS_LS = (17, 18)
FRACTION = 1 / 6
SEED = 20261018


def parse_qid(q):
    s = q.str.extract(r"^ls(?P<ls>\d+)-s(?P<session>\d+)-(?P<qtype>[SU])-")
    s["ls"] = s["ls"].astype(int)
    s["session"] = s["session"].astype(int)
    return s


def draw(qids, census_ls=CENSUS_LS, fraction=FRACTION, seed=SEED, gold=()):
    d = pd.DataFrame({"qid": sorted(set(qids))})
    d = pd.concat([d, parse_qid(d["qid"])], axis=1)
    if d[["ls", "session", "qtype"]].isna().any().any():
        raise ValueError("unparseable qid(s): " + ", ".join(d.loc[d["qtype"].isna(), "qid"].head(5)))
    d["stratum"] = "ls" + d["ls"].astype(str) + "|s" + d["session"].astype(str) + "|" + d["qtype"]
    d["N_stratum"] = d.groupby("stratum")["qid"].transform("size")
    rng = np.random.default_rng(seed)
    d["role"] = ""
    d.loc[d["ls"].isin(census_ls), "role"] = "census"
    for st, g in d[~d["ls"].isin(census_ls)].groupby("stratum", sort=True):
        n = max(1, int(round(len(g) * fraction)))
        pick = rng.choice(g.index.to_numpy(), size=n, replace=False)
        d.loc[pick, "role"] = "random"
    d["n_stratum"] = d.groupby("stratum")["role"].transform(lambda r: (r != "").sum())
    gold = set(gold)
    d.loc[(d["role"] == "") & d["qid"].isin(gold), "role"] = "gold"
    d["is_gold"] = d["qid"].isin(gold)
    d = d[d["role"] != ""].copy()
    d["m2_weight"] = np.where(d["role"] == "gold", 0.0, d["N_stratum"] / d["n_stratum"])
    d["seed"] = seed
    return d[["qid", "ls", "session", "qtype", "stratum", "role", "is_gold", "N_stratum", "n_stratum", "m2_weight",
              "seed"]].sort_values("qid").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    qids = pd.read_parquet(QUESTIONS, columns=["qid"])["qid"].tolist()
    gold = set()
    for f in GOLD_FILES:
        gold |= set(pd.read_csv(f, usecols=["qid"])["qid"])
    missing = gold - set(qids)
    if missing:
        raise SystemExit(f"{len(missing)} gold qids not in questions_analysis, e.g. {sorted(missing)[:3]}")
    d = draw(qids, gold=gold)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d.to_csv(a.out, index=False)
    s = d.groupby(["ls", "role"]).agg(q=("qid", "size"), w=("m2_weight", "sum")).unstack(fill_value=0)
    print(s.to_string())
    print(f"total {len(d):,} questions ({(d.role == 'gold').sum()} gold-only extras); weighted population "
          f"{d.m2_weight.sum():,.0f} of {len(qids):,} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
