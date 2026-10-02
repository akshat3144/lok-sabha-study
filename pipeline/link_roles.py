"""Link every question to its askers' party at election and their role relative to the Union
government ON THE QUESTION DATE, plus the government period. Writes the analysis-ready table.

    python -I -X utf8 pipeline/link_roles.py

Inputs : data/clean/questions.parquet, data/ref/mp_party.csv, data/ref/coalitions.csv (via
         data/ref/coalition_lookup.py), data/ref/govt_periods.csv
Outputs: data/final/questions_analysis.parquet   (one row per question)
         data/final/question_askers.parquet      (one row per question x asker)
         data/final/link_report.json             (coverage numbers)
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(DATA, "ref"))
from coalition_lookup import role, needs_manual  # noqa: E402

OUT = os.path.join(DATA, "final")
os.makedirs(OUT, exist_ok=True)


def norm(s):
    return " ".join(str(s).split()).upper()


def main():
    q = pd.read_parquet(os.path.join(DATA, "clean", "questions.parquet"))
    mp = pd.read_csv(os.path.join(DATA, "ref", "mp_party.csv"), low_memory=False)
    gp = pd.read_csv(os.path.join(DATA, "ref", "govt_periods.csv"))
    gp["start"] = pd.to_datetime(gp["start_date"])
    gp["end"] = pd.to_datetime(gp["end_date"]).fillna(pd.Timestamp("2100-01-01"))

    mp["key"] = mp["ls"].astype(int).astype(str) + "|" + mp["asker_name"].map(norm)
    mp = mp.drop_duplicates("key")
    party = mp.set_index("key")[["party_at_election", "party_short", "mpsno", "canonical_name",
                                 "constituency_at_election", "election_type", "needs_review"]].to_dict("index")

    q["date_ts"] = pd.to_datetime(q["date"], errors="coerce")
    # government period by date
    per = pd.Series(index=q.index, dtype="object")
    maj = pd.Series(index=q.index, dtype="object")
    for _, g in gp.iterrows():
        m = (q["date_ts"] >= g["start"]) & (q["date_ts"] <= g["end"])
        per[m], maj[m] = g["period_id"], g["majority_type"]
    q["govt_period"], q["majority_type"] = per, maj

    rows, role_cache = [], {}
    for r in q[["qid", "ls", "date", "members"]].itertuples(index=False):
        members = [m for m in (list(r.members) if r.members is not None else []) if m and str(m).strip()]
        for m in members:
            p = party.get(f"{int(r.ls)}|{norm(m)}")
            rec = {"qid": r.qid, "ls": r.ls, "date": r.date, "asker": m}
            if not isinstance(r.date, str) or len(r.date) < 10:  # no date -> role undefined
                rec.update(party_at_election=None if p is None else p["party_at_election"],
                           party_short=None if p is None else p["party_short"], role=None, role_strict=None,
                           party_resolved=p is not None, needs_review=True, split_party_window=False)
            elif p is None or pd.isna(p["party_at_election"]):
                rec.update(party_at_election=None, party_short=None, role=None, role_strict=None,
                           party_resolved=False, needs_review=True, split_party_window=False)
            else:
                pn = p["party_at_election"]
                ck = (pn, r.date, p["canonical_name"], p["constituency_at_election"])
                if ck not in role_cache:
                    role_cache[ck] = (role(pn, r.date, mp=p["canonical_name"], pc=p["constituency_at_election"]),
                                      role(pn, r.date, mp=p["canonical_name"], pc=p["constituency_at_election"],
                                           min_conf="medium"),
                                      bool(needs_manual(pn, r.date)))
                rl, rs, nm = role_cache[ck]
                rec.update(party_at_election=pn, party_short=p["party_short"], mpsno=p["mpsno"],
                           election_type=p["election_type"], role=rl, role_strict=rs,
                           party_resolved=True, needs_review=bool(p["needs_review"]), split_party_window=nm)
            rows.append(rec)
    a = pd.DataFrame(rows)
    a.to_parquet(os.path.join(OUT, "question_askers.parquet"), index=False)

    def bloc(roles):
        rs = [x for x in roles if isinstance(x, str)]
        if not rs:
            return "unresolved"
        s = set(rs)
        if s == {"government"}:
            return "government"
        if s == {"opposition"}:
            return "opposition"
        if s == {"outside_support"}:
            return "outside_support"
        return "mixed"

    g = a.groupby("qid")
    agg = pd.DataFrame({
        "n_askers": g.size(),
        "n_askers_resolved": g["party_resolved"].sum(),
        "n_gov": g["role"].apply(lambda s: (s == "government").sum()),
        "n_opp": g["role"].apply(lambda s: (s == "opposition").sum()),
        "n_support": g["role"].apply(lambda s: (s == "outside_support").sum()),
        "asker_bloc": g["role"].apply(bloc),
        "asker_bloc_strict": g["role_strict"].apply(bloc),
        "asker_parties": g["party_short"].apply(lambda s: sorted({x for x in s if isinstance(x, str)})),
        "any_needs_review": g["needs_review"].any(),
        "any_split_party_window": g["split_party_window"].any(),
    }).reset_index()
    agg["any_opposition"] = agg["n_opp"] > 0
    out = q.drop(columns=["date_ts"]).merge(agg, on="qid", how="left")
    out["asker_bloc"] = out["asker_bloc"].fillna("unresolved")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_ocr import apply_ocr  # scanned-image answers filled from Gemini OCR text
    out = apply_ocr(out)
    out.to_parquet(os.path.join(OUT, "questions_analysis.parquet"), index=False)

    rep = {"questions": int(len(out)), "question_x_asker_rows": int(len(a)),
           "asker_party_resolved_pct": round(100 * a["party_resolved"].mean(), 2),
           "questions_all_askers_resolved_pct": round(100 * (out["n_askers_resolved"] == out["n_askers"]).mean(), 2),
           "questions_no_govt_period": int(out["govt_period"].isna().sum()),
           "by_ls": {}}
    for ls, d in out.groupby("ls"):
        rep["by_ls"][int(ls)] = {
            "questions": int(len(d)),
            "all_askers_resolved_pct": round(100 * (d["n_askers_resolved"] == d["n_askers"]).mean(), 2),
            "asker_bloc": {k: int(v) for k, v in Counter(d["asker_bloc"]).items()},
            "govt_period": {k: int(v) for k, v in Counter(d["govt_period"].fillna("NONE")).items()},
        }
    json.dump(rep, open(os.path.join(OUT, "link_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
