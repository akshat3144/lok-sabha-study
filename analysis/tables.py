"""Descriptive corpus tables (REAL data; no outcome needed).

    python -I -X utf8 analysis/tables.py

Writes results/tables/corpus_by_ls.{csv,md}, corpus_by_period.{csv,md}, corpus_by_ministry_group.{csv,md}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

def load_questions():
    cols = ["qid", "ls", "session", "date", "qtype", "ministry", "n_askers", "n_question_parts", "asker_bloc",
            "any_opposition", "govt_period", "majority_type", "answer_missing", "statement_laid", "grouped_answer",
            "all_parts_covered", "has_annexure", "source_a", "question_len", "answer_len"]
    q = pd.read_parquet(C.QUESTIONS, columns=cols)
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    return q


def part_counts():
    """parts per question from data/measures/parts.parquet (falls back to max(n_question_parts, 1))."""
    if os.path.exists(C.PARTS):
        p = pd.read_parquet(C.PARTS, columns=["qid"])
        return p.groupby("qid").size().rename("n_parts_table")
    return None


def mp_counts():
    a = pd.read_parquet(C.ASKERS, columns=["qid", "ls", "mpsno"])
    return a.dropna(subset=["mpsno"]).groupby("ls").mpsno.nunique()


def summarise(g, n_parts, n_mps=None):
    n = len(g)
    pc = n_parts.reindex(g["qid"]).fillna(1).sum() if n_parts is not None else g["n_question_parts"].clip(lower=1).sum()
    res = {
        "Sessions": g["session"].nunique(),
        "Questions": n,
        "Parts": int(pc),
        "Parts per question": pc / n,
        "% starred": 100 * (g["qtype"] == "starred").mean(),
        "% any opposition asker": 100 * g["any_opposition"].eq(True).mean(),
        "% opposition-only askers": 100 * (g["asker_bloc"] == "opposition").mean(),
        "% government-only askers": 100 * (g["asker_bloc"] == "government").mean(),
        "% answer text present": 100 * (1 - g["answer_missing"].astype(bool).mean()),
        "% 'statement laid' answers": 100 * g["statement_laid"].astype(bool).mean(),
        "% grouped answers": 100 * g["grouped_answer"].astype(bool).mean(),
        "% with annexure": 100 * g["has_annexure"].astype(bool).mean(),
        "% answer from docx/pdf": 100 * g["source_a"].isin(["docx", "pdf"]).mean(),
        "Median question chars": g["question_len"].median(),
        "Median answer chars": g["answer_len"].median(),
    }
    if n_mps is not None:
        res["Distinct asking MPs"] = int(n_mps)
    return res


def by_ls(q, n_parts, mps):
    rows = []
    for ls, g in q.groupby("ls"):
        r = {"Lok Sabha": f"{ls}th",
             "Period": " / ".join(sorted(g["govt_period"].dropna().unique())),
             "Majority type": " / ".join(sorted(g["majority_type"].dropna().unique())),
             "First question": g["date"].min().date().isoformat(), "Last question": g["date"].max().date().isoformat()}
        r.update(summarise(g, n_parts, mps.get(ls)))
        rows.append(r)
    tot = {"Lok Sabha": "Total", "Period": "", "Majority type": "",
           "First question": q["date"].min().date().isoformat(), "Last question": q["date"].max().date().isoformat()}
    tot.update(summarise(q, n_parts, mps.sum()))
    tot["Sessions"] = int(q.groupby(["ls", "session"]).ngroups)
    tot["Distinct asking MPs"] = int(pd.read_parquet(C.ASKERS, columns=["mpsno"]).mpsno.nunique())
    rows.append(tot)
    return pd.DataFrame(rows)


def by_majority(q, n_parts):
    rows = []
    for mt, g in q.dropna(subset=["majority_type"]).groupby("majority_type"):
        r = {"Majority type": mt}
        r.update(summarise(g, n_parts))
        rows.append(r)
    return pd.DataFrame(rows)


def by_group(q, n_parts):
    mm = pd.read_csv(C.MINISTRY_MAP)[["ministry", "ministry_group"]].drop_duplicates("ministry")
    q = q.merge(mm, on="ministry", how="left")
    q["ministry_group"] = q["ministry_group"].fillna("unmapped")
    rows = []
    for gname, g in q.groupby("ministry_group"):
        r = {"Ministry group": gname}
        r.update(summarise(g, n_parts))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("Questions", ascending=False)


def write(df, stem, title):
    d = C.outdir("tables")
    df.to_csv(os.path.join(d, stem + ".csv"), index=False)
    with open(os.path.join(d, stem + ".md"), "w", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n")
        fh.write(C.df_to_markdown(df, floatfmt="{:,.1f}"))
    return os.path.join(d, stem + ".md")


def main():
    q = load_questions()
    n_parts = part_counts()
    mps = mp_counts()
    t1 = by_ls(q, n_parts, mps)
    write(t1, "corpus_by_ls", "Table 1. Corpus by Lok Sabha (real data)")
    write(by_majority(q, n_parts), "corpus_by_majority_type", "Corpus by majority type")
    write(by_group(q, n_parts), "corpus_by_ministry_group", "Corpus by ministry group")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t1.T.to_string())


if __name__ == "__main__":
    main()
