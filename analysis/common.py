"""Shared paths and small helpers for the analysis scripts (no analysis logic here).

All analysis scripts are run as ``python -I -X utf8 analysis/<script>.py``; in isolated mode the script directory is
not on sys.path, so every script does ``sys.path.insert(0, <its own dir>)`` before ``import common``.
"""
import os

import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FINAL = os.path.join(DATA, "final")
MEASURES = os.path.join(DATA, "measures")
SYN_MEASURES = os.path.join(MEASURES, "synthetic")      # synthetic outcomes live ONLY here
CODING = os.path.join(DATA, "coding")
REF = os.path.join(DATA, "ref")
RESULTS = os.path.join(ROOT, "results")

QUESTIONS = os.path.join(FINAL, "questions_analysis.parquet")
ASKERS = os.path.join(FINAL, "question_askers.parquet")
PARTS = os.path.join(MEASURES, "parts.parquet")
MINISTRY_MAP = os.path.join(REF, "ministry_map.csv")

SEED_TOPICS = 20261017
SEED_SYNTHETIC = 7
OUTCOMES = ("real", "synthetic")

SECURITY_GROUP = "security"   # Home Affairs, Defence, External Affairs (ministry_map.csv, fixed before analysis)


def measures_dir(outcome):
    return {"real": MEASURES, "synthetic": SYN_MEASURES}[outcome]


def frame_path(outcome):
    return os.path.join(measures_dir(outcome), f"analysis_frame_{outcome}.parquet")


def outdir(kind, outcome="real"):
    """results/<kind>/ for real outputs; results/synthetic/<kind>/ for synthetic ones (never mixed)."""
    base = RESULTS if outcome == "real" else os.path.join(RESULTS, "synthetic")
    p = os.path.join(base, kind)
    os.makedirs(p, exist_ok=True)
    return p


def df_to_markdown(df, floatfmt="{:,.3f}", index=False):
    """Minimal GitHub-markdown writer (no `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]

    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else floatfmt.format(x)
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        return str(x)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines) + "\n"
