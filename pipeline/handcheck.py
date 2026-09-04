"""Independent spot check of mp_party_members.csv: for a random sample of resolved MPs, compare the party with the
'Members of Parliament' table of the MP's Wikipedia constituency page (a source independent of the ECI csv files and of
the per-Lok-Sabha Wikipedia member lists used to build the table).

    python -I pipeline/handcheck.py [--n 20] [--seed 7]    -> data/ref/mp_party_handcheck.csv
Rows where the page has no parsable entry are reported as 'no-evidence' and are NOT counted as errors or successes.
"""
import argparse
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
from lxml import html as lhtml

import parties
import ref_build as R
from namematch import name_sim


def page_rows(const):
    h = R.wp_const_html(f"{const} Lok Sabha constituency")
    if not h:
        return []
    out = []
    for t in lhtml.fromstring(h).xpath('//table[contains(@class,"wikitable")]'):
        g = R._expand(t)
        if len(g) < 3:
            continue
        hdr = [c.text_content().strip().lower() for c in g[0]]
        yi = next((i for i, x in enumerate(hdr) if x.startswith("year") or x == "election"), None)
        mi = next((i for i, x in enumerate(hdr) if x.startswith("member") or x in ("mp", "name", "winner")), None)
        pi = next((i for i, x in enumerate(hdr) if "party" in x), None)
        if None in (yi, mi, pi):
            continue
        for row in g[1:]:
            if len(row) <= max(yi, mi, pi):
                continue
            p = ""
            for c in row[pi:pi + 2]:
                tx = R._text(c).strip()
                if tx:
                    p = tx
            out.append((R._text(row[yi]).strip(), R._text(row[mi]).strip(), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    m = pd.read_csv(os.path.join(R.REF, "mp_party_members.csv"), keep_default_na=False)
    m = m[(m.party_at_election != "") & m.election_type.isin(["general", "bye"])]
    rng = random.Random(a.seed)
    pools = [list(m[m.election_type == "bye"].index), list(m[m.election_type == "general"].index)]
    for p in pools:
        rng.shuffle(p)
    want = [a.n // 2, a.n - a.n // 2]            # half by-election winners, half general-election winners
    res = []
    for pool, k in zip(pools, want):
        got = 0
        for i in pool:
            if got >= k:
                break
            r = m.loc[i]
            rows = page_rows(r.constituency_at_election)
            hit = [(name_sim(r.canonical_name, nm), y, nm, p) for y, nm, p in rows
                   if name_sim(r.canonical_name, nm) >= 80 and p]
            if not hit:
                continue                   # not checkable from this page; draw another MP
            hit.sort(reverse=True)
            # a person can appear for several terms with different parties: prefer rows in this LS's year window
            lo, hi = R.LS_SPAN[int(r.ls)]
            inwin = [h for h in hit if any(lo <= int(y) <= hi for y in re.findall(r"(?:19|20)\d{2}", h[1]))] or hit
            s, y, nm, p = inwin[0]
            indep = parties.canonical(p)[0]
            res.append(dict(ls=r.ls, mp=r.canonical_name, constituency=r.constituency_at_election,
                            our_party=r.party_at_election, our_type=r.election_type, our_source=r.source[:60],
                            wiki_page_row=f"{y} | {nm} | {p}", wiki_party_canonical=indep,
                            agree=indep == r.party_at_election))
            got += 1
    d = pd.DataFrame(res)
    d.to_csv(os.path.join(R.REF, "mp_party_handcheck.csv"), index=False)
    print(d[["ls", "mp", "constituency", "our_party", "wiki_party_canonical", "agree"]].to_string())
    print(f"checked {len(d)}; disagreements: {(~d.agree).sum()}")


if __name__ == "__main__":
    main()
