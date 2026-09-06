"""MP -> party-at-the-time reference table for Lok Sabha Question Hour askers (13th-18th Lok Sabha).

    python -I pipeline/ref_build.py        # (once, or with --refresh-wiki) builds data/ref/seat_terms.csv
    python -I pipeline/mp_party.py         # re-runnable on the growing data/raw/api; applies data/ref/mp_party_manual.csv
    python -I pipeline/mp_party.py --validate   # also writes data/ref/mp_party_validation.txt (+ hand-check sample)

Inputs  (read-only): data/raw/api/ls{N}/*.json.gz (question records), data/ref/source/mem{N}.json (sansad.in member
        lists: identity + state only; their party/constituency are the MP's LATEST values, never used as the answer),
        data/ref/seat_terms.csv (who was elected where/for which party: ECI general results + by-elections).
Outputs (data/ref/): mp_party.csv, mp_party_review.csv, mp_party_members.csv (one row per MP per Lok Sabha),
        party_names.csv, [mp_party_validation.txt].  Optional input: mp_party_manual.csv.

Party = party AT ELECTION (general election or by-election), in ECI full-name spelling + a short code.  It does not
change if the MP later defects.

Sources and licences
  * ECI Statistical Report on General Elections 1999 (13th LS), Vol. I -- public document of the Election Commission
    of India, via the Wayback Machine copy of eci.nic.in.
  * ECI general-election results 2004-2024 as published on the India Data Portal (ODC-BY 1.0), staged in source/.
  * Wikipedia (CC BY-SA 4.0): "List of members of the Nth Lok Sabha", "By-elections to the 18th Lok Sabha",
    constituency pages (by-election result boxes).
  * sansad.in member lists (identity/state only).
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
from rapidfuzz import fuzz, process

import parties
from linking import assign_all, score_candidates, classify, WEAK
from namematch import name_sim, state_key, const_key, tokens, pkey, is_subset

ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "data", "ref")
SRC = os.path.join(REF, "source")
RAW = os.path.join(ROOT, "data", "raw", "api")
LSES = (13, 14, 15, 16, 17, 18)

OFFICIAL_MIN = 85          # asker -> official-list name similarity needed for an automatic identity
OUT_COLS = ["ls", "asker_name", "mpsno", "canonical_name", "state_at_election", "constituency_at_election",
            "party_at_election", "party_short", "election_type", "election_date", "source", "match_score",
            "needs_review", "n_questions", "review_reason"]


# ------------------------------------------------------------------------------------------------ inputs
def read_askers(ls):
    """-> Counter {asker string as it appears: number of question records}, plus list of per-record asker tuples"""
    cnt, recs = Counter(), []
    for f in sorted(glob.glob(os.path.join(RAW, f"ls{ls}", "*.json.gz"))):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue                                   # partial / being written
        for blk in data:
            for q in blk.get("listOfQuestions") or []:
                ms = tuple(m for m in (q.get("member") or []) if m and m.strip())
                recs.append(ms)
                for m in set(ms):
                    cnt[m] += 1
    return cnt, recs


def load_official(ls):
    return json.load(open(os.path.join(SRC, f"mem{ls}.json"), encoding="utf-8"))["membersDtoList"]


def load_terms():
    t = pd.read_csv(os.path.join(REF, "seat_terms.csv"), keep_default_na=False)
    return {ls: t[t.ls == ls].to_dict("records") for ls in LSES}


def load_manual():
    p = os.path.join(REF, "mp_party_manual.csv")
    if not os.path.exists(p):
        return {}
    m = pd.read_csv(p, keep_default_na=False, dtype=str)
    return {(int(r["ls"]), r["asker_name"].strip()): r for r in m.to_dict("records") if r.get("ls") and r.get("asker_name")}


# ------------------------------------------------------------------------------------------------ members
def build_members(ls, terms):
    """link every official-list member of this LS to a seat-term row -> list of dict (one per member)"""
    off = load_official(ls)
    rows = []
    reg = [m for m in off if "omin" not in (m["constName"] or "")]
    links = assign_all([dict(name=m["mpFirstLastName"], state=m["stateName"], const=m["constName"],
                             alt=(m.get("mpLastFirstName") or "",)) for m in reg], terms)
    link_of = {id(m): l for m, l in zip(reg, links)}
    for m in off:
        base = dict(ls=ls, mpsno=m["mpsno"], canonical_name=m["mpFirstLastName"], official_state=m["stateName"],
                    official_const=m["constName"] or "", official_party_now=m.get("partyFname") or "",
                    official_party_now_short=m.get("partySname") or "", status=m.get("status") or "")
        if "omin" in (m["constName"] or ""):
            rows.append({**base, "state_at_election": m["stateName"], "constituency_at_election": m["constName"],
                         "party_at_election": "Nominated (no party)", "party_short": "NOM", "election_type": "nominated",
                         "election_date": "", "source": "sansad.in member list (nominated Anglo-Indian member)",
                         "match_score": 100.0, "needs_review": False, "review_reason": "", "term_idx": None})
            continue
        l = link_of[id(m)]
        if l["idx"] is None:
            sug = [f'{terms[c[4]]["member_name"]} | {terms[c[4]]["constituency"]} | {terms[c[4]]["state"]} | '
                   f'{terms[c[4]]["party_short"]} | {terms[c[4]]["election_type"]} | {c[1]:.0f}' for c in l["cands"][:3]]
            rows.append({**base, "state_at_election": "", "constituency_at_election": "", "party_at_election": "",
                         "party_short": "", "election_type": "", "election_date": "", "source": "",
                         "match_score": round(l["ns"], 1), "needs_review": True,
                         "review_reason": "no general/by-election row found for this member (probably a by-election "
                                          "winner missing from the sources)", "term_idx": None,
                         "suggestions": " || ".join(sug)})
            continue
        t = terms[l["idx"]]
        reason = []
        c0 = l["cands"][0] if l["cands"] else (0, 0, False, 0, None, False)
        sub, s_ok, cs = c0[5], c0[2], c0[3]
        seat_subset = sub and s_ok and cs >= 85        # same state, same constituency, name contained in the other
        if l["level"] == "medium" and l["ns"] < 80 and not seat_subset:
            reason.append(f"weak name match to election record '{t['member_name']}' ({l['ns']:.0f})")
        if l["level"] == "leftover-seat" and l["ns"] < 65:
            reason.append(f"same constituency as election record '{t['member_name']}' but names differ ({l['ns']:.0f})")
        if l["level"] == "leftover" and l["ns"] < 85 and not sub:
            reason.append(f"matched by name only to '{t['member_name']}' in {t['constituency']} ({l['ns']:.0f})")
        if l["ambiguous"]:
            reason.append("another election record matches almost equally well")
        rows.append({**base, "state_at_election": t["state"], "constituency_at_election": t["constituency"],
                     "party_at_election": t["party_name"], "party_short": t["party_short"],
                     "election_type": t["election_type"], "election_date": t["election_date"], "source": t["source"],
                     "match_score": round(l["ns"], 1), "needs_review": bool(reason), "review_reason": "; ".join(reason),
                     "term_idx": l["idx"], "term_name": t["member_name"], "link_level": l["level"],
                     "ambiguous": l["ambiguous"]})
    return rows


# ------------------------------------------------------------------------------------------------ askers
def match_asker(ls, name, members, terms, mem_tokens):
    """asker string -> (member dict | None, score, ambiguous, how, top3).

    Accepted automatically when the name similarity is >= OFFICIAL_MIN, or >= 72 with the asker's name fully
    contained in the official name (or vice versa) and no other member within 8 points ('Supriya Sadanand Sule' ->
    'Supriya Sule').  Sub-85 acceptances are flagged for review by the caller."""
    q = " ".join(sorted(pkey(t) for t in tokens(name)))
    cand = process.extract(q, mem_tokens["keys"], scorer=fuzz.token_set_ratio, limit=15) if q else []
    best = []
    for _, _, i in cand:
        m = members[i]
        s = max(name_sim(name, m["canonical_name"]), name_sim(name, m.get("alt_name") or ""))
        sub = is_subset(name, m["canonical_name"])
        best.append((s, i, sub))
    best.sort(reverse=True)
    top3 = [(s, i) for s, i, _ in best[:3]]
    if not best:
        return None, 0.0, False, "no-official-match", top3
    s0, i0, sub0 = best[0]
    others = [b for b in best[1:] if members[b[1]]["mpsno"] != members[i0]["mpsno"]]
    s1 = others[0][0] if others else 0.0
    amb = s1 >= 75 and s1 >= s0 - 3
    ok = s0 >= OFFICIAL_MIN or (s0 >= 72 and sub0 and s0 - s1 >= 8)
    return (members[i0] if ok else None), s0, amb, "official" if ok else "below-threshold", top3


def resolve_ls(ls, terms, askers_cnt, manual):
    members = build_members(ls, terms[ls])
    off = {m["mpsno"]: m for m in load_official(ls)}
    for m in members:
        o = off[m["mpsno"]]
        m["alt_name"] = o.get("mpLastFirstName") or ""
    mem_tokens = {"keys": [" ".join(sorted(pkey(t) for t in tokens(m["canonical_name"]))) for m in members]}
    out = []
    for name, n in sorted(askers_cnt.items()):
        row = dict(ls=ls, asker_name=name, n_questions=n)
        mk = manual.get((ls, name.strip()))
        m, score, amb, how, top = match_asker(ls, name, members, terms[ls], mem_tokens)
        if m is not None:
            row.update({k: m[k] for k in ("mpsno", "canonical_name", "state_at_election", "constituency_at_election",
                                           "party_at_election", "party_short", "election_type", "election_date",
                                           "source")})
            row["match_score"] = round(min(score, m["match_score"] if m["party_at_election"] else score), 1)
            reasons = []
            if m["needs_review"]:
                reasons.append(m["review_reason"])
            if amb:
                reasons.append("asker name matches two different official members almost equally")
            if score < 78:
                reasons.append(f"asker name only loosely matches official name '{m['canonical_name']}' ({score:.0f})")
            if not m["party_at_election"]:
                reasons.append("party unresolved (see members file)")
            row["needs_review"] = bool(reasons)
            row["review_reason"] = "; ".join(r for r in reasons if r)
            row["_amb"] = amb
        else:
            # not in the official list: try the election rows directly (nominated members, odd spellings)
            l = assign_all([dict(name=name, state="", const="")], terms[ls])[0]
            if l["idx"] is not None and l["ns"] >= 85:
                t = terms[ls][l["idx"]]
                row.update(dict(mpsno="", canonical_name=t["member_name"], state_at_election=t["state"],
                                constituency_at_election=t["constituency"], party_at_election=t["party_name"],
                                party_short=t["party_short"], election_type=t["election_type"],
                                election_date=t["election_date"], source=t["source"] + " [name-only match, not in sansad list]",
                                match_score=round(l["ns"], 1), needs_review=True,
                                review_reason="asker not found in official member list; matched election record by name only"))
            else:
                row.update(dict(mpsno="", canonical_name="", state_at_election="", constituency_at_election="",
                                party_at_election="", party_short="", election_type="", election_date="", source="",
                                match_score=round(score, 1), needs_review=True,
                                review_reason="asker not found in official member list"))
            row["_amb"] = False
        row["_top"] = [(members[i]["canonical_name"], members[i]["official_const"], members[i]["official_state"], round(s))
                       for s, i in top]
        row["_member"] = m
        # manual override
        if mk is not None:
            for k in ("mpsno", "canonical_name", "state_at_election", "constituency_at_election", "party_at_election",
                      "party_short", "election_type", "election_date"):
                if mk.get(k, "") != "":
                    row[k] = mk[k]
            if mk.get("party_at_election") and not mk.get("party_short"):
                row["party_short"] = parties.canonical(mk["party_at_election"], "manual")[1]
                row["party_at_election"] = parties.canonical(mk["party_at_election"], "manual")[0]
            elif mk.get("party_at_election"):
                parties.canonical(mk["party_at_election"], "manual")
            if mk.get("mpsno") and not mk.get("party_at_election"):
                mm = next((x for x in members if str(x["mpsno"]) == str(mk["mpsno"])), None)
                if mm and mm["party_at_election"]:
                    for k in ("canonical_name", "state_at_election", "constituency_at_election", "party_at_election",
                              "party_short", "election_type", "election_date"):
                        row[k] = mm[k]
            row["source"] = "manual override (mp_party_manual.csv)"
            row["match_score"] = 100.0
            row["needs_review"] = False
            row["review_reason"] = ""
        out.append(row)
    return members, out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=20260101)
    a = ap.parse_args()

    terms = load_terms()
    manual = load_manual()
    for ls in LSES:                     # register every party spelling seen, for party_names.csv
        for t in terms[ls]:
            parties.canonical(t["party_raw"], t["source"].split(":")[0].split(" (")[0])

    all_rows, all_members, recs_by_ls, askers_by_ls = [], [], {}, {}
    for ls in LSES:
        cnt, recs = read_askers(ls)
        recs_by_ls[ls], askers_by_ls[ls] = recs, cnt
        members, rows = resolve_ls(ls, terms, cnt, manual)
        for m in load_official(ls):
            parties.canonical(m.get("partyFname"), "sansad.in (current party, not used)")
        all_members += members
        all_rows += rows
        print(f"LS{ls}: members={len(members)} askers={len(cnt)} resolved={sum(1 for r in rows if r['party_at_election'])} "
              f"review={sum(1 for r in rows if r['needs_review'])}")

    # --- outputs
    mem_df = pd.DataFrame(all_members)
    mem_cols = ["ls", "mpsno", "canonical_name", "state_at_election", "constituency_at_election", "party_at_election",
                "party_short", "election_type", "election_date", "source", "match_score", "needs_review", "review_reason",
                "official_state", "official_const", "official_party_now", "status", "link_level", "term_name"]
    mem_df[mem_cols].to_csv(os.path.join(REF, "mp_party_members.csv"), index=False, encoding="utf-8")
    df = pd.DataFrame(all_rows)
    if len(df):
        df[OUT_COLS].to_csv(os.path.join(REF, "mp_party.csv"), index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=OUT_COLS).to_csv(os.path.join(REF, "mp_party.csv"), index=False)

    rev = []
    for r in all_rows:
        if not r["needs_review"]:
            continue
        m = r.get("_member") or {}
        sug = " || ".join(f"{n} | {c} | {s} | {sc}" for n, c, s, sc in r["_top"])
        rev.append(dict(ls=r["ls"], asker_name=r["asker_name"], n_questions=r["n_questions"], reason=r["review_reason"],
                        auto_mpsno=r.get("mpsno", ""), auto_name=r.get("canonical_name", ""),
                        auto_party=r.get("party_at_election", ""), auto_type=r.get("election_type", ""),
                        match_score=r.get("match_score", ""),
                        official_current_party=m.get("official_party_now", ""),
                        official_state=m.get("official_state", ""), official_const=m.get("official_const", ""),
                        suggestions=sug,
                        election_row_suggestions=m.get("suggestions", ""),
                        # to resolve: copy this row into mp_party_manual.csv and fill the columns below
                        party_at_election="", state_at_election="", constituency_at_election="", election_type="",
                        election_date="", mpsno="", canonical_name=""))
    pd.DataFrame(rev).sort_values(["ls", "n_questions"], ascending=[True, False]).to_csv(
        os.path.join(REF, "mp_party_review.csv"), index=False, encoding="utf-8") if rev else \
        pd.DataFrame(columns=["ls", "asker_name"]).to_csv(os.path.join(REF, "mp_party_review.csv"), index=False)
    parties.write_table(os.path.join(REF, "party_names.csv"))
    print("wrote mp_party.csv", len(df), "| review queue", len(rev), "| party_names", len(parties.SEEN))

    if a.validate or True:
        validate(all_rows, all_members, recs_by_ls, askers_by_ls, terms, a.seed, write=a.validate)


# ------------------------------------------------------------------------------------------------ validation
SWITCHERS = [  # (name fragment, ls, expected party short(s)) -- expectations from the task brief
    ("Jyotiraditya Scindia", 13, {"INC"}), ("Jyotiraditya Scindia", 14, {"INC"}),
    ("Jyotiraditya Scindia", 15, {"INC"}), ("Jyotiraditya Scindia", 18, {"BJP"}),
    ("Kirti Azad", 13, {"BJP"}), ("Kirti Azad", 15, {"BJP"}), ("Kirti Azad", 16, {"BJP"}),
    ("Shatrughan Sinha", 15, {"BJP"}), ("Shatrughan Sinha", 16, {"BJP"}), ("Shatrughan Sinha", 17, {"AITC"}),
    ("Shatrughan Sinha", 18, {"AITC"}),
    ("Suvendu Adhikari", 15, {"AITC"}), ("Suvendu Adhikari", 16, {"AITC"}),
    ("Babul Supriyo", 17, {"BJP"}), ("Kunwar Danish Ali", 17, {"BSP"}), ("Ravneet Singh Bittu", 15, {"INC"}),
    ("Ravneet Singh Bittu", 16, {"INC"}), ("Ravneet Singh Bittu", 17, {"INC"}),
]


def validate(rows, members, recs_by_ls, askers_by_ls, terms, seed, write=False):
    L = []
    P = L.append
    P("== Coverage of question askers (automatic = party assigned AND not flagged for review) ==")
    P("LS  askers  resolved%  confident%  mentions  mentions-resolved%  records  records-all-askers-resolved%  ambiguous")
    by = defaultdict(list)
    for r in rows:
        by[r["ls"]].append(r)
    for ls in LSES:
        rs = by[ls]
        if not rs:
            P(f"{ls}   (no question data in data/raw/api/ls{ls} yet)")
            continue
        res = sum(1 for r in rs if r["party_at_election"])
        conf = sum(1 for r in rs if r["party_at_election"] and not r["needs_review"])
        ment = sum(r["n_questions"] for r in rs)
        mres = sum(r["n_questions"] for r in rs if r["party_at_election"] and not r["needs_review"])
        ok = {r["asker_name"] for r in rs if r["party_at_election"] and not r["needs_review"]}
        recs = recs_by_ls[ls]
        rall = sum(1 for ms in recs if ms and all(m in ok for m in ms))
        amb = sum(1 for r in rs if r.get("_amb"))
        P(f"{ls}  {len(rs):5d}  {100*res/len(rs):6.1f}%  {100*conf/len(rs):6.1f}%  {ment:7d}  {100*mres/ment:6.1f}%        "
          f"{len(recs):6d}  {100*rall/max(len(recs),1):6.1f}%  {amb}")
    P("")
    P("== Member-level coverage (every MP of the official sansad list per LS; independent of which MPs asked) ==")
    P("LS  members  party-found%  confident%  nominated  by-election-members  ambiguous-links")
    md = defaultdict(list)
    for m in members:
        md[m["ls"]].append(m)
    for ls in LSES:
        ms = md[ls]
        f = sum(1 for m in ms if m["party_at_election"])
        c = sum(1 for m in ms if m["party_at_election"] and not m["needs_review"])
        P(f"{ls}  {len(ms):5d}  {100*f/len(ms):8.1f}%  {100*c/len(ms):8.1f}%  "
          f"{sum(1 for m in ms if m['election_type']=='nominated'):5d}  {sum(1 for m in ms if m['election_type']=='bye'):8d}  "
          f"{sum(1 for m in ms if m.get('ambiguous')):5d}")
    P("")
    P("== Known switchers (party AT THE TIME), member-level ==")
    fails = 0
    for frag, ls, exp in SWITCHERS:
        hits = [m for m in md[ls] if name_sim(frag, m["canonical_name"]) >= 80 and
                (is_subset(frag, m["canonical_name"]) or name_sim(frag, m["canonical_name"]) >= 90)]
        got = [(m["canonical_name"], m["party_short"] or "?", m["election_type"], m["election_date"][:10],
                m["constituency_at_election"]) for m in hits]
        ok = bool(hits) and all(g[1] in exp for g in got)
        fails += not ok
        P(f"  LS{ls} {frag:11s} expect {sorted(exp)} -> {'OK ' if ok else 'FAIL'} {got}")
    P(f"  switcher failures: {fails}/{len(SWITCHERS)}")
    txt = "\n".join(L)
    print(txt)
    if write:
        open(os.path.join(REF, "mp_party_validation.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
