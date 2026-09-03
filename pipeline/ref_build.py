"""Build data/ref/seat_terms.csv : who held each Lok Sabha seat in the 13th-18th Lok Sabha, and for which party
they were ELECTED (general election or by-election).

Sources (see README block at the end of mp_party.py for licences):
  * 13th LS (1999): ECI "Statistical Report on General Elections, 1999 to the 13th Lok Sabha, Vol. I"
    (detailed results, PDF).  Downloaded from the Wayback Machine copy of eci.nic.in, cached in data/ref/source/.
  * 14th-18th LS: data/ref/source/eci0414.csv / eci1924.csv (India Data Portal, ECI data, ODC-BY).
  * By-elections: per-LS Wikipedia member lists (successor rows + "Elected on" annotations),
    Wikipedia "By-elections to the 18th Lok Sabha", and (fallback) the "Bye-election" boxes on Wikipedia
    constituency pages, looked up for every official-list member who is not otherwise explained.

Run:  python -I pipeline/ref_build.py [--no-fetch] [--refresh-wiki]
Re-runnable; network is used only for files missing from data/ref/source/ (and --refresh-wiki).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import requests
from lxml import html as lhtml

import parties
from linking import best_link, assign_all, score_candidates, STRONG, MEDIUM
from namematch import name_sim, state_key, const_key, const_sim

ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "data", "ref")
SRC = os.path.join(REF, "source")
WPC = os.path.join(SRC, "wp_const")
UA = "lok-sabha-research/1.0 (academic; contact tech@stellarisvp.com)"

URL_1999 = "https://web.archive.org/web/20140718183222if_/http://eci.nic.in/eci_main/StatisticalReports/LS_1999/Vol_I_LS_99.pdf"
PDF_1999 = os.path.join(SRC, "eci1999_vol1.pdf")

YEAR = {13: 1999, 14: 2004, 15: 2009, 16: 2014, 17: 2019, 18: 2024}
RESULT_DATE = {13: "1999-10-06", 14: "2004-05-13", 15: "2009-05-16", 16: "2014-05-16", 17: "2019-05-23",
               18: "2024-06-04"}
# (first year, last year) in which a by-election of that Lok Sabha can fall
LS_SPAN = {13: (1999, 2004), 14: (2004, 2009), 15: (2009, 2014), 16: (2014, 2019), 17: (2019, 2024), 18: (2024, 2027)}
WP_TITLE = {13: "13th", 14: "14th", 15: "15th", 16: "16th", 17: "17th", 18: "18th"}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA
NO_FETCH = False


def fetch(url, path=None, params=None, sleep=1.0, binary=False):
    if NO_FETCH:
        raise RuntimeError(f"network disabled, missing {path or url}")
    for k in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=90)
            r.raise_for_status()
            time.sleep(sleep)
            if path:
                with open(path, "wb") as f:
                    f.write(r.content)
            return r.content if binary else r.text
        except Exception as e:  # pragma: no cover
            if k == 2:
                raise
            time.sleep(3 * (k + 1))


# ------------------------------------------------------------------------------------------------- 1999 ECI PDF
def _clean_state_1999(s):
    s = s.title().replace(" & ", " & ")
    return {"Delhi": "Delhi", "Orissa": "Odisha", "Pondicherry": "Puducherry", "Jammu & Kashmir": "Jammu & Kashmir"}.get(s, s)


def parse_eci1999():
    """-> (candidates DataFrame, party_abbr -> full name dict).  Cached as data/ref/source/eci1999_candidates.csv"""
    cand_csv = os.path.join(SRC, "eci1999_candidates.csv")
    abbr_csv = os.path.join(SRC, "eci1999_party_abbr.csv")
    if os.path.exists(cand_csv) and os.path.exists(abbr_csv):
        d = pd.read_csv(cand_csv, keep_default_na=False)
        a = pd.read_csv(abbr_csv, keep_default_na=False)
        return d, dict(zip(a.abbr, a.party))
    from pypdf import PdfReader
    if not os.path.exists(PDF_1999):
        print("downloading ECI 1999 Vol I from Wayback ...")
        fetch(URL_1999, PDF_1999, binary=True)
    rd = PdfReader(PDF_1999)
    # party abbreviations (pp. 1-5 of the report)
    txt = "\n".join((rd.pages[i].extract_text() or "") for i in range(2, 7))
    abbr = {}
    for l in txt.split("\n"):
        m = re.match(r"^([A-Z0-9()\-.]+)\s+(.*?)(\d+)\s*\.\s*$", l.strip())
        if m and not l.startswith(("PARTYTYPE", "(rpt")):
            abbr[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).strip()
    # detailed results (report pp. 164-263)
    raw = "\n".join((rd.pages[i].extract_text() or "") for i in range(164, len(rd.pages)))
    lines = [l.strip() for l in raw.split("\n")]
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if re.match(r"^\d+\s*\.\s", l) and not l.endswith("%"):      # wrapped candidate name
            j = i
            while j + 1 < len(lines) and not lines[j].endswith("%") and j - i < 4:
                j += 1
                l = l + " " + lines[j]
            i = j
        out.append(l)
        i += 1
    cand_re = re.compile(r"^(\d+)\s*\.\s+(.*)\s+([MFT])\s+(\S+)\s+(\d+)\s+([\d.]+)%\s*$")
    cons_re = re.compile(r"^Constituency\s+(\d+)\s+(.*)$")
    hdr = re.compile(r"^[A-Z][A-Z &]+$")
    state = cons = None
    rows = []
    for l in out:
        m = cons_re.match(l)
        if m:
            cons = (int(m.group(1)), m.group(2).strip())
            continue
        m = cand_re.match(l)
        if m and cons:
            rows.append((state, cons[0], cons[1], int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip(),
                         m.group(3), m.group(4), int(m.group(5))))
            continue
        if hdr.match(l) and not l.startswith("GENERAL") and "DETAILED" not in l and l not in ("WOMEN CANDIDATES", "CONSTITUENCY"):
            state = l.replace("NATIONAL CAPITAL TERRITORY OF DELHI", "DELHI")
    d = pd.DataFrame(rows, columns=["state", "cno", "constituency", "rank", "candidate", "sex", "party_abbr", "votes"])
    d["state"] = d["state"].map(_clean_state_1999)
    assert d.groupby(["state", "cno"]).ngroups == 543, "1999 parse: expected 543 constituencies"
    d.to_csv(cand_csv, index=False)
    pd.DataFrame(sorted(abbr.items()), columns=["abbr", "party"]).to_csv(abbr_csv, index=False)
    return d, abbr


def general_terms():
    """winner of every seat in 1999-2024 general elections -> list of dict"""
    terms = []
    d99, abbr = parse_eci1999()
    w = d99.loc[d99.groupby(["state", "cno"])["votes"].idxmax()]
    assert (w["rank"] == 1).all()
    for r in w.itertuples():
        full = abbr.get(r.party_abbr, r.party_abbr)
        terms.append(dict(ls=13, state=r.state, constituency=re.sub(r"\s*\((SC|ST)\)\s*$", "", r.constituency).title(),
                          member_name=r.candidate.title(), party_raw=full, election_type="general",
                          election_date=RESULT_DATE[13], date_precision="day",
                          source="ECI Statistical Report 1999 Vol I (detailed results)", votes=r.votes))
    for f, years in (("eci0414.csv", (2004, 2009, 2014)), ("eci1924.csv", (2019, 2024))):
        e = pd.read_csv(os.path.join(SRC, f), low_memory=False).reset_index(drop=True)
        e["v"] = pd.to_numeric(e["votes_secured_general"], errors="coerce").fillna(0) + \
            pd.to_numeric(e["votes_secured_postal"], errors="coerce").fillna(0)
        e = e.dropna(subset=["pc_name", "state_name"]).reset_index(drop=True)
        win = e.loc[e.groupby(["year", "state_name", "pc_name"])["v"].idxmax()]
        for r in win.itertuples():
            ls = {v: k for k, v in YEAR.items()}[int(r.year)]
            terms.append(dict(ls=ls, state=r.state_name.title().replace(" And ", " and "), constituency=r.pc_name,
                              member_name=str(r.candidate_name).strip(), party_raw=r.party_name,
                              election_type="general", election_date=RESULT_DATE[ls], date_precision="day",
                              source="ECI via India Data Portal (ODC-BY)", votes=int(r.v)))
    return terms


# ------------------------------------------------------------------------------------------------- Wikipedia lists
def _expand(t):
    grid, pend = [], {}
    for r in t.xpath(".//tr"):
        row, col = [], 0
        for c in r.xpath("./th|./td"):
            while col in pend and pend[col][1] > 0:
                row.append(pend[col][0]); pend[col][1] -= 1; col += 1
            rs, cs = int(re.sub(r"\D", "", c.get("rowspan") or "1") or 1), int(re.sub(r"\D", "", c.get("colspan") or "1") or 1)
            for _ in range(cs):
                row.append(c)
                if rs > 1:
                    pend[col] = [c, rs - 1]
                col += 1
        while col in pend and pend[col][1] > 0:
            row.append(pend[col][0]); pend[col][1] -= 1; col += 1
        grid.append(row)
    return grid


def _text(c):
    for br in c.xpath(".//br"):
        br.tail = "\n" + (br.tail or "")
    return re.sub(r"\[[^\]]*\]", "", c.text_content()).replace("\xa0", " ")


MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_date(text):
    """first date in text -> (iso, precision) ; handles 3.6.2002 / 3 June 2002 / Dec 2008 / 2008"""
    if not text:
        return "", ""
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", text)
    if m:
        d, mo, y = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}", "day"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", text)
    if m and m.group(2)[:3].lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)[:3].lower()]:02d}-{int(m.group(1)):02d}", "day"
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{4})", text)
    if m and m.group(1)[:3].lower() in MONTHS:
        return f"{int(m.group(2)):04d}-{MONTHS[m.group(1)[:3].lower()]:02d}", "month"
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return m.group(0), "year"
    return "", ""


def wiki_html(n, refresh=False):
    p = os.path.join(SRC, f"wphtml{n}.json")
    if refresh or not os.path.exists(p):
        fetch("https://en.wikipedia.org/w/api.php", p, params=dict(
            action="parse", page=f"List of members of the {WP_TITLE[n]} Lok Sabha", prop="text|revid", format="json",
            formatversion=2))
    return json.load(open(p, encoding="utf-8"))["parse"]


def wiki_list_persons(n, refresh=False):
    """every person row of the per-LS Wikipedia list: dict(state, const, name, party, ann, seq)"""
    root = lhtml.fromstring(wiki_html(n, refresh)["text"])
    body = root.xpath('//div[contains(@class,"mw-parser-output")]')[0]
    state, res = None, []
    for el in body:
        cls = el.get("class") or ""
        if "mw-heading" in cls or el.tag in ("h2", "h3"):
            hh = el.xpath(".//h2|.//h3|self::h2|self::h3")
            if hh:
                state = hh[0].text_content().replace("[edit]", "").strip()
        elif el.tag == "table" and "wikitable" in cls:
            g = _expand(el)
            if not g:
                continue
            hdr = [c.text_content().strip().lower() for c in g[0]]
            if not any("constituency" in h for h in hdr):
                continue
            ci = [i for i, h in enumerate(hdr) if "constituency" in h][0]
            ni = [i for i, h in enumerate(hdr) if "name" in h or "member" in h][0]
            pi0 = [i for i, h in enumerate(hdr) if "party" in h][0]
            lastc, seq = None, 0
            for row in g[1:]:
                if len(row) <= ni:
                    continue
                const = _text(row[ci]).strip()
                if row[ci] is not lastc:
                    seq, lastc = 0, row[ci]
                ptxt = ""
                for c in row[pi0:pi0 + 2]:
                    tx = _text(c).strip()
                    if tx:
                        ptxt = tx
                cur = None
                for l in [x.strip() for x in _text(row[ni]).split("\n") if x.strip()]:
                    if l.startswith("(") and cur is not None:
                        cur["ann"] += " " + l
                    else:
                        cur = dict(state=state, const=const, name=l, party=ptxt, ann="", seq=seq)
                        seq += 1
                        res.append(cur)
    # names such as "X (Resigned on 3.3.2008)" carry the annotation inside the name cell
    for p in res:
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", p["name"])
        if m and re.search(r"resign|elected|died|expel|disqual|switch|assassin|expired|ceased|won on", m.group(2), re.I):
            p["name"], p["ann"] = m.group(1).strip(), (p["ann"] + " (" + m.group(2) + ")").strip()
        p["name"] = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", p["name"])).strip()
    return res


def wiki_bye_terms(n):
    """by-election successors from the per-LS list (rows with seq>0).

    Skipped: 'Vacant' rows, and rows that are the SAME person as the seat's first member (the lists record a later
    party switch that way -- 'Switched from BRS to INC' -- which is not a new election)."""
    out, first = [], {}
    for p in wiki_list_persons(n):
        k = (p["state"], p["const"])
        if p["seq"] == 0:
            first[k] = p["name"]
            continue
        if not p["name"] or re.match(r"(?i)^(vacant|vacancy|n/?a)\b", p["name"]) or re.match(r"(?i)^vacant", p["party"] or ""):
            continue
        if re.fullmatch(r"[-\u2013\u2014 ]*", p["name"]):
            continue
        ann = p["ann"]
        if re.search(r"(?i)switch|joined|defect|ticket", ann + " " + p["name"]):
            continue
        if first.get(k) and name_sim(first[k], p["name"]) >= 85:
            continue
        # 'elected on' annotation belongs to this person; if absent, date unknown
        dt, prec = parse_date(re.sub(r"(?i).*?elected (on|in)", "", ann)) if re.search(r"(?i)elected", ann) else ("", "")
        out.append(dict(ls=n, state=p["state"], constituency=re.sub(r"\s*\((SC|ST)\)\s*$", "", p["const"]),
                        member_name=p["name"], party_raw=p["party"], election_type="bye", election_date=dt,
                        date_precision=prec, source=f"Wikipedia: List of members of the {WP_TITLE[n]} Lok Sabha",
                        votes=""))
    return out


def wiki_bye18_terms():
    p = os.path.join(SRC, "wpbye18.json")
    if not os.path.exists(p):
        fetch("https://en.wikipedia.org/w/api.php", p, params=dict(
            action="parse", page="By-elections to the 18th Lok Sabha", prop="text|revid", format="json", formatversion=2))
    root = lhtml.fromstring(json.load(open(p, encoding="utf-8"))["parse"]["text"])
    out = []
    for t in root.xpath('//table[contains(@class,"wikitable")]'):
        g = _expand(t)
        if len(g) < 3 or "Constituency" not in " ".join(c.text_content() for c in g[0]):
            continue
        for row in g[2:]:
            tx = [_text(c).strip() for c in row]
            if len(tx) < 12 or tx[9] in ("", "TBD") or tx[0] == "":
                continue
            dt, prec = parse_date(tx[5])
            out.append(dict(ls=18, state=tx[2], constituency=tx[1], member_name=tx[9],
                            party_raw=tx[11] or tx[10], election_type="bye", election_date=dt, date_precision=prec,
                            source="Wikipedia: By-elections to the 18th Lok Sabha", votes=""))
    return out


def merge_wikipedia_general(gen):
    """Cross-check ECI general-election winners against the per-LS Wikipedia lists (seq-0 rows).

    Pass 1 links a Wikipedia row to an ECI winner by NAME (strong evidence).  Pass 2 links the rest by SEAT (same
    state group + similar constituency name; a seat has exactly one general winner) provided the names do not
    flatly contradict.  A Wikipedia row whose seat has no (unclaimed) ECI winner is added as a general-election row
    -- the staged ECI files lack e.g. Odisha and Chhattisgarh for 2014, Surat 2024 (unopposed) and a few seats that
    are mislabelled.  A seat whose ECI winner has a different name from Wikipedia is reported as 'seat-conflict'
    (ECI kept; Wikipedia lists sometimes show a later MP)."""
    out, xc = list(gen), []
    for n in (13, 14, 15, 16, 17, 18):
        tl = [t for t in gen if t["ls"] == n]
        W = [p for p in wiki_list_persons(n) if p["seq"] == 0 and p["state"] not in ("Nominated", None) and p["name"]]
        link, claimed = {}, {}
        for wi, p in enumerate(W):                                    # pass 1: name AND constituency agree
            r = best_link(p["name"], p["state"], p["const"], tl)
            if r["idx"] is not None and r["level"] == STRONG and r["idx"] not in claimed and                     (const_sim(p["const"], tl[r["idx"]]["constituency"]) >= 70 or r["ns"] >= 95):
                link[wi] = (r["idx"], "name"); claimed[r["idx"]] = wi
        for wi, p in enumerate(W):                                    # pass 1b: name only (mislabelled ECI seat)
            if wi in link:
                continue
            r = best_link(p["name"], p["state"], p["const"], tl, exclude=set(claimed))
            if r["idx"] is not None and r["level"] == STRONG and r["ns"] >= 90:
                link[wi] = (r["idx"], "name-relabel"); claimed[r["idx"]] = wi
        for wi, p in enumerate(W):                                    # pass 2: seats
            if wi in link:
                continue
            sk = state_key(p["state"])
            seat = sorted(((const_sim(p["const"], t["constituency"]), i) for i, t in enumerate(tl)
                           if state_key(t["state"]) == sk), reverse=True)
            seat = [x for x in seat if x[0] >= 85 and x[1] not in claimed]
            if seat and (len(seat) == 1 or seat[0][0] - seat[1][0] >= 8):
                t = tl[seat[0][1]]
                ns = name_sim(p["name"], t["member_name"])
                same_party = parties.canonical(p["party"])[0] == parties.canonical(t["party_raw"])[0]
                if ns >= 60 or (ns >= 40 and same_party):
                    link[wi] = (seat[0][1], "seat"); claimed[seat[0][1]] = wi
                else:
                    link[wi] = (seat[0][1], "seat-conflict")           # keep ECI winner, report
                    claimed[seat[0][1]] = wi
            else:
                r = best_link(p["name"], p["state"], p["const"], tl, exclude=set(claimed))
                if r["idx"] is not None:
                    link[wi] = (r["idx"], "name"); claimed[r["idx"]] = wi
                else:       # same state, name clearly (sub)matching, constituency label unusable (mislabelled in ECI file)
                    c = [x for x in score_candidates(p["name"], p["state"], "", tl, exclude=set(claimed))
                         if x[2] and ((x[5] and x[1] >= 60) or x[1] >= 80)]
                    if c and (len(c) == 1 or c[0][1] - c[1][1] >= 8):
                        link[wi] = (c[0][4], "name"); claimed[c[0][4]] = wi
        for wi, p in enumerate(W):
            pc = parties.canonical(p["party"])[0]
            if wi in link:
                idx, how = link[wi]
                t = tl[idx]
                ns = name_sim(p["name"], t["member_name"])
                ep = parties.canonical(t["party_raw"])[0]
                if how == "name-relabel" and ns >= 90 and const_sim(p["const"], t["constituency"]) < 60:
                    t["constituency"] = re.sub(r"\s*\((SC|ST)\)\s*$", "", p["const"])   # ECI file mislabels a few seats
                    t["source"] += " [constituency label corrected from Wikipedia]"
                xc.append(dict(ls=n, state=p["state"], const=p["const"], wiki_name=p["name"], eci_name=t["member_name"],
                               name_sim=round(ns), wiki_party=pc, eci_party=ep, status=how,
                               party_agree=int(pc == ep)))
            else:
                xc.append(dict(ls=n, state=p["state"], const=p["const"], wiki_name=p["name"], eci_name="", name_sim=0,
                               wiki_party=pc, eci_party="", status="no-eci-row", party_agree=0))
                if n != 13:
                    out.append(dict(ls=n, state=p["state"], constituency=re.sub(r"\s*\((SC|ST)\)\s*$", "", p["const"]),
                                    member_name=p["name"], party_raw=p["party"], election_type="general",
                                    election_date=RESULT_DATE[n], date_precision="day",
                                    source=f"Wikipedia: List of members of the {WP_TITLE[n]} Lok Sabha (no ECI row)",
                                    votes=""))
    return out, xc


# ------------------------------------------------------------------------------------------------- constituency pages
def wp_const_page(title, refresh=False):
    """wikitext of a Wikipedia constituency page (cached); tries the title, then a search."""
    os.makedirs(WPC, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:80]
    p = os.path.join(WPC, slug + ".json")
    if os.path.exists(p) and not refresh:
        return json.load(open(p, encoding="utf-8")).get("wikitext", "")
    wt = ""
    for attempt in (title, None):
        t = attempt
        if t is None:
            js = json.loads(fetch("https://en.wikipedia.org/w/api.php", params=dict(
                action="query", list="search", srsearch=f'intitle:"{title.replace(" Lok Sabha constituency", "")}" Lok Sabha constituency',
                srlimit=3, format="json")))
            hits = [h["title"] for h in js["query"]["search"] if "Lok Sabha constituency" in h["title"]]
            if not hits:
                break
            t = hits[0]
        try:
            js = json.loads(fetch("https://en.wikipedia.org/w/api.php", params=dict(
                action="parse", page=t, prop="wikitext", redirects=1, format="json", formatversion=2)))
        except Exception:
            continue
        if "parse" in js:
            wt = js["parse"]["wikitext"]
            break
    json.dump({"title": title, "wikitext": wt}, open(p, "w", encoding="utf-8"))
    return wt


def _plain(s):
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<ref.*?</ref>|<ref[^>]*/>", "", s, flags=re.S)
    return re.sub(r"\{\{[^{}]*\}\}", "", s).strip()


def _template_end(wt, start):
    """index just past the '}}' that closes the template opening at wt[start:start+2] == '{{'"""
    depth, i = 0, start
    while i < len(wt) - 1:
        two = wt[i:i + 2]
        if two == "{{":
            depth += 1; i += 2
        elif two == "}}":
            depth -= 1; i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(wt)


def parse_bye_boxes(wt):
    """-> list of dict(year, title, winner, party, votes) for by-election boxes on a constituency page.
    The winner is the 'winning candidate' row, or else the first candidate row of the box."""
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>/]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    res = []
    for m in re.finditer(r"\{\{\s*Election box begin", wt, re.I):
        e = _template_end(wt, m.start())
        title = re.search(r"title\s*=\s*([^|}]*)", wt[m.start():e])
        title = re.sub(r"\s+", " ", _plain(title.group(1))) if title else ""
        before = wt[max(0, m.start() - 250):m.start()]
        heads = re.findall(r"=+\s*([^=\n]+?)\s*=+", before)
        ctx = title + " " + (heads[-1] if heads else "")
        if not re.search(r"by-?\s?-?election|bye|bypoll|by-?\s?poll", ctx, re.I):
            continue
        end = wt.find("Election box end", e)
        blk = wt[e: end if end > 0 else e + 4000]
        rows = []
        for r in re.finditer(r"\{\{\s*Election box (winning )?candidate", blk, re.I):
            rows.append((bool(r.group(1)), blk[r.start(): _template_end(blk, r.start())]))
        if not rows:
            continue
        win = [x for x in rows if x[0]] or rows[:1]
        body = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", win[0][1])

        def par(k):
            mm = re.search(r"\|\s*" + k + r"\s*=\s*([^|\n}]*)", body)
            return mm.group(1).strip() if mm else ""
        ym = re.search(r"\b(?:19|20)\d{2}\b", ctx)
        res.append(dict(year=int(ym.group(0)) if ym else None, title=title, winner=_plain(par("candidate")),
                        party=_plain(par("party")), votes=par("votes")))
    return res


def wp_const_html(title, refresh=False):
    """rendered HTML of a Wikipedia constituency page (cached next to the wikitext)"""
    os.makedirs(WPC, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:80]
    p = os.path.join(WPC, slug + ".html.json")
    if os.path.exists(p) and not refresh:
        return json.load(open(p, encoding="utf-8")).get("html", "")
    h = ""
    try:
        js = json.loads(fetch("https://en.wikipedia.org/w/api.php", params=dict(
            action="parse", page=title, prop="text", redirects=1, format="json", formatversion=2)))
        h = js.get("parse", {}).get("text", "")
    except Exception:
        h = ""
    json.dump({"title": title, "html": h}, open(p, "w", encoding="utf-8"))
    return h


def lookup_member_table(const, member, ls, refresh=False):
    """fallback: 'Members of Parliament' table of the constituency page (Year | Member | Party, rowspan-expanded).
    Returns dict(party_raw, year, ...) for a row in the Lok Sabha's years whose member name matches."""
    lo, hi = LS_SPAN[ls]
    h = wp_const_html(f"{const} Lok Sabha constituency", refresh)
    if not h:
        return None
    root = lhtml.fromstring(h)
    best = None
    for t in root.xpath('//table[contains(@class,"wikitable")]'):
        g = _expand(t)
        if len(g) < 3:
            continue
        hdr = [c.text_content().strip().lower() for c in g[0]]
        yi = next((i for i, x in enumerate(hdr) if x.startswith("year") or x == "election"), None)
        mi = next((i for i, x in enumerate(hdr) if x in ("member", "member of parliament", "mp", "name", "winner")
                   or x.startswith("member")), None)
        pi = next((i for i, x in enumerate(hdr) if "party" in x), None)
        if yi is None or mi is None or pi is None:
            continue
        for row in g[1:]:
            if len(row) <= max(yi, mi, pi):
                continue
            ytxt = _text(row[yi]).strip()
            ym = re.search(r"\b(19|20)\d{2}\b", ytxt)
            if not ym:
                continue
            y = int(ym.group(0))
            marked = bool(re.search(r"[\^*#" + chr(0x2020) + r"]|by|bye|poll", ytxt, re.I))
            # a plain year equal to a general-election year is that election's winner, not a by-election
            if not (lo < y < hi or (marked and lo <= y <= hi)):
                continue
            s = name_sim(member, _text(row[mi]).strip())
            ptxt = ""
            for c in row[pi:pi + 2]:
                tx = _text(c).strip()
                if tx:
                    ptxt = tx
            if s >= 80 and ptxt and (best is None or s > best[0]):
                best = (s, dict(party_raw=ptxt, year=int(ym.group(0)), winner=_text(row[mi]).strip(),
                                title=f"members table, year cell '{ytxt}'", votes="", page=f"{const} Lok Sabha constituency"))
    return best[1] if best else None


def lookup_bye_via_constituency(member, ls, refresh=False):
    """try to find (party, year, source) for an official-list member who was NOT a general-election winner"""
    lo, hi = LS_SPAN[ls]
    const = member["constName"]
    wt = wp_const_page(f"{const} Lok Sabha constituency", refresh)
    if not wt:
        return None
    best = None
    pass_through = lambda: lookup_member_table(const, member["mpFirstLastName"], ls, refresh)
    for b in parse_bye_boxes(wt):
        if b["year"] and not (lo <= b["year"] <= hi):
            continue
        s = name_sim(member["mpFirstLastName"], b["winner"])
        if s >= 80 and (best is None or s > best[0]):
            best = (s, b)
    if best:
        b = best[1]
        return dict(party_raw=b["party"], year=b["year"], title=b["title"], winner=b["winner"], votes=b["votes"],
                    page=f"{const} Lok Sabha constituency")
    return pass_through()


# ------------------------------------------------------------------------------------------------- main
def load_official(n):
    return json.load(open(os.path.join(SRC, f"mem{n}.json"), encoding="utf-8"))["membersDtoList"]


def main():
    global NO_FETCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="never touch the network (cached files only)")
    ap.add_argument("--refresh-wiki", action="store_true", help="re-download Wikipedia pages")
    ap.add_argument("--no-const-lookup", action="store_true", help="skip Wikipedia constituency-page by-election lookups")
    a = ap.parse_args()
    NO_FETCH = a.no_fetch
    os.makedirs(REF, exist_ok=True)

    gen = general_terms()
    print("ECI general-election winners found:", pd.Series([t["ls"] for t in gen]).value_counts().sort_index().to_dict())
    for t in gen:                      # register ECI spellings first so they become the canonical ones
        parties.canonical(t["party_raw"])
    gen, xcheck = merge_wikipedia_general(gen)
    x = pd.DataFrame(xcheck)
    x.to_csv(os.path.join(REF, "wiki_vs_eci_general.csv"), index=False)
    print("Wikipedia general-list vs ECI winners:")
    print(x.groupby("ls").agg(rows=("status", "size"),
                             linked=("status", lambda v: v.isin(["name", "seat", "name-relabel"]).sum()),
                             conflict=("status", lambda v: (v == "seat-conflict").sum()),
                             added_from_wiki=("status", lambda v: (v == "no-eci-row").sum()),
                             name_ok=("name_sim", lambda v: (v >= 75).sum()),
                             party_agree=("party_agree", "sum")).to_string())

    bye = []
    for n in (14, 15, 16, 17, 18):          # the 13th list has (almost) no by-elections; see constituency lookup
        try:
            bye += wiki_bye_terms(n)
        except Exception as e:
            print("WARN wikipedia by-elections LS", n, e)
    try:
        bye += wiki_bye18_terms()
    except Exception as e:
        print("WARN wikipedia 18th by-election page", e)
    uniq = []                                # same LS, same constituency, same person from two Wikipedia pages
    for t in bye:
        dup = [u for u in uniq if u["ls"] == t["ls"] and const_key(u["constituency"]) == const_key(t["constituency"])
               and name_sim(u["member_name"], t["member_name"]) >= 85]
        if dup:
            if t["election_date"] and (not dup[0]["election_date"] or t["source"].startswith("Wikipedia: By-elections")):
                dup[0]["election_date"], dup[0]["date_precision"] = t["election_date"], t["date_precision"]
            continue
        uniq.append(t)
    bye = uniq
    print("by-election rows from Wikipedia lists:", pd.Series([t["ls"] for t in bye]).value_counts().sort_index().to_dict())

    terms = gen + bye
    # Explain every official-list member of each LS with a general/by-election row; for the rest look up the
    # by-election box on the Wikipedia constituency page.
    audit = []
    for n in (13, 14, 15, 16, 17, 18):
        tl = [t for t in terms if t["ls"] == n]
        off = [m for m in load_official(n) if "omin" not in (m["constName"] or "")]
        links = assign_all([dict(name=m["mpFirstLastName"], state=m["stateName"], const=m["constName"]) for m in off], tl)
        for m, r in zip(off, links):
            if r["idx"] is not None:
                continue
            found = None
            if not a.no_const_lookup:
                try:
                    found = lookup_bye_via_constituency(m, n, a.refresh_wiki)
                except Exception as e:
                    print("WARN lookup", m["mpFirstLastName"], e)
            if found:
                yr = found["year"]
                terms.append(dict(ls=n, state=m["stateName"], constituency=m["constName"],
                                  member_name=found["winner"] or m["mpFirstLastName"], party_raw=found["party_raw"],
                                  election_type="bye", election_date=str(yr or ""), date_precision="year" if yr else "",
                                  source=f"Wikipedia constituency page: {found['page']} ({found['title']})",
                                  votes=found["votes"]))
                audit.append((n, m["mpsno"], m["mpFirstLastName"], m["constName"], "found-on-constituency-page",
                              found["party_raw"], yr))
            else:
                audit.append((n, m["mpsno"], m["mpFirstLastName"], m["constName"], "UNEXPLAINED", "", ""))
    pd.DataFrame(audit, columns=["ls", "mpsno", "official_name", "official_const", "status", "party", "year"]).to_csv(
        os.path.join(REF, "seat_terms_unexplained.csv"), index=False)

    for t in terms:
        c, s, cur = parties.canonical(t["party_raw"], source=t["source"].split(":")[0].split(" (")[0])
        t["party_name"], t["party_short"] = c, s
    df = pd.DataFrame(terms)[["ls", "state", "constituency", "member_name", "party_name", "party_short",
                              "election_type", "election_date", "date_precision", "source", "party_raw", "votes"]]
    df.to_csv(os.path.join(REF, "seat_terms.csv"), index=False, encoding="utf-8")
    print("wrote seat_terms.csv", len(df), df.groupby(["ls", "election_type"]).size().to_dict())
    print("official-list members not explained by general/by-election data:",
          sum(1 for x in audit if x[4] == "UNEXPLAINED"))
    return df


if __name__ == "__main__":
    main()
