"""Link a person (name + state + constituency) to a seat-term row (general-election winner or by-election winner)."""
from rapidfuzz import fuzz, process
from namematch import name_sim, state_key, const_sim, const_key, tokens, pkey, is_subset

STRONG = "strong"
MEDIUM = "medium"
WEAK = "weak"

_IDX = {}


def _index(terms):
    """cache of cheap keys used to pre-filter candidates before the (slower) name_sim"""
    k = id(terms)
    if k not in _IDX or _IDX[k][0] is not terms or len(_IDX[k][1]) != len(terms):
        _IDX[k] = (terms, [" ".join(sorted(pkey(t) for t in tokens(x["member_name"]))) for x in terms],
                   [const_key(x["constituency"]) for x in terms])
    return _IDX[k]


def _prefilter(names, const, terms, n=40):
    _, keys, ckeys = _index(terms)
    idx = set()
    for nm in names:
        q = " ".join(sorted(pkey(t) for t in tokens(nm)))
        if not q:
            continue
        for _, _, i in process.extract(q, keys, scorer=fuzz.token_set_ratio, limit=n):
            idx.add(i)
    ck = const_key(const) if const else ""
    if ck:
        idx.update(i for i, c in enumerate(ckeys) if c and fuzz.ratio(ck, c) >= 85)
    return sorted(idx)


def score_candidates(name, state, const, terms, alt_names=(), top=5, exclude=()):
    """terms: list of dicts (member_name, state, constituency, ...) for ONE Lok Sabha.

    Returns the best `top` candidates [(total, name_sim, state_ok, const_sim, term_idx, subset)], best first.
    `alt_names` are extra spellings of the same person (e.g. the asker's spelling next to the official one).
    """
    sk = state_key(state) if state else ""
    names = [n for n in (name, *alt_names) if n]
    out = []
    for i in _prefilter(names, const, terms):
        if i in exclude:
            continue
        t = terms[i]
        ns = max(name_sim(n, t["member_name"]) for n in names)
        sub = any(is_subset(n, t["member_name"]) for n in names)
        if ns < 40:
            continue
        s_ok = bool(sk) and sk == state_key(t["state"])
        cs = const_sim(const, t["constituency"]) if const else 0.0
        total = ns + (8 if s_ok else 0) + (10 if cs >= 85 else 5 if cs >= 70 else 0)
        if sk and not s_ok and cs < 70:
            total -= 10
        out.append((total, ns, s_ok, cs, i, sub))
    out.sort(reverse=True)
    return out[:top]


def classify(c):
    """c = (total, ns, s_ok, cs, idx, subset) -> STRONG / MEDIUM / WEAK"""
    total, ns, s_ok, cs, _, sub = c
    if ns >= 85 and (s_ok or cs >= 80):
        return STRONG
    if ns >= 92:
        return MEDIUM
    if ns >= 75 and s_ok and cs >= 70:
        return MEDIUM
    if ns >= 80 and (s_ok or cs >= 80):
        return MEDIUM
    if ns >= 65 and s_ok and cs >= 85:      # same state and (near-)same constituency: spelling/initial variants
        return MEDIUM
    if sub and s_ok and cs >= 90 and ns >= 45:   # e.g. ECI "MANOJ" vs "Manoj Kumar Sinha", same seat
        return MEDIUM
    if sub and s_ok and ns >= 60 and cs >= 50:      # ('Yaduveer Wadiyar' within 'Yaduveer Krishnadatta ... Wadiyar')
        return MEDIUM
    return WEAK


def best_link(name, state, const, terms, alt_names=(), exclude=()):
    """-> dict(idx, level, total, ns, ambiguous, cands) ; idx None if nothing acceptable"""
    cands = score_candidates(name, state, const, terms, alt_names, exclude=exclude)
    if not cands:
        return {"idx": None, "level": WEAK, "total": 0, "ns": 0, "ambiguous": False, "cands": []}
    best = cands[0]
    level = classify(best)
    amb = False
    if len(cands) > 1:
        r = cands[1]
        # runner-up that is a different person and almost as good
        if classify(r) != WEAK and best[0] - r[0] <= 4 and \
                terms[r[4]]["member_name"] != terms[best[4]]["member_name"]:
            amb = True
    return {"idx": best[4] if level != WEAK else None, "level": level, "total": best[0], "ns": best[1],
            "ambiguous": amb, "cands": cands}


def assign_all(people, terms):
    """people: list of dict(name, state, const[, alt]) of ONE Lok Sabha  ->  list of link dicts (same order).

    Phase 1: ordinary best_link.  Phase 2 ('leftover'): a person still unlinked is paired with a seat-term row that
    NOBODY else was linked to, when the name is a clear match (>=70) regardless of state/constituency.  This
    recovers members who later moved to another constituency/state (the official list only has the latest one)."""
    res = [best_link(p["name"], p["state"], p["const"], terms, p.get("alt", ())) for p in people]
    claimed = {r["idx"] for r in res if r["idx"] is not None}
    for k, p in enumerate(people):
        if res[k]["idx"] is not None:
            continue
        c = score_candidates(p["name"], "", "", terms, p.get("alt", ()), exclude=claimed)
        c = [x for x in c if x[1] >= 70]
        if c and (len(c) == 1 or c[0][1] - c[1][1] >= 8):
            res[k] = {"idx": c[0][4], "level": "leftover", "total": c[0][0], "ns": c[0][1], "ambiguous": False,
                      "cands": c}
            claimed.add(c[0][4])
    # Phase 3: same state + (near-)identical constituency, row still unclaimed, name only loosely similar
    # (ECI "Manoj" / "Tabassum Begum" vs "Manoj Kumar Sinha" / "Tabassum Hasan"); a seat-term row nobody else owns.
    pend = []
    for k, p in enumerate(people):
        if res[k]["idx"] is None and p.get("const"):
            c = score_candidates(p["name"], p["state"], p["const"], terms, p.get("alt", ()), exclude=claimed)
            c = [x for x in c if x[2] and x[3] >= 90 and x[1] >= 45]
            if c:
                pend.append((c[0][1], k, c))
    for _, k, c in sorted(pend, reverse=True):
        c = [x for x in c if x[4] not in claimed]
        if c:
            res[k] = {"idx": c[0][4], "level": "leftover-seat", "total": c[0][0], "ns": c[0][1], "ambiguous": False,
                      "cands": c}
            claimed.add(c[0][4])
    return res
