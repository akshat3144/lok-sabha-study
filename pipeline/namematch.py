"""Name / place normalisation and similarity helpers shared by the MP->party pipeline.

Indian names arrive in many spellings ("Rizwan Zaheer Khan" vs "RIZVAN ZAHEER ALIAS RIJJU BHAIYA",
"Sis Ram Ola" vs "SISRAM OLA").  The similarity here is token-aligned on a light phonetic key, so
that spelling variants, joined/split words, initials and extra middle names are tolerated.
"""
import re
import unicodedata
from rapidfuzz import fuzz

HON = {"shri", "smt", "dr", "prof", "professor", "adv", "advocate", "sh", "mr", "mrs", "ms", "sri",
       "col", "retd", "capt", "captain", "major", "gen", "lt", "thiru", "alias", "urf", "shrimati", "smti",
       "sushree", "sushri", "maj", "brig", "vsm", "ret", "late", "rajmata", "maharaja", "raja", "rani", "kunwar",
       "tmt", "selvi", "sardar", "swami", "mahant", "justice", "the", "hon", "ble", "yogi"}
# honorific-like tokens that are also real name parts are kept out of the strip list on purpose (e.g. kunwar, raja)
HON -= {"kunwar", "raja", "rani", "yogi", "swami", "mahant", "sardar", "maharaja", "rajmata", "sushree", "sushri"}


def _ascii(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


_FIRST_ONLY = {"kumari", "kum", "km", "ku"}          # honorific only in first position ("Kumari Selja" vs "Roop Kumari")
_SYN = {"md": "mohammad", "mohd": "mohammad", "mohamad": "mohammad", "mohamed": "mohammad", "mohammed": "mohammad",
        "muhammad": "mohammad", "muhammed": "mohammad", "mohmad": "mohammad"}


def tokens(name, keep_initials=False):
    n = _ascii(name).lower().replace("`", "'").replace("�", " ")
    n = re.sub(r"\(.*?\)", " ", n)          # nicknames / bracketed bits
    n = re.sub(r"\(.*$", " ", n)            # unclosed bracket
    n = re.sub(r"[^a-z ]", " ", n.replace("'", ""))
    out = []
    for k, t in enumerate(n.split()):
        if t in HON or (t in _FIRST_ONLY and not out):
            continue
        if len(t) == 1 and not keep_initials:
            continue
        out.append(_SYN.get(t, t))
    return out


def pkey(t):
    """light phonetic key for one token"""
    t = t.lower()
    for a, b in (("ph", "f"), ("w", "v"), ("z", "j"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"),
                 ("th", "t"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks")):
        t = t.replace(a, b)
    t = t.replace("aa", "a").replace("ee", "i").replace("oo", "u").replace("ou", "u").replace("ei", "i")
    t = re.sub(r"(.)\1+", r"\1", t)
    # vowel soup: ai/ay -> a ; ey -> i ; au/ow -> o ; y -> i
    t = t.replace("ai", "a").replace("ay", "a").replace("ey", "i").replace("au", "o").replace("ow", "o")
    return t.replace("y", "i")


_VOW = set("aeiouy")


def _skel(t):
    return "".join(c for c in t if c not in _VOW)


def _tok_sim(a, b):
    if a == b:
        return 100.0
    r = fuzz.ratio(a, b)
    # prefix abbreviations (e.g. "mohan" vs "mohana")
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        r = max(r, 88.0)
    # transliteration noise: same consonant skeleton, only vowels differ ("saugata"/"sougata", "ray"/"roy")
    if len(a) >= 2 and len(b) >= 2 and a[0] == b[0]:
        sa, sb = _skel(a), _skel(b)
        if sa == sb and (len(sa) >= 3 or (len(sa) == 2 and max(len(a), len(b)) <= 5) or len(a) == len(b)
                         or max(len(a), len(b)) <= 3):
            r = max(r, 90.0 if len(sa) >= 3 else 85.0)
    if r < 88:
        r *= 0.9        # near-misses are weak evidence ("Rajesh" vs "Ramesh", "Devendra" vs "Virendra")
    return r


def _variants(name):
    """main name, plus alternatives for 'X alias Y' and 'X (Y)' forms"""
    base = re.sub(r"\(.*?\)", " ", str(name))
    base = re.sub(r"\(.*$", " ", base)
    vs = [base]
    m = re.split(r"\b(?:alias|urf)\b", base, flags=re.I)
    if len(m) > 1:
        vs += [x for x in m if len(tokens(x)) >= 1]
    for br in re.findall(r"\(([^()]*)\)", str(name)):
        if len(tokens(br)) >= 2:
            vs.append(br)
    return vs


_GENERIC = {pkey(x) for x in ("kumar", "kumari", "singh", "devi", "prasad", "lal", "das", "rao", "reddy", "patel",
                              "yadav", "khan", "sharma", "verma", "gupta", "nath", "ram", "chandra", "bai", "ben",
                              "bhai", "begum", "chowdhury", "mohan", "kishore", "shankar", "pal")}


def _sim1(a, b):
    """similarity of two simple names; initials ("L. K.") may stand for full tokens ("Lal Krishna")"""
    fa = [pkey(t) for t in tokens(a)]
    fb = [pkey(t) for t in tokens(b)]
    ia = [t for t in tokens(a, True) if len(t) == 1]
    ib = [t for t in tokens(b, True) if len(t) == 1]
    if not fa and not fb:
        return 0.0
    if not fa or not fb:
        # one side is only initials / empty: weak evidence by construction
        return 0.0
    sa, sb = "".join(fa), "".join(fb)
    concat = max(fuzz.ratio(sa, sb), fuzz.ratio("".join(sorted(fa)), "".join(sorted(fb))))
    na, nb = len(fa) + len(ia), len(fb) + len(ib)
    pairs = []
    for i, x in enumerate(fa):
        for j, y in enumerate(fb):
            s = _tok_sim(x, y)
            if s >= 72:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_a, used_b, tot, wsum = set(), set(), 0.0, 0.0
    for s, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        w = 0.5 if (fa[i] in _GENERIC or fb[j] in _GENERIC) else 1.0
        tot += s * w
    # initials of one side vs unmatched full tokens of the other side (first letter), and initial == initial
    # (initials may only stand for a full token if every full token of the initial-bearing side is matched --
    #  otherwise "Jyotiraditya M Scindia" would match his father "Madhavrao Scindia")
    for ini, full, used, own, own_used in ((ia, fb, used_b, fa, used_a), (ib, fa, used_a, fb, used_b)):
        if len(own_used) < len(own):
            continue
        for c in ini:
            for k, tok in enumerate(full):
                if k not in used and tok[:1] == pkey(c)[:1]:
                    used.add(k); tot += 85.0
                    break
    common = set(ia) & set(ib)
    tot += 100.0 * len(common)
    short, long_ = min(na, nb), max(na, nb)
    sfull, sini = (fa, len(ia)) if na <= nb else (fb, len(ib))
    dens = sum(0.5 if t in _GENERIC else 1.0 for t in sfull) + sini + 0.5 * (long_ - short)
    align = tot / dens
    res = max(min(align, 100.0), concat * 0.97 if len(fa) != len(fb) else 0.0)   # concat: joined/split words
    # shorter name's letters contained in the longer one's ("Hema Malini" in "Hemamalini Dharmendra Deol")
    cs, cl = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if len(cs) >= 7 and fuzz.partial_ratio(cs, cl) >= 95:
        res = max(res, 78.0)
    if ia and ib and not (set(ia) & set(ib)):       # conflicting initials ("C. R. Chaudhary" vs "P. P. Chaudhary")
        res *= 0.75
    return res


def name_sim(a, b):
    """0..100 similarity between two person names (best over alias/bracket variants)."""
    best = 0.0
    va, vb = _variants(a), _variants(b)
    for i, x in enumerate(va):
        for j, y in enumerate(vb):
            s = _sim1(x, y)
            if i or j:
                s *= 0.95   # alternative forms are slightly weaker evidence
            best = max(best, s)
    return best


# ---- places ---------------------------------------------------------------------------
_STATE_FIX = {
    "orissa": "odisha", "pondicherry": "puducherry", "nct of delhi": "delhi", "national capital territory of delhi": "delhi",
    "chattisgarh": "chhattisgarh", "uttaranchal": "uttarakhand", "jammu and kashmir": "jammu and kashmir",
}
# collapse states that were split/merged so that old and new names compare equal
_STATE_GROUP = {
    "jharkhand": "bihar", "chhattisgarh": "madhya pradesh", "uttarakhand": "uttar pradesh", "telangana": "andhra pradesh",
    "ladakh": "jammu and kashmir", "daman and diu": "dnh dd", "dadra and nagar haveli": "dnh dd",
    "dadra and nagar haveli and daman and diu": "dnh dd",
}


def state_key(s):
    s = _ascii(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z ]", " ", s)
    s = " ".join(s.split())
    if s.startswith("the "):
        s = s[4:]
    s = _STATE_FIX.get(s, s)
    return _STATE_GROUP.get(s, s)


def const_key(c):
    c = _ascii(c).lower().replace("&", " and ")
    c = re.sub(r"\((sc|st|gen|general)\)", " ", c)
    c = re.sub(r"[^a-z ]", " ", c)
    c = c.replace("gauhati", "guwahati").replace("pondicherry", "puducherry").replace("trichur", "thrissur")
    return " ".join(c.split())


def const_sim(a, b):
    a, b = const_key(a), const_key(b)
    if not a or not b:
        return 0.0
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))   # (no subset matching: 'Mumbai North' != 'Mumbai North East')


def is_subset(a, b, thr=85):
    """True if every full (non-initial) token of the shorter name is matched in the longer one."""
    best = False
    for x in _variants(a):
        for y in _variants(b):
            fa = [pkey(t) for t in tokens(x)]
            fb = [pkey(t) for t in tokens(y)]
            if not fa or not fb:
                continue
            s, l = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
            if all(any(_tok_sim(p, q) >= thr for q in l) for p in s):
                best = True
    return best
