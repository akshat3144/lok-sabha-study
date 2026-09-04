"""Party-name normalisation across ECI (1999 report, India Data Portal), Wikipedia and sansad.in spellings.

canonical(raw) -> (canonical_name, short_code)
Every spelling seen is written to data/ref/party_names.csv by mp_party.py / ref_build.py.
"""
import re
import unicodedata

SMALL = {"of", "and", "for", "the", "in", "to"}

# raw-text repairs (ECI csv has mojibake; the 1999 PDF has typos)
_REPAIR = [
        ("Indi a", "India"), ("Lenninist", "Leninist"), ("M ovement", "Movement"),
    ("Mahasangha", "Mahasangh"),
]


def _clean(raw):
    """whitespace/punctuation clean-up; typographic quotes and dashes -> ASCII (the ECI csv is valid UTF-8 but
    Windows consoles/Excel often show curly quotes and en dashes as '?' or mojibake)"""
    s = str(raw)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("�", "'")
    s = re.sub(r"\s+-\s+", " - ", s)
    for a, b in _REPAIR:
        s = s.replace(a, b)
    s = s.replace("'S ", "'s ").replace("'S)", "'s)").replace("'S,", "'s,")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;")
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)", ")", s)
    s = re.sub(r"\)\s*\(", ") (", s)
    return s


def key(raw):
    s = _clean(raw).lower().replace("&", " and ")
    s = s.replace("-", " ").replace("'s ", "s ").replace("'", "")
    s = re.sub(r"\bpolitician\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# key (or alias key) -> (canonical display name, short code).  Aliases map several spellings to one entry.
_CUR = {}


def _add(canon, short, *aliases):
    for k in (canon,) + aliases:
        _CUR[key(k)] = (canon, short)


_add("Bharatiya Janata Party", "BJP", "BJP", "Bhartiya Janata Party")
_add("Indian National Congress", "INC", "Indian National Congress (Indira)", "INC")
_add("Bahujan Samaj Party", "BSP")
_add("Samajwadi Party", "SP")
_add("All India Trinamool Congress", "AITC", "Trinamool Congress", "AITC")
_add("Dravida Munnetra Kazhagam", "DMK")
_add("All India Anna Dravida Munnetra Kazhagam", "AIADMK", "AIADMK", "ADMK")
_add("Shivsena", "SS", "Shiv Sena", "Shiv Sena (Balasaheb)")
_add("Shiv Sena (Uddhav Balasaheb Thackrey)", "SS(UBT)", "Shiv Sena (Uddhav Balasaheb Thackeray)",
     "Shiv Sena (UBT)", "Shiv Sena (Uddhav Thackeray)")
_add("Telugu Desam", "TDP", "Telugu Desam Party")
_add("Communist Party Of India (Marxist)", "CPI(M)", "Communist Party of India (Marxist)", "CPM", "CPI(M)")
_add("Communist Party Of India", "CPI", "Communist Party of India")
_add("Communist Party Of India (Marxist-Leninist) (Liberation)", "CPI(ML)L", "Communist Party of India (Marxist-Leninist) (Liberation)",
     "Communist Party of India (Marxist–Leninist) Liberation", "Communist Party of India (Marxist-Leninist) Liberation",
     "Communist Party of India (Marxist-Leninist)", "Communist Party of India (Marxist–Leninist)")
_add("Janata Dal (United)", "JD(U)", "JDU")
_add("Janata Dal (Secular)", "JD(S)")
_add("Rashtriya Janata Dal", "RJD")
_add("Nationalist Congress Party", "NCP")
_add("Nationalist Congress Party - Sharadchandra Pawar", "NCP(SP)", "Nationalist Congress Party (Sharadchandra Pawar)",
     "Nationalist Congress Party - Sharadchandra Pawar")
_add("Yuvajana Sramika Rythu Congress Party", "YSRCP", "YSR Congress Party", "YSR Congress", "YSRCP")
_add("Biju Janata Dal", "BJD")
_add("Aam Aadmi Party", "AAP")
_add("Lok Jan Shakti Party", "LJP", "Lok Janshakti Party")
_add("Lok Shakti", "LS")
_add("Lok Janshakti Party(Ram Vilas)", "LJP(RV)", "Lok Janshakti Party (Ram Vilas)", "Lok Jan Shakti Party (Ram Vilas)")
_add("Rashtriya Lok Janshakti Party", "RLJP", "Rashtriya Lok Jan Shakti Party")
_add("Telangana Rashtra Samithi", "TRS", "Telangana Rashtra Samiti")
_add("Bharat Rashtra Samithi", "BRS", "Bharat Rashtra Samiti")
_add("Shiromani Akali Dal", "SAD")
_add("Shiromani Akali Dal (Amritsar)", "SAD(A)", "Shiromani Akali Dal (Amritsar)(Simranjit Singh Mann)",
     "Shiromani Akali Dal (Simranjit Singh Mann)")
_add("Jharkhand Mukti Morcha", "JMM")
_add("All India Forward Bloc", "AIFB")
_add("Rashtriya Lok Dal", "RLD")
_add("Apna Dal", "AD")
_add("Apna Dal (Soneylal)", "AD(S)", "Apna Dal (Sonelal)", "Apna Dal (Sonewal)", "Apna Dal (Sonelal)")
_add("Jammu & Kashmir National Conference", "JKNC", "Jammu and Kashmir National Conference", "JKN", "JKNC")
_add("Revolutionary Socialist Party", "RSP")
_add("Indian National Lok Dal", "INLD")
_add("All India Majlis-E-Ittehadul Muslimeen", "AIMIM", "All India Majlis-e-Ittehadul Muslimeen", "AIMIM")
_add("Indian Union Muslim League", "IUML", "Muslim League Kerala State Committee", "Muslim League")
_add("Pattali Makkal Katchi", "PMK")
_add("Marumalarchi Dravida Munnetra Kazhagam", "MDMK")
_add("Viduthalai Chiruthaigal Katchi", "VCK")
_add("Asom Gana Parishad", "AGP")
_add("Samata Party", "SAP")
_add("Janata Party", "JP")
_add("Independent", "IND", "Left Democratic Front (Kerala)", "Independent politician", "Independent (politician)", "IND", "Independent (India)")
_add("None Of The Above", "NOTA")
_add("Jammu & Kashmir Peoples Democratic Party", "PDP", "Jammu and Kashmir Peoples Democratic Party",
     "Jammu and Kashmir People's Democratic Party")
_add("Mizo National Front", "MNF")
_add("Sikkim Democratic Front", "SDF")
_add("Sikkim Krantikari Morcha", "SKM")
_add("National Peoples Party", "NPP", "National People's Party", "National People'S Party")
_add("Naga Peoples Front", "NPF", "Nagaland Peoples Front", "Naga People's Front")
_add("All India United Democratic Front", "AIUDF", "Assam United Democratic Front")
_add("Socialist Unity Centre Of India (Communist)", "SUCI(C)", "Socialist Unity Centre of India (Communist)")
_add("Republican Party Of India (A)", "RPI(A)", "Republican Party of India (A)", "Republican Party Of India(A)", "Republican Party of India (Athawale)")
_add("Republican Party Of India", "RPI", "Republican Party of India")
_add("Janasena Party", "JSP", "Jana Sena Party")
_add("Kerala Congress (M)", "KC(M)", "KEC(M)")
_add("Kerala Congress", "KC", "KEC")
_add("Hindustani Awam Morcha (Secular)", "HAM(S)", "Hindustani Awam Morcha")
_add("Ajsu Party", "AJSU", "AJSU Party", "All Jharkhand Students Union")
_add("Vanchit Bahujan Aaghadi", "VBA", "Vanchit Bahujan Aghadi")
_add("Naam Tamilar Katchi", "NTK")
_add("Rashtriya Loktantrik Party", "RLP")
_add("Rashtriya Lok Samta Party", "RLSP")
_add("Desiya Murpokku Dravida Kazhagam", "DMDK")
_add("Peoples Party Of India (Democratic)", "PPID", "Peoples Party of India (Democratic)", "People's Party of India (Democratic)")
_add("Zoram Peoples Movement", "ZPM", "Zoram People's Movement")
_add("Voice of the People Party", "VPP")
_add("Aazad Samaj Party (Kanshi Ram)", "ASP(KR)", "Azad Samaj Party (Kanshi Ram)")
_add("Swabhimani Paksha", "SWP", "Swabhimani Paksha")
_add("Bharat Adivasi Party", "BAP")
_add("Hindustan Awam Morcha", "HAM")
_add("Pragatishil Samajwadi Party (Lohia)", "PSP(L)", "Pragatishil Samajwadi Party (Lohia)")
_add("Rashtriya Samaj Paksha", "RSPK")
_add("Lok Dal", "LD")
_add("Haryana Janhit Congress (Bl)", "HJC(BL)", "Haryana Janhit Congress (BL)")
_add("Himachal Vikas Congress", "HVC")
_add("Haryana Vikas Party", "HVP")
_add("Maharashtra Rashtravadi Congress", "MRC")
_add("Federal Party of Manipur", "FPM")
_add("Manipur Peoples Party", "MPP", "Manipur People's Party")
_add("Manipur State Congress Party", "MSCP")
_add("Arunachal Congress", "AC")
_add("Sikkim Sangram Parishad", "SSP")
_add("Tamil Maanila Congress (Moopanar)", "TMC(M)", "Tamil Maanila Congress")
_add("Puthiya Tamilagam", "PT")
_add("Peasants And Workers Party Of India", "PWPI", "Peasants and Workers Party of India")
_add("Hill State Peoples Democratic Party", "HSPDP", "Hill State People's Democratic Party")
_add("Nationalist Democratic Progressive Party", "NDPP")
_add("Rashtriya Samanta Dal", "RSD")
_add("Uttarakhand Kranti Dal", "UKD")
_add("United Democratic Party", "UDP")
_add("United Goans Democratic Party", "UGDP")
_add("Akhil Bharat Hindu Mahasabha", "ABHM")
_add("All India N.R. Congress", "AINRC")
_add("Jharkhand Vikas Morcha (Prajatantrik)", "JVM(P)")
_add("Bodoland Peoples Front", "BPF", "Bodaland Peoples Front")
_add("Akhil Bharatiya Loktantrik Congress", "ABLTC", "Akhil Bhartiya Lok Tantrik Congress")
_add("Pyramid Party Of India", "PPOI", "Pyramid Party of India")
_add("Bharipa Bahujan Mahasangh", "BBM")
_add("Mahagujarat Janta Party", "MGJP")
_add("National Lokhind Party", "NLHP")

_STOP = {"of", "and", "the", "for", "in", "party", "dal", "india", "indian"}


def _auto_short(canon, taken):
    words = re.findall(r"[A-Za-z]+", canon)
    code = "".join(w[0].upper() for w in words if w.lower() not in SMALL) or canon[:4].upper()
    c, i = code, 2
    while c in taken:
        c = f"{code}{i}"; i += 1
    return c


def _pretty(raw):
    """auto-generated canonical = ECI spelling (whitespace/mojibake cleaned, ECI casing kept)"""
    return _clean(raw)


_AUTO = {}          # key -> (canonical, short)  for parties without a curated entry
_TAKEN = {v[1] for v in _CUR.values()}


SEEN = {}   # (source, raw spelling) -> (canonical, short, curated) ; written by write_table()


def canonical(raw, source=""):
    """Return (canonical_name, short_code, curated:bool)."""
    if raw is None or str(raw).strip() in ("", "nan"):
        return ("", "", True)
    k = key(raw)
    if k in _CUR:
        res = (*_CUR[k], True)
    else:
        if k not in _AUTO:
            canon = _pretty(raw)
            short = _auto_short(canon, _TAKEN)
            _TAKEN.add(short)
            _AUTO[k] = (canon, short)
        res = (*_AUTO[k], False)
    if source:
        SEEN[(source, re.sub(r"\s+", " ", str(raw)).strip())] = res
    return res


def write_table(path):
    import csv
    rows = sorted(SEEN.items(), key=lambda kv: (kv[1][0].lower(), kv[0][0], kv[0][1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "raw_name", "canonical_name", "short_code", "curated"])
        for (src, raw), (c, s, cur) in rows:
            w.writerow([src, raw, c, s, int(cur)])
