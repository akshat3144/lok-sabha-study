"""Build data/ref/ministry_map.csv: every distinct normalised ministry label (LS13-18) -> ministry_canonical,
ministry_group, lineage_family, predecessor_of, notes.

    python -I -X utf8 pipeline/ministry_build.py [path/to/questions.parquet]

The decisions are hand-coded below (RULES).  Policy, applied conservatively:
  * spelling / spacing / case / '&' vs 'AND' / truncation variants           -> collapsed silently;
  * plain RENAMES (one ministry, new name, same remit)                       -> collapsed to the latest name,
    the old label is recorded in `predecessor_of`;
  * MERGERS, SPLITS and re-organisations (remit changes)                     -> kept as separate canonical
    entities; `lineage_family` links them so analysts can pool them if they want.
Row counts / LS ranges (n_rows, ls_first, ls_last) are taken from the parquet if given (default
data/clean/questions.parquet) and are informational only.
clean.py reads this CSV; any label not in it is reported loudly (see clean.py: ministry_unmapped).
"""
import csv
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "ref", "ministry_map.csv")

SEC, ECO, INF = "security", "economy_finance", "infrastructure_transport"
SOC, AGR, ENE = "social_welfare_health_education", "agriculture_rural", "energy_environment_resources"
IND, GOV = "industry_commerce_labour", "governance_law_other"

# canonical name -> (group, lineage_family, [normalised source labels], notes)
# labels = clean.norm_ministry(raw): upper-case, whitespace collapsed, '&' -> 'AND'.
# a label listed under a canonical name that differs from the label itself is a rename (see predecessor_of).
RULES = {
    # ------------------------------------------------------------------ security
    "Home Affairs": (SEC, "Home Affairs", ["HOME AFFAIRS"], ""),
    "Defence": (SEC, "Defence", ["DEFENCE"], ""),
    "External Affairs": (SEC, "External Affairs", ["EXTERNAL AFFAIRS"],
                         "Overseas Indian Affairs was merged into External Affairs in 2016 but is kept separate here"),
    # ------------------------------------------------------------------ economy / finance
    "Finance": (ECO, "Finance", ["FINANCE"], ""),
    "Finance and Company Affairs": (ECO, "Finance", ["FINANCE AND COMPANY AFFAIRS"],
                                    "LS13 combined ministry; Company Affairs was later separated (see Corporate Affairs)"),
    "Disinvestment": (ECO, "Finance", ["DISINVESTMENT"], "LS13 only; later a Department inside Finance"),
    "Corporate Affairs": (ECO, "Corporate Affairs", ["CORPORATE AFFAIRS", "COMPANY AFFAIRS"],
                          "'Ministry of Company Affairs' was renamed Corporate Affairs (2007)"),
    "Planning": (ECO, "Planning", ["PLANNING"],
                 "Planning Commission portfolio; the LS16-18 label PLANNING covers the NITI Aayog era - not split"),
    "Statistics and Programme Implementation": (ECO, "Statistics and Programme Implementation",
                                                ["STATISTICS AND PROGRAMME IMPLEMENTATION"], ""),
    # ------------------------------------------------------------------ infrastructure / transport / telecom / urban
    "Railways": (INF, "Railways", ["RAILWAYS"], ""),
    "Road Transport and Highways": (INF, "Roads and Shipping", ["ROAD TRANSPORT AND HIGHWAYS"],
                                    "LS14 label SHIPPING, ROAD TRANSPORT AND HIGHWAYS is a merged ministry kept separate"),
    "Surface Transport": (INF, "Roads and Shipping", ["SURFACE TRANSPORT"],
                          "LS13 ministry covering roads + shipping; predecessor of the later Road Transport and Highways / Shipping split"),
    "Shipping, Road Transport and Highways": (INF, "Roads and Shipping", ["SHIPPING, ROAD TRANSPORT AND HIGHWAYS"],
                                              "LS14 merged ministry (split again in 2009)"),
    "Ports, Shipping and Waterways": (INF, "Roads and Shipping", ["PORTS, SHIPPING AND WATERWAYS", "SHIPPING"],
                                      "'Ministry of Shipping' renamed Ports, Shipping and Waterways (2020-21); plain SHIPPING rows collapse into it"),
    "Civil Aviation": (INF, "Civil Aviation", ["CIVIL AVIATION"], ""),
    "Communications": (INF, "Communications / IT", ["COMMUNICATIONS", "COMMUNICATION"],
                       "COMMUNICATION (LS18) is a typo/variant of COMMUNICATIONS"),
    "Communications and Information Technology": (INF, "Communications / IT",
                                                   ["COMMUNICATIONS AND INFORMATION TECHNOLOGY"],
                                                   "merged ministry 2004-2016 (telecom + IT); split into Communications and Electronics and IT in 2016"),
    "Information Technology": (INF, "Communications / IT", ["INFORMATION TECHNOLOGY"], "LS13 stand-alone IT ministry"),
    "Electronics and Information Technology": (INF, "Communications / IT",
                                               ["ELECTRONICS AND INFORMATION TECHNOLOGY"],
                                               "judgement call: grouped with telecom as digital infrastructure"),
    "Urban Development": (INF, "Urban and Housing", ["URBAN DEVELOPMENT"],
                          "merged with Housing and Urban Poverty Alleviation in 2017 to form Housing and Urban Affairs"),
    "Urban Development and Poverty Alleviation": (INF, "Urban and Housing",
                                                  ["URBAN DEVELOPMENT AND POVERTY ALLEVIATION", "URBAN DEVELOPMENT AND POVERTY ALLEVATION"],
                                                  "LS13 only; 'ALLEVATION' is a source typo"),
    "Urban Employment and Poverty Alleviation": (INF, "Urban and Housing", ["URBAN EMPLOYMENT AND POVERTY ALLEVIATION"],
                                                 "LS13-14; probably the predecessor of Housing and Urban Poverty Alleviation (2004) - unverified, kept separate"),
    "Housing and Urban Poverty Alleviation": (INF, "Urban and Housing", ["HOUSING AND URBAN POVERTY ALLEVIATION"], ""),
    "Housing and Urban Affairs": (INF, "Urban and Housing", ["HOUSING AND URBAN AFFAIRS"],
                                  "2017 merger of Urban Development and Housing and Urban Poverty Alleviation"),
    # ------------------------------------------------------------------ social / welfare / health / education
    "Health and Family Welfare": (SOC, "Health and Family Welfare", ["HEALTH AND FAMILY WELFARE"], ""),
    "Education": (SOC, "Education", ["EDUCATION", "HUMAN RESOURCE DEVELOPMENT"],
                  "'Human Resource Development' was renamed Education (2020); NB LS13 HRD also contained Women and Child "
                  "Development (separate ministry only from 2006) and, earlier, parts of Culture/Youth"),
    "Women and Child Development": (SOC, "Women and Child Development", ["WOMEN AND CHILD DEVELOPMENT"],
                                    "separate ministry from 2006; LS13 equivalent questions sit under Education (HRD)"),
    "Social Justice and Empowerment": (SOC, "Social Justice and Empowerment", ["SOCIAL JUSTICE AND EMPOWERMENT"], ""),
    "Tribal Affairs": (SOC, "Tribal Affairs", ["TRIBAL AFFAIRS"], ""),
    "Minority Affairs": (SOC, "Minority Affairs", ["MINORITY AFFAIRS"], ""),
    "AYUSH": (SOC, "AYUSH", ["AYUSH", "AYURVEDA,YOGA AND NATUROPATHY,UNANI,SIDDHA AND HOMEOPATHY (AYUSH)"],
              "long LS16-17 label is the same ministry as the LS18 short label 'AYUSH'"),
    "Youth Affairs and Sports": (SOC, "Youth Affairs and Sports", ["YOUTH AFFAIRS AND SPORTS"], ""),
    "Skill Development and Entrepreneurship": (SOC, "Skill Development", ["SKILL DEVELOPMENT AND ENTREPRENEURSHIP"],
                                               "judgement call: grouped with education/training rather than labour"),
    "Skill Development, Entrepreneurship, Youth Affairs and Sports": (
        SOC, "Skill Development", ["SKILL DEVELOPMENT, ENTREPRENEURSHIP,YOUTH AFFAIRS AND SPORTS"],
        "LS16 only (101 rows): label used while one minister held both portfolios; kept separate"),
    # ------------------------------------------------------------------ agriculture / rural
    "Agriculture and Farmers Welfare": (AGR, "Agriculture", ["AGRICULTURE AND FARMERS WELFARE", "AGRICULTURE"],
                                        "'Agriculture' renamed Agriculture and Farmers Welfare (2015); fisheries/animal husbandry "
                                        "were inside it until 2019 (see Fisheries, Animal Husbandry and Dairying)"),
    "Rural Development": (AGR, "Rural Development", ["RURAL DEVELOPMENT"], ""),
    "Panchayati Raj": (AGR, "Panchayati Raj", ["PANCHAYATI RAJ"], ""),
    "Cooperation": (AGR, "Cooperation", ["COOPERATION"], "new ministry 2021"),
    "Fisheries, Animal Husbandry and Dairying": (AGR, "Fisheries, Animal Husbandry and Dairying",
                                                 ["FISHERIES, ANIMAL HUSBANDRY AND DAIRYING"],
                                                 "new ministry 2019 (carved out of Agriculture)"),
    "Consumer Affairs, Food and Public Distribution": (
        AGR, "Consumer Affairs, Food and Public Distribution",
        ["CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION", "CONSUMER AFFAIRS AND PUBLIC DISTRIBUTION"],
        "judgement call: PDS/food procurement dominates, so grouped with agriculture; the LS13-only label "
        "'Consumer Affairs and Public Distribution' is treated as a short-lived variant (could be a 2001-02 reorganisation)"),
    # ------------------------------------------------------------------ energy / environment / resources
    "Power": (ENE, "Power", ["POWER"], ""),
    "Petroleum and Natural Gas": (ENE, "Petroleum and Natural Gas", ["PETROLEUM AND NATURAL GAS"], ""),
    "Coal": (ENE, "Coal and Mines", ["COAL"], ""),
    "Mines": (ENE, "Coal and Mines", ["MINES", "MINES AND MINERALS"],
              "LS13 'Mines and Minerals' treated as a name variant of Mines"),
    "Coal and Mines": (ENE, "Coal and Mines", ["COAL AND MINES"], "LS13-14 merged ministry"),
    "New and Renewable Energy": (ENE, "New and Renewable Energy", ["NEW AND RENEWABLE ENERGY", "NON-CONVENTIONAL ENERGY SOURCES"],
                                 "'Non-Conventional Energy Sources' renamed New and Renewable Energy (2006)"),
    "Atomic Energy": (ENE, "Atomic Energy", ["ATOMIC ENERGY"], "judgement call: grouped with energy"),
    "Environment, Forest and Climate Change": (
        ENE, "Environment",
        ["ENVIRONMENT, FOREST AND CLIMATE CHANGE", "ENVIRONMENT, FORESTS AND CLIMATE CHANGE", "ENVIRONMENT AND FORESTS"],
        "'Environment and Forests' renamed Environment, Forest and Climate Change (2014)"),
    "Water Resources, River Development and Ganga Rejuvenation": (
        ENE, "Water", ["WATER RESOURCES, RIVER DEVELOPMENT AND GANAGA REJUVENATION", "WATER RESOURCES"],
        "'Water Resources' renamed (2014); 'GANAGA' is a source typo; merged into Jal Shakti in 2019"),
    "Drinking Water and Sanitation": (ENE, "Water", ["DRINKING WATER AND SANITATION"],
                                      "ministry 2011-2019 (formerly a Department of Rural Development); merged into Jal Shakti"),
    "Jal Shakti": (ENE, "Water", ["JAL SHAKTI"], "2019 merger of Water Resources and Drinking Water and Sanitation"),
    "Earth Sciences": (ENE, "Earth Sciences", ["EARTH SCIENCES"], "judgement call: meteorology/ocean/geo resources"),
    "Ocean Development": (ENE, "Earth Sciences", ["OCEAN DEVELOPMENT"], "LS13-14; absorbed into Earth Sciences (2006)"),
    # ------------------------------------------------------------------ industry / commerce / labour
    "Commerce and Industry": (IND, "Commerce and Industry", ["COMMERCE AND INDUSTRY"], ""),
    "Textiles": (IND, "Textiles", ["TEXTILES"], ""),
    "Steel": (IND, "Steel", ["STEEL"], ""),
    "Heavy Industries": (IND, "Heavy Industries", ["HEAVY INDUSTRIES", "HEAVY INDUSTRIES AND PUBLIC ENTERPRISES"],
                         "'Heavy Industries and Public Enterprises' became Heavy Industries when Public Enterprises moved "
                         "to Finance (2019-21); collapsed as the continuing body"),
    "Micro, Small and Medium Enterprises": (IND, "MSME", ["MICRO, SMALL AND MEDIUM ENTERPRISES", "MICRO,SMALL AND MEDIUM ENTERPRISES"],
                                            "2007 merger of Small Scale Industries and Agro and Rural Industries"),
    "Small Scale Industries": (IND, "MSME", ["SMALL SCALE INDUSTRIES"], "LS13-14; merged into MSME"),
    "Agro and Rural Industries": (IND, "MSME", ["AGRO AND RURAL INDUSTRIES"], "LS13-14; merged into MSME"),
    "Small Scale Industries and Agro and Rural Industries": (
        IND, "MSME", ["SMALL SCALE INDUSTRIES AND RURAL AND AGRO INDUSTR"],
        "LS13 combined ministry; source label truncated ('INDUSTR')"),
    "Chemicals and Fertilizers": (IND, "Chemicals and Fertilizers", ["CHEMICALS AND FERTILIZERS"], ""),
    "Labour and Employment": (IND, "Labour", ["LABOUR AND EMPLOYMENT", "LABOUR"],
                              "'Labour' renamed Labour and Employment (2004)"),
    "Tourism": (IND, "Tourism and Culture", ["TOURISM"], ""),
    "Tourism and Culture": (IND, "Tourism and Culture", ["TOURISM AND CULTURE"],
                            "LS13-14 merged ministry; judgement call: grouped with tourism (economic)"),
    "Food Processing Industries": (IND, "Food Processing Industries", ["FOOD PROCESSING INDUSTRIES"],
                                   "judgement call: manufacturing rather than farm sector"),
    # ------------------------------------------------------------------ governance / law / other
    "Law and Justice": (GOV, "Law and Justice", ["LAW AND JUSTICE"], ""),
    "Law, Justice and Company Affairs": (GOV, "Law and Justice", ["LAW , JUSTICE AND COMPANY AFFAIRS"],
                                         "LS13 merged ministry (Company Affairs later moved out)"),
    "Parliamentary Affairs": (GOV, "Parliamentary Affairs", ["PARLIAMENTARY AFFAIRS"], ""),
    "Prime Minister": (GOV, "Prime Minister", ["PRIME MINISTER"], "questions answered by the PM"),
    "Personnel, Public Grievances and Pensions": (GOV, "Personnel, Public Grievances and Pensions",
                                                  ["PERSONNEL, PUBLIC GRIEVANCES AND PENSIONS", "PERSONNEL,PUBLIC GRIEVANCES AND PENSIONS"], ""),
    "Information and Broadcasting": (GOV, "Information and Broadcasting", ["INFORMATION AND BROADCASTING"], ""),
    "Science and Technology": (GOV, "Science and Technology", ["SCIENCE AND TECHNOLOGY"], "no science group exists: catch-all"),
    "Space": (GOV, "Space", ["SPACE"], "no science group exists: catch-all"),
    "Culture": (GOV, "Tourism and Culture", ["CULTURE"], ""),
    "Development of North Eastern Region": (GOV, "Development of North Eastern Region",
                                            ["DEVELOPMENT OF NORTH EASTERN REGION"], ""),
    "Overseas Indian Affairs": (GOV, "Overseas Indian Affairs", ["OVERSEAS INDIAN AFFAIRS", "NON-RESIDENT INDIANS AFFAIRS"],
                                "'Non-Resident Indians' Affairs' renamed Overseas Indian Affairs (2004); merged into External "
                                "Affairs in 2016; kept out of the security group"),
}


def squash(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def build_map():
    """-> {normalised label: dict(canonical, group, lineage_family, predecessor_of, notes)}"""
    out = {}
    for canon, (grp, fam, labels, notes) in RULES.items():
        for lab in labels:
            if lab in out:
                raise ValueError(f"label listed twice: {lab}")
            # spelling / spacing / typo variants of the canonical name are not "predecessors"
            same = difflib.SequenceMatcher(None, squash(lab), squash(canon)).ratio() >= 0.9
            out[lab] = {"ministry_canonical": canon, "ministry_group": grp, "lineage_family": fam,
                        "predecessor_of": "" if same else canon, "notes": notes}
    return out


def counts_from_parquet(path):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["ls", "ministry", "ministry_raw"]).to_pandas()
    g = t.groupby("ministry").agg(n_rows=("ls", "size"), ls_first=("ls", "min"), ls_last=("ls", "max"),
                                  raw=("ministry_raw", lambda s: " || ".join(sorted({repr(x) for x in s}))))
    return g.to_dict("index")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pq_path = argv[0] if argv else os.path.join(ROOT, "data", "clean", "questions.parquet")
    mp = build_map()
    cnt = counts_from_parquet(pq_path) if os.path.exists(pq_path) else {}
    missing = [k for k in cnt if k not in mp]
    if missing:
        print("UNMAPPED ministries (add to RULES):", missing)
    keys = sorted(set(mp) | set(cnt), key=lambda k: (mp.get(k, {}).get("ministry_group", "zzz"),
                                                      mp.get(k, {}).get("ministry_canonical", "zzz"), k))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ministry", "ministry_canonical", "ministry_group", "lineage_family", "predecessor_of", "notes",
                    "ls_first", "ls_last", "n_rows", "ministry_raw_variants"])
        for k in keys:
            m = mp.get(k, {})
            c = cnt.get(k, {})
            w.writerow([k, m.get("ministry_canonical", ""), m.get("ministry_group", ""), m.get("lineage_family", ""),
                        m.get("predecessor_of", ""), m.get("notes", ""), c.get("ls_first", ""), c.get("ls_last", ""),
                        c.get("n_rows", ""), c.get("raw", "")])
    n_canon = len({v["ministry_canonical"] for v in mp.values()})
    print(f"wrote {OUT}: {len(keys)} labels -> {n_canon} canonical ministries; unmapped={len(missing)}")
    return missing


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
