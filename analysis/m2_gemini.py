"""Measure M2: label every question part with Gemini (Vertex AI).

    python -I -X utf8 analysis/m2_gemini.py estimate --models gemini-3.5-flash-lite,gemini-3.1-flash-lite,...
    python -I -X utf8 analysis/m2_gemini.py pilot  --project P --qids-file pilot_qids.csv --models a,b,c
    python -I -X utf8 analysis/m2_gemini.py batch  --project P --gcs-bucket B --model gemini-3.1-flash-lite
    python -I -X utf8 analysis/m2_gemini.py ocr    --project P --limit 50

Auth: Application Default Credentials only (`gcloud auth application-default login`). No keys are read or stored.
The unit of one request is ONE QUESTION with all its parts; the model returns one JSON entry per part label.
Details, limits and the resume logic: README_M2.md. Prompt text: m2_prompt.py (hash stored with every label).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # works under `python -I`
import codebook as CB  # noqa: E402
import m2_prompt as MP  # noqa: E402

ROOT = os.path.dirname(HERE)
MEASURES = os.path.join(ROOT, "data", "measures")
PARTS_PARQUET = os.path.join(MEASURES, "parts.parquet")
QUESTIONS_PARQUET = os.path.join(ROOT, "data", "final", "questions_analysis.parquet")
LABELS_PARQUET = os.path.join(MEASURES, "m2_labels.parquet")
OCR_PARQUET = os.path.join(MEASURES, "ocr_text.parquet")
BATCH_DIR = os.path.join(MEASURES, "m2_batch")
FALLBACK_DIR = os.path.join(ROOT, "data", "raw", "docs_fallback")
DOCX_PDF_DIR = os.path.join(ROOT, "data", "derived", "ocr_pdf")  # image-only .docx -> PDF (analysis/docx_to_pdf.py)

# ----------------------------------------------------------------------------------------------------------
# Prices (USD per 1M tokens). Source: https://ai.google.dev/gemini-api/docs/pricing, as of 2026-10-16.
# These are Gemini Developer API prices; CONFIRM the Vertex AI prices in the Cloud console before a large run.
# Each entry: list of (effective_from_date, input, output), BATCH pricing (= 50% of real-time).
# ----------------------------------------------------------------------------------------------------------
PRICES_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICES_AS_OF = "2026-10-16"
BATCH_PRICES = {
    "gemini-3.5-flash-lite": [("2000-01-01", 0.15, 1.25)],
    "gemini-3.1-flash-lite": [("2000-01-01", 0.125, 0.75)],
    "gemini-2.5-flash-lite": [("2000-01-01", 0.05, 0.20)],
    "gemini-3.8-flash": [("2000-01-01", 0.375, 1.875), ("2027-01-01", 0.75, 3.75)],
}
SYNC_MULTIPLIER = 2.0  # real-time price = 2 x batch price (batch is a 50% discount)

# Vertex AI batch prediction limits (docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/
# batch-inference, read 2026-10-17): up to 200,000 requests and 1 GB input file per job; queue wait up to 72 h;
# most jobs finish within 24 h of starting. Defaults below stay well inside those limits.
VERTEX_MAX_REQUESTS_PER_JOB = 200_000
VERTEX_MAX_FILE_BYTES = 1_000_000_000
DEFAULT_CHUNK_REQUESTS = 100_000
DEFAULT_CHUNK_MB = 900
RETRY_WORKERS = 8  # parallel synchronous retries of invalid batch rows

# Output tokens per part, MEASURED offline with measure_output_tokens() (see `estimate --measure`): a realistic
# response (rationale <= 15 words, or no rationale) rendered for 200 real questions and counted with json_tokens().
# Thinking tokens are assumed ~0 at the minimum thinking setting.
# Measured 2026-10-17 on 200 random real questions (833 parts, seed 1): 293.4 / 175.7 tokens per response.
MEASURED_OUT_TOKENS = {True: {"per_part": 69.25, "overhead": 5.0},    # with rationale (<= 15 words)
                       False: {"per_part": 40.99, "overhead": 5.0}}   # --no-rationale


class CostCapError(RuntimeError):
    pass


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------------------------------------
# Token / cost estimation (offline: chars/4 heuristic; non-ASCII text, e.g. Devanagari, counted at 2 chars/token)
# ----------------------------------------------------------------------------------------------------------
def est_tokens(text):
    if not text:
        return 0.0
    n_ascii = len(text.encode("ascii", "ignore"))
    return n_ascii / 4.0 + (len(text) - n_ascii) / 2.0


def price_for(model, on=None, override=None):
    if override:
        return override
    if model not in BATCH_PRICES:
        raise KeyError(f"no price known for {model!r}; pass --price-in/--price-out (USD per 1M, batch) "
                       f"or add it to BATCH_PRICES")
    on = on or datetime.date.today().isoformat()
    cur = None
    for eff, pin, pout in BATCH_PRICES[model]:
        if eff <= on:
            cur = (pin, pout)
    return cur


def cost_usd(model, in_tok, out_tok, mode="batch", on=None, override=None):
    pin, pout = price_for(model, on, override)
    c = (in_tok * pin + out_tok * pout) / 1e6
    return c * (SYNC_MULTIPLIER if mode == "sync" else 1.0)


# ----------------------------------------------------------------------------------------------------------
# Generation config (thinking minimum, temperature 0, structured output)
# ----------------------------------------------------------------------------------------------------------
@dataclass
class GenCfg:
    temperature: float = None   # None = auto: 0 for Gemini 2.x, 1.0 for Gemini 3+ (Google: below 1.0 may loop)
    thinking: str = "auto"      # auto | off | budget:N | level:MINIMAL|LOW|MEDIUM|HIGH
    rationale: bool = True      # False = cheaper variant without the rationale field
    seed: int = 0               # fixed seed for reproducibility (needed when temperature > 0)
    strip_property_ordering: bool = False  # drop "propertyOrdering" from the schema if Vertex rejects it
    base_max_tokens: int = 600
    per_part_max_tokens: int = 160

    def max_tokens(self, n_parts):
        return min(8192, self.base_max_tokens + self.per_part_max_tokens * n_parts)

    def temp_for(self, model):
        if self.temperature is not None:
            return self.temperature
        return 0.0 if "gemini-2" in model else 1.0

    def thinking_params(self, model):
        """REST-style dict for generationConfig.thinkingConfig, or None to leave the model default.
        auto (thinking tokens are billed as output, so use the lowest level the model allows):
          gemini-2.5*            -> thinkingBudget 0 (docs: thinking can be turned off for 2.5 Flash / Flash-Lite;
                                    the same page lists a 512 minimum for Flash-Lite: if 0 is rejected use budget:512)
          gemini-3.x *-lite      -> thinkingLevel MINIMAL (supported, and the default, for 3.5 / 3.1 Flash-Lite)
          other gemini-3.x (3.8-flash) -> thinkingLevel LOW (MINIMAL is NOT supported there; default is MEDIUM)
        Never send thinkingLevel and thinkingBudget together (API error)."""
        t = (self.thinking or "auto").strip()
        if t == "off":
            return None
        if t == "auto":
            if "2.5" in model:
                return {"thinkingBudget": 0}
            return {"thinkingLevel": "MINIMAL" if "lite" in model else "LOW"}
        if t.startswith("budget:"):
            return {"thinkingBudget": int(t.split(":", 1)[1])}
        if t.startswith("level:"):
            return {"thinkingLevel": t.split(":", 1)[1].upper()}
        raise ValueError(f"bad --thinking {t!r}")

    def system_prompt(self):
        return MP.system_prompt(self.rationale)

    def phash(self):
        return MP.prompt_hash(self.rationale)

    def schema(self):
        return MP.response_schema(self.rationale)

    def fingerprint(self):
        return {"temperature": self.temperature, "thinking": self.thinking, "rationale": self.rationale,
                "seed": self.seed, "strip_property_ordering": self.strip_property_ordering,
                "base_max_tokens": self.base_max_tokens, "per_part_max_tokens": self.per_part_max_tokens}


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg.fingerprint(), sort_keys=True).encode()).hexdigest()[:6]


def vertex_schema(node):
    """Vertex REST `responseSchema` wants OpenAPI-style upper-case type names (STRING, OBJECT, ARRAY)."""
    if isinstance(node, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else vertex_schema(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [vertex_schema(x) for x in node]
    return node


def strip_ordering(node):
    if isinstance(node, dict):
        return {k: strip_ordering(v) for k, v in node.items() if k != "propertyOrdering"}
    if isinstance(node, list):
        return [strip_ordering(x) for x in node]
    return node


def schema_for(cfg):
    sch = cfg.schema()
    return strip_ordering(sch) if cfg.strip_property_ordering else sch


def generation_config_dict(model, cfg, n_parts):
    g = {"temperature": cfg.temp_for(model), "seed": cfg.seed, "maxOutputTokens": cfg.max_tokens(n_parts),
         "responseMimeType": "application/json", "responseSchema": vertex_schema(schema_for(cfg))}
    tp = cfg.thinking_params(model)
    if tp:
        g["thinkingConfig"] = tp
    return g


def request_body(model, user_text, n_parts, cfg):
    """One Vertex batch request line (REST JSON): system instruction = codebook, user turn = the question."""
    return {"request": {
        "systemInstruction": {"parts": [{"text": cfg.system_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": generation_config_dict(model, cfg, n_parts)}}


def request_key(user_text):
    return hashlib.sha1(user_text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------------------------------------
FLAG_COLS = ["qa_mismatch", "answer_garbled"]  # kept in M2 labelling; carried to m2_labels.parquet


@dataclass
class Unit:
    qid: str
    question_text: str
    answer_text: str
    annexure_text: str
    parts: list = field(default_factory=list)  # [(part_label, part_text)]
    qa_mismatch: bool = False    # pipeline flags carried into the label rows (analysis excludes them consistently)
    answer_garbled: bool = False

    @property
    def labels(self):
        return [p[0] for p in self.parts]

    def user_prompt(self, retry_error=None):
        return MP.build_user_prompt(self.question_text, self.answer_text, self.annexure_text, self.parts, retry_error)


class Store:
    """qid -> Unit, built from parts.parquet (part structure) and questions_analysis.parquet (full texts)."""

    def __init__(self, parts_by_qid, qa):
        self.parts_by_qid = parts_by_qid
        self.qa = qa  # DataFrame indexed by qid with question_text, answer_text, annexure_text
        self.qids = sorted(q for q in parts_by_qid if q in qa.index)
        self.n_dropped = len(parts_by_qid) - len(self.qids)

    @classmethod
    def from_frames(cls, parts_df, qa_df):
        parts = {}
        for q, lab, txt in zip(parts_df["qid"], parts_df["part_label"], parts_df["question_part"]):
            parts.setdefault(q, []).append((lab, txt or ""))
        qa = qa_df.drop_duplicates("qid").set_index("qid")
        for c in ("question_text", "answer_text", "annexure_text"):
            if c not in qa:
                qa[c] = ""
        qa[["question_text", "answer_text", "annexure_text"]] = qa[["question_text", "answer_text",
                                                                      "annexure_text"]].fillna("")
        for c in FLAG_COLS:
            qa[c] = qa[c].fillna(False).astype(bool) if c in qa else False
        return cls(parts, qa[["question_text", "answer_text", "annexure_text"] + FLAG_COLS])

    @classmethod
    def load(cls, sel, use_ocr=True):
        parts_df = pd.read_parquet(PARTS_PARQUET, columns=["qid", "part_label", "question_part"])
        qids = select_qids(parts_df["qid"].drop_duplicates().tolist(), sel)
        parts_df = parts_df[parts_df["qid"].isin(set(qids))]
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(QUESTIONS_PARQUET).schema_arrow.names)
        cols = ["qid", "question_text", "answer_text", "annexure_text"] + [c for c in FLAG_COLS if c in have]
        qa_df = pd.read_parquet(QUESTIONS_PARQUET, columns=cols, filters=[("qid", "in", qids)])
        store = cls.from_frames(parts_df, qa_df)
        if use_ocr and os.path.exists(OCR_PARQUET):
            ocr = pd.read_parquet(OCR_PARQUET)
            ocr = ocr[(ocr["status"] == "ok") & ocr["text"].fillna("").str.strip().ne("")]
            m = dict(zip(ocr["qid"], ocr["text"]))
            empty = store.qa["answer_text"].str.strip().eq("")
            fill = [q for q in store.qa.index[empty] if q in m]
            if fill:
                store.qa.loc[fill, "answer_text"] = [m[q] for q in fill]
        if getattr(sel, "skip_no_answer", False):
            has = store.qa["answer_text"].str.strip().ne("") | store.qa["annexure_text"].str.strip().ne("")
            keep = set(store.qa.index[has])
            store.qids = [q for q in store.qids if q in keep]
        return store

    def unit(self, qid):
        r = self.qa.loc[qid]
        return Unit(qid, r["question_text"], r["answer_text"], r["annexure_text"], self.parts_by_qid[qid],
                    bool(r["qa_mismatch"]), bool(r["answer_garbled"]))


def read_id_list(path):
    """qids from a .txt (one per line), .csv or .parquet (column qid or part_id, else first column).
    part_ids like 'ls17-...#a' are reduced to the qid."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        with open(path, encoding="utf-8") as f:
            vals = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        return sorted({v.split("#")[0] for v in vals})
    col = "qid" if "qid" in df else ("part_id" if "part_id" in df else df.columns[0])
    return sorted({str(v).split("#")[0] for v in df[col].dropna()})


def select_qids(all_qids, sel):
    qids = sorted(set(all_qids))
    if getattr(sel, "qids_file", None):
        want = set(read_id_list(sel.qids_file))
        qids = [q for q in qids if q in want]
    if getattr(sel, "ls", None):
        pre = tuple(f"ls{n}-" for n in sel.ls)
        qids = [q for q in qids if q.startswith(pre)]
    if getattr(sel, "sample", None):
        rng = random.Random(sel.seed)
        qids = sorted(rng.sample(qids, min(sel.sample, len(qids))))
    if getattr(sel, "limit", None):
        qids = qids[:sel.limit]
    return qids


# ----------------------------------------------------------------------------------------------------------
# Response validation
# ----------------------------------------------------------------------------------------------------------
def clip_words(text, n):
    """Defensive cap on stored rationales (the prompt asks for <= n words)."""
    w = str(text).split()
    return " ".join(w) if len(w) <= n else " ".join(w[:n]) + " ..."


_LEAD_LABEL = re.compile(r"^\s*[(\[]?\s*(_all|[a-z]{1,4}|\d{1,2})\s*[)\].:]\s+\S", re.I)


def _norm_label(x):
    """'a', '(a)', 'A.' -> 'a'. The model sometimes echoes the whole part ('(a) the details of …'): keep the label."""
    s = str(x)
    m = _LEAD_LABEL.match(s)
    return (m.group(1) if m else s.strip().strip("()[] .")).lower()


def validate_text(text, expected_labels, need_rationale=True):
    """-> (list of part dicts in expected order, None) or (None, error message)."""
    if text is None or not str(text).strip():
        return None, "empty response"
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except Exception as e:  # noqa: BLE001
        return None, f"not valid JSON ({str(e)[:80]})"
    parts = obj.get("parts") if isinstance(obj, dict) else None
    if not isinstance(parts, list):
        return None, "missing 'parts' array"
    want = [_norm_label(x) for x in expected_labels]
    got = {}
    for p in parts:
        if not isinstance(p, dict) or "part_label" not in p:
            return None, "entry without part_label"
        k = _norm_label(p["part_label"])
        if k in got:
            return None, f"duplicate part_label {p['part_label']!r}"
        got[k] = p
    missing = [x for x in want if x not in got]
    extra = [x for x in got if x not in want]
    if missing or extra:
        return None, f"part_labels must be exactly {list(expected_labels)} (missing {missing}, unexpected {extra})"
    out = []
    for lab, k in zip(expected_labels, want):
        p = got[k]
        for key, allowed in (("label", CB.LABELS), ("request_type", CB.REQUEST_TYPES),
                             ("evasion_type", CB.EVASION_CHOICES)):
            if p.get(key) not in allowed:
                return None, f"part {lab}: {key}={p.get(key)!r} not in {allowed}"
        rat = p.get("rationale")
        if rat is None and not need_rationale:
            rat = ""
        if not isinstance(rat, str):
            return None, f"part {lab}: rationale must be a string"
        # codebook: FULL has no evasion type (same rule as the human gold labels, validate.build_gold)
        ev = "NONE_APPLICABLE" if p["label"] == "FULL" else p["evasion_type"]
        out.append({"part_label": lab, "label": p["label"], "request_type": p["request_type"],
                    "evasion_type": ev, "rationale": clip_words(rat, CB.RATIONALE_MAX_WORDS)})
    return out, None


@dataclass
class GenResult:
    text: str = None
    input_tokens: int = 0
    output_tokens: int = 0
    model_version: str = None
    finish_reason: str = None
    error: str = None
    cached_tokens: int = 0  # implicit-cache hits reported by the API (billed at the cached rate)


LABEL_COLS = ["part_id", "qid", "part_label", "label", "evasion_type", "request_type", "rationale", "model_id",
              "codebook_version", "prompt_hash", "status", "input_tokens", "output_tokens", "model_version",
              "cached_tokens", "attempts", "source", "qa_mismatch", "answer_garbled"]


def rows_for_unit(unit, model, parsed, error, results, source, attempts, phash=None):
    """One label row per part. Tokens (request level) are split evenly over the parts so sums stay exact."""
    n = max(1, len(unit.parts))
    in_tok = sum(r.input_tokens for r in results) / n
    out_tok = sum(r.output_tokens for r in results) / n
    cached = sum(getattr(r, 'cached_tokens', 0) or 0 for r in results) / n
    mv = next((r.model_version for r in reversed(results) if r.model_version), None)
    rows = []
    for i, (lab, _) in enumerate(unit.parts):
        base = {"part_id": f"{unit.qid}#{lab}", "qid": unit.qid, "part_label": lab, "model_id": model,
                "codebook_version": CB.CODEBOOK_VERSION, "prompt_hash": phash or MP.PROMPT_HASH, "input_tokens": in_tok,
                "output_tokens": out_tok, "model_version": mv, "cached_tokens": cached, "attempts": attempts, "source": source,
                "qa_mismatch": unit.qa_mismatch, "answer_garbled": unit.answer_garbled}
        if parsed:
            base.update(parsed[i])
            base["status"] = "ok"
        else:
            base.update({"label": None, "evasion_type": None, "request_type": None,
                         "rationale": (error or "")[:300], "status": "invalid"})
        rows.append(base)
    return rows


def to_frame(rows):
    df = pd.DataFrame(rows)
    for c in LABEL_COLS:
        if c not in df:
            df[c] = None
    return df[LABEL_COLS]


# ----------------------------------------------------------------------------------------------------------
# Clients (thin wrappers so tests can inject mocks)
# ----------------------------------------------------------------------------------------------------------
class SyncClient:
    """Real synchronous client: google-genai on Vertex AI with Application Default Credentials."""

    def __init__(self, project, location="global"):
        from google import genai
        self._genai = genai
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def _config(self, model, cfg, n_parts):
        from google.genai import types
        tp = cfg.thinking_params(model)
        tc = None
        if tp and "thinkingBudget" in tp:
            tc = types.ThinkingConfig(thinking_budget=tp["thinkingBudget"])
        elif tp:
            tc = types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, tp["thinkingLevel"]))
        return types.GenerateContentConfig(
            system_instruction=cfg.system_prompt(), temperature=cfg.temp_for(model), seed=cfg.seed,
            max_output_tokens=cfg.max_tokens(n_parts), response_mime_type="application/json",
            response_schema=schema_for(cfg), thinking_config=tc)

    def _call(self, fn, tries=6):
        delay = 2.0
        for i in range(tries):
            try:
                return fn(), None
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", None)
                if i + 1 < tries and (code in (429, 500, 502, 503, 504) or code is None):
                    time.sleep(delay + random.random())
                    delay = min(delay * 2, 60)
                    continue
                return None, f"{type(e).__name__}: {str(e)[:300]}"

    @staticmethod
    def _result(resp):
        u = resp.usage_metadata
        fr = None
        if resp.candidates:
            fr = getattr(resp.candidates[0].finish_reason, "name", str(resp.candidates[0].finish_reason))
        out = ((getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)) if u else 0
        return GenResult(text=resp.text, input_tokens=(getattr(u, "prompt_token_count", 0) or 0) if u else 0,
                         output_tokens=out, model_version=getattr(resp, "model_version", None), finish_reason=fr,
                         cached_tokens=(getattr(u, "cached_content_token_count", 0) or 0) if u else 0)

    def generate(self, model, user_text, cfg, n_parts):
        resp, err = self._call(lambda: self.client.models.generate_content(
            model=model, contents=user_text, config=self._config(model, cfg, n_parts)))
        return GenResult(error=err) if err else self._result(resp)

    def generate_pdf(self, model, prompt, pdf_bytes, max_tokens=32000):
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens)
        resp, err = self._call(lambda: self.client.models.generate_content(
            model=model, contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
            config=cfg))
        return GenResult(error=err) if err else self._result(resp)


class VertexBatchClient:
    TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PARTIALLY_SUCCEEDED",
                "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED"}

    def __init__(self, project, location="global"):
        from google import genai
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def create(self, model, src_uri, dest_uri, display_name):
        from google.genai.types import CreateBatchJobConfig
        job = self.client.batches.create(model=model, src=src_uri,
                                         config=CreateBatchJobConfig(dest=dest_uri, display_name=display_name))
        return job.name

    def get(self, name):
        job = self.client.batches.get(name=name)
        return {"state": getattr(job.state, "name", str(job.state)), "error": str(job.error) if job.error else None}


class GcsCli:
    """GCS through the `gcloud storage` CLI (already authenticated by the user; no extra Python dependency)."""

    def __init__(self, project=None):
        self.gcloud = shutil.which("gcloud")
        if not self.gcloud:
            raise RuntimeError("`gcloud` not found on PATH (install the Google Cloud CLI)")
        self.project = project

    def _run(self, args):
        cmd = [self.gcloud, "storage"] + args + ([f"--project={self.project}"] if self.project else [])
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"gcloud storage {' '.join(args[:2])} failed: {r.stderr.strip()[:500]}")
        return r.stdout

    def upload(self, local, uri):
        self._run(["cp", local, uri])

    def download(self, uri, local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        self._run(["cp", uri, local])

    def list_jsonl(self, prefix):
        try:
            out = self._run(["ls", prefix.rstrip("/") + "/**"])
        except RuntimeError:
            return []  # nothing there (yet)
        return sorted(u.strip() for u in out.splitlines() if u.strip().endswith(".jsonl"))


# ----------------------------------------------------------------------------------------------------------
# Labelling one question (sync): used by pilot and for retrying invalid batch rows
# ----------------------------------------------------------------------------------------------------------
def label_unit(client, unit, model, cfg, retry=True):
    """-> (rows, raw_text, attempts). Validation failure => one retry with the error appended, else 'invalid'."""
    results, err, parsed, raw = [], None, None, None
    attempts = 2 if retry else 1
    used = 0
    for i in range(attempts):
        used = i + 1
        res = client.generate(model, unit.user_prompt(retry_error=err if i else None), cfg, len(unit.parts))
        results.append(res)
        raw = res.text
        if res.error:
            err = f"API error: {res.error}"
            continue
        if res.finish_reason not in (None, "STOP"):
            err = f"finish_reason {res.finish_reason}"
            continue
        parsed, err = validate_text(res.text, unit.labels, cfg.rationale)
        if parsed:
            break
    src = "sync_retry" if used > 1 else "sync"
    return rows_for_unit(unit, model, parsed, err, results, src, used, cfg.phash()), raw, used


# ----------------------------------------------------------------------------------------------------------
# Estimate
# ----------------------------------------------------------------------------------------------------------
_TOK_RE = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]+")


def json_tokens(text):
    """Offline token heuristic for JSON output (no tokenizer is available offline): lower-case words ~5 chars/token,
    UPPER_CASE enum words ~3 chars/token, digit groups of 3 = 1 token, punctuation clusters ~2 chars/token."""
    n = 0
    for m in _TOK_RE.finditer(text):
        t = m.group()
        if t[0].isalpha():
            n += math.ceil(len(t) / (3 if t.isupper() and len(t) > 1 else 5))
        elif t[0].isdigit():
            n += 1
        else:
            n += math.ceil(len(t) / 2)
    return n


_EVASION_WEIGHTS = {"DEFERRAL": 3, "NOT_MAINTAINED": 4, "JURISDICTION": 3, "DOES_NOT_ARISE": 1, "GENERIC": 5,
                    "DIFFERENT_QUESTION": 2, "ANNEXURE_ONLY": 1, "OTHER": 1}


def render_response(unit, cfg, rng):
    """A realistic model response for this question: FULL/PARTIAL/NONE ~ 45/25/30 %, rationale = a lead-in plus
    8-12 words quoted from the answer (total <= RATIONALE_MAX_WORDS), JSON with default separators."""
    words = (unit.answer_text or "").split()
    parts = []
    for lab, _ in unit.parts:
        label = rng.choices(["FULL", "PARTIAL", "NONE"], [45, 25, 30])[0]
        ev = "NONE_APPLICABLE" if label == "FULL" else rng.choices(list(_EVASION_WEIGHTS),
                                                                     list(_EVASION_WEIGHTS.values()))[0]
        d = {"part_label": lab, "request_type": rng.choices(CB.REQUEST_TYPES, [60, 15, 15, 10])[0]}
        if cfg.rationale:
            if words:
                k = rng.randint(8, 12)
                i = rng.randrange(0, max(1, len(words) - k))
                d["rationale"] = 'Answer says "' + " ".join(words[i:i + k]) + '"'
            else:
                d["rationale"] = "Answer text is missing."
            d["rationale"] = clip_words(d["rationale"], CB.RATIONALE_MAX_WORDS)
        d["label"], d["evasion_type"] = label, ev
        parts.append(d)
    return json.dumps({"parts": parts})


def measure_output_tokens(store, cfg, n=200, seed=1):
    """-> {"per_part", "overhead", "tokens_per_response", "n_questions", "n_parts"} on n random real questions."""
    rng = random.Random(seed)
    qs = rng.sample(store.qids, min(n, len(store.qids)))
    overhead = json_tokens(json.dumps({"parts": []}))
    tot = n_parts = 0
    for q in qs:
        u = store.unit(q)
        tot += json_tokens(render_response(u, cfg, rng))
        n_parts += len(u.parts)
    return {"per_part": (tot - overhead * len(qs)) / n_parts, "overhead": overhead,
            "tokens_per_response": tot / len(qs), "n_questions": len(qs), "n_parts": n_parts}


def out_tokens(cfg, n_parts, measured=None):
    m = (measured or MEASURED_OUT_TOKENS)[cfg.rationale]
    return m["overhead"] + m["per_part"] * n_parts


def question_profile(store):
    """[(user_prompt_tokens, n_parts)] per question: computed once, reused for every model/variant."""
    out = []
    for q in store.qids:
        u = store.unit(q)
        out.append((est_tokens(u.user_prompt()), len(u.parts)))
    return out


def estimate_profile(profile, model, cfg, on=None, price_override=None, measured=None):
    sys_tok = est_tokens(cfg.system_prompt())
    in_tok = sum(t for t, _ in profile) + sys_tok * len(profile)
    parts = sum(p for _, p in profile)
    out_tok = sum(out_tokens(cfg, p, measured) for _, p in profile)
    return {"model": model, "rationale": cfg.rationale, "requests": len(profile), "parts": parts,
            "system_tokens": sys_tok, "input_tokens": in_tok, "output_tokens": out_tok,
            "batch_usd": cost_usd(model, in_tok, out_tok, "batch", on, price_override),
            "sync_usd": cost_usd(model, in_tok, out_tok, "sync", on, price_override)}


def estimate_store(store, model, cfg, on=None, price_override=None, measured=None):
    return estimate_profile(question_profile(store), model, cfg, on, price_override, measured)


# ----------------------------------------------------------------------------------------------------------
# Batch: planning, state, build, submit, poll, collect
# ----------------------------------------------------------------------------------------------------------
def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class State:
    """data/measures/m2_batch/<model>__<prompt_hash>_<cfg_hash>/state.json. Chunk status machine:
    planned -> built -> uploaded -> submitted -> done.  A done chunk lists `missing` qids (no usable output line);
    those are re-planned into new chunks on the next run. A chunk that is submitted is never submitted again."""

    def __init__(self, path, data):
        self.path, self.data = path, data

    @classmethod
    def load(cls, work_dir, model, cfg):
        ph = cfg.phash()
        path = os.path.join(work_dir, "state.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d["model"] != model or d["prompt_hash"] != ph:
                raise RuntimeError(f"state at {path} belongs to another model/prompt")
        else:
            d = {"model": model, "prompt_hash": ph, "codebook_version": CB.CODEBOOK_VERSION,
                 "gen_cfg": cfg.fingerprint(), "created": now_iso(), "chunks": {}}
        return cls(path, d)

    def save(self):
        self.data["updated"] = now_iso()
        atomic_write_json(self.path, self.data)

    @property
    def chunks(self):
        return self.data["chunks"]

    def covered(self):
        """qids handled by some chunk, excluding those a finished chunk reported as missing."""
        wd = os.path.dirname(self.path)
        cov = set()
        for cid, c in self.chunks.items():
            cov |= set(load_manifest(wd, cid)["qids"]) - set(c.get("missing") or [])
        return cov

    def next_chunk_id(self):
        return f"{len(self.chunks) + 1:04d}"


def manifest_path(work_dir, cid):
    return os.path.join(work_dir, f"chunk_{cid}.manifest.json")


def load_manifest(work_dir, cid):
    with open(manifest_path(work_dir, cid), encoding="utf-8") as f:
        return json.load(f)


def plan_chunks(store, qids, model, cfg, chunk_requests, max_bytes, on=None, price_override=None):
    """Split qids into chunks respecting request-count and byte limits. Returns list of dicts
    {qids, keys, bytes, in_tokens, out_tokens, est_usd}. Builds every prompt once (no files written)."""
    chunk_requests = min(chunk_requests, VERTEX_MAX_REQUESTS_PER_JOB)
    max_bytes = min(max_bytes, VERTEX_MAX_FILE_BYTES)
    sys_tok = est_tokens(cfg.system_prompt())
    chunks, cur = [], None

    def fresh():
        return {"qids": [], "keys": [], "bytes": 0, "in_tokens": 0.0, "out_tokens": 0.0}

    for q in qids:
        u = store.unit(q)
        text = u.user_prompt()
        line = json.dumps(request_body(model, text, len(u.parts), cfg), ensure_ascii=False)
        nb = len(line.encode("utf-8")) + 1
        if cur is None or len(cur["qids"]) >= chunk_requests or (cur["bytes"] + nb > max_bytes and cur["qids"]):
            cur = fresh()
            chunks.append(cur)
        cur["qids"].append(q)
        cur["keys"].append(request_key(text))
        cur["bytes"] += nb
        cur["in_tokens"] += sys_tok + est_tokens(text)
        cur["out_tokens"] += out_tokens(cfg, len(u.parts))
    for c in chunks:
        c["est_usd"] = cost_usd(model, c["in_tokens"], c["out_tokens"], "batch", on, price_override)
    return chunks


def check_cost_cap(chunks, max_usd):
    total = sum(c["est_usd"] for c in chunks)
    if total > max_usd:
        raise CostCapError(f"estimated batch cost ${total:,.2f} exceeds --max-usd {max_usd:,.2f}; "
                           f"raise the cap deliberately or select fewer rows")
    return total


def build_chunk_file(store, cid, manifest, model, cfg, work_dir):
    path = os.path.join(work_dir, f"chunk_{cid}.jsonl")
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for q in manifest["qids"]:
            u = store.unit(q)
            f.write(json.dumps(request_body(model, u.user_prompt(), len(u.parts), cfg), ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def _line_key(line):
    try:
        parts = line["request"]["contents"][0]["parts"]
        return request_key("".join(p.get("text", "") for p in parts))
    except Exception:  # noqa: BLE001
        return None


def parse_output_line(line):
    """One predictions.jsonl row -> (key, GenResult)."""
    key = _line_key(line)
    st = line.get("status")
    resp = line.get("response") or {}
    if (st and str(st).strip()) or not resp.get("candidates"):
        return key, GenResult(error=str(st)[:300] if st else "empty response")
    cand = resp["candidates"][0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = resp.get("usageMetadata") or {}
    return key, GenResult(text=text, input_tokens=u.get("promptTokenCount", 0) or 0,
                          output_tokens=(u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0),
                          model_version=resp.get("modelVersion"), finish_reason=cand.get("finishReason"),
                          cached_tokens=u.get("cachedContentTokenCount", 0) or 0)


def parse_chunk_outputs(store, manifest, out_files, model, retry_fn=None, max_sync_retries=0, cfg=None):
    """Match output lines to the manifest by request-text key; validate; (optionally) retry invalid ones
    synchronously. Returns (DataFrame of label rows, list of missing qids, stats)."""
    cfg = cfg or GenCfg()
    phash, need_rat = cfg.phash(), cfg.rationale
    by_key = {}
    for q, k in zip(manifest["qids"], manifest["keys"]):
        by_key.setdefault(k, []).append(q)
    got = {}  # qid -> GenResult
    unmatched = 0
    for fp in out_files:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                key, res = parse_output_line(json.loads(ln))
                qs = by_key.get(key)
                if not qs:
                    unmatched += 1
                    continue
                for q in qs:  # identical prompts (rare) get the identical result
                    got.setdefault(q, res)
    rows, missing, todo = [], [], []
    for q in manifest["qids"]:
        res = got.get(q)
        if res is None:
            missing.append(q)
            continue
        u = store.unit(q)
        parsed, err = (None, f"API error: {res.error}") if res.error else (None, None)
        if not res.error:
            if res.finish_reason not in (None, "STOP"):
                err = f"finish_reason {res.finish_reason}"
            else:
                parsed, err = validate_text(res.text, u.labels, need_rat)
        if parsed:
            rows += rows_for_unit(u, model, parsed, None, [res], "batch", 1, phash)
        else:
            todo.append((u, err, res))
    retry, rest = (todo[:max_sync_retries], todo[max_sync_retries:]) if retry_fn is not None else ([], todo)
    if retry:  # invalid batch rows: one synchronous retry each, in parallel
        with ThreadPoolExecutor(max_workers=RETRY_WORKERS) as ex:
            for r2, _raw, _att in ex.map(lambda t: retry_fn(*t), retry):
                rows += r2
    for u, err, res in rest:
        rows += rows_for_unit(u, model, None, err, [res], "batch", 1, phash)
    df = to_frame(rows)
    n_invalid = int(df.loc[df["status"] != "ok", "qid"].nunique()) if len(df) else 0
    return df, missing, {"unmatched_lines": unmatched, "retried": len(retry), "invalid": n_invalid}


def consolidate_labels(work_dir, out_path=LABELS_PARQUET):
    """Merge all chunk label files of this run into m2_labels.parquet (idempotent; key = part_id, model_id,
    prompt_hash; newest wins; rows from other models/prompt versions are kept)."""
    files = sorted(f for f in os.listdir(work_dir) if f.endswith(".labels.parquet"))
    if not files:
        return None
    new = pd.concat([pd.read_parquet(os.path.join(work_dir, f)) for f in files], ignore_index=True)
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(["part_id", "model_id", "prompt_hash"], keep="last")
    # codebook: FULL has no evasion type (validate_text applies this too; chunk files written before it did not)
    new.loc[new["label"] == "FULL", "evasion_type"] = "NONE_APPLICABLE"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new.to_parquet(out_path + ".part", index=False)
    os.replace(out_path + ".part", out_path)
    return new


def run_dir(model, cfg, base=BATCH_DIR):
    return os.path.join(base, f"{slug(model)}__{cfg.phash()[:8]}_{cfg_hash(cfg)}")


def run_batch(store, model, cfg, *, work_dir, gcs, vertex, bucket, gcs_prefix="lok_sabha_m2",
              chunk_requests=DEFAULT_CHUNK_REQUESTS, max_bytes=DEFAULT_CHUNK_MB * 1_000_000, max_usd=150.0,
              max_concurrent=4, poll_seconds=60, no_wait=False, build_only=False, retry_fn=None,
              max_sync_retries=2000, labels_path=LABELS_PARQUET, price_override=None, on=None,
              sleep=time.sleep, log=print):
    """Resumable batch run. Safe to interrupt and re-run: finished chunks are never rebuilt or re-submitted."""
    os.makedirs(work_dir, exist_ok=True)
    state = State.load(work_dir, model, cfg)

    # 1) which qids still need labels?  (a qid is covered once it sits in a chunk that did not report it missing)
    covered = state.covered()
    remaining = [q for q in store.qids if q not in covered]
    log(f"[batch] {len(store.qids)} selected questions, {len(remaining)} not yet covered")

    # 2) plan new chunks (cost-capped BEFORE anything is built or uploaded)
    if remaining:
        plan = plan_chunks(store, remaining, model, cfg, chunk_requests, max_bytes, on, price_override)
        total = check_cost_cap(plan, max_usd)
        log(f"[batch] plan: {len(plan)} chunk(s), est. ${total:,.2f} (cap ${max_usd:,.2f})")
        for c in plan:
            cid = state.next_chunk_id()
            atomic_write_json(manifest_path(work_dir, cid), {"chunk": cid, "model": model, "qids": c["qids"],
                                                              "keys": c["keys"]})
            state.chunks[cid] = {"status": "planned", "n_requests": len(c["qids"]), "bytes": c["bytes"],
                                 "est_usd": round(c["est_usd"], 4), "job_name": None, "missing": [],
                                 "created": now_iso()}
            state.save()

    def stage_in(cid):
        return f"gs://{bucket}/{gcs_prefix}/{slug(os.path.basename(work_dir))}/in/chunk_{cid}.jsonl"

    def stage_out(cid):
        return f"gs://{bucket}/{gcs_prefix}/{slug(os.path.basename(work_dir))}/out/chunk_{cid}"

    def collect(cid):
        c = state.chunks[cid]
        files = []
        for i, uri in enumerate(gcs.list_jsonl(stage_out(cid))):
            lp = os.path.join(work_dir, f"chunk_{cid}.out{i}.jsonl")
            gcs.download(uri, lp)
            files.append(lp)
        man = load_manifest(work_dir, cid)
        df, missing, stats = parse_chunk_outputs(store, man, files, model, retry_fn, max_sync_retries, cfg)
        if len(df):
            df.to_parquet(os.path.join(work_dir, f"chunk_{cid}.labels.parquet"), index=False)
        c.update({"status": "done", "missing": missing, "stats": stats, "finished": now_iso(),
                  "n_labelled_questions": int(df["qid"].nunique()) if len(df) else 0})
        state.save()
        consolidate_labels(work_dir, labels_path)
        log(f"[batch] chunk {cid} collected: {c['n_labelled_questions']} questions, {len(missing)} missing, {stats}")

    # 3) drive chunks through the state machine
    while True:
        running = [cid for cid, c in state.chunks.items() if c["status"] == "submitted"]
        for cid in ([] if build_only else running):  # poll
            info = vertex.get(state.chunks[cid]["job_name"])
            state.chunks[cid]["job_state"] = info["state"]
            if info["state"] in VertexBatchClient.TERMINAL:
                state.chunks[cid]["job_error"] = info.get("error")
                log(f"[batch] chunk {cid} job {info['state']}")
                collect(cid)
        state.save()
        running = [cid for cid, c in state.chunks.items() if c["status"] == "submitted"]
        pending = [cid for cid, c in state.chunks.items() if c["status"] in ("planned", "built", "uploaded")]
        progressed = False
        for cid in pending:
            if len(running) >= max_concurrent and not build_only:
                break
            c = state.chunks[cid]
            if c["status"] == "planned":
                build_chunk_file(store, cid, load_manifest(work_dir, cid), model, cfg, work_dir)
                c["status"] = "built"
                state.save()
                log(f"[batch] chunk {cid} built ({c['n_requests']} requests, {c['bytes'] / 1e6:.0f} MB)")
            if build_only:
                continue
            if c["status"] == "built":
                gcs.upload(os.path.join(work_dir, f"chunk_{cid}.jsonl"), stage_in(cid))
                c.update({"status": "uploaded", "gcs_input": stage_in(cid), "gcs_output": stage_out(cid)})
                state.save()
            if c["status"] == "uploaded":
                c["job_name"] = vertex.create(model, c["gcs_input"], c["gcs_output"],
                                              f"m2-{slug(model)}-{cid}-{cfg.phash()[:8]}")
                c.update({"status": "submitted", "submitted": now_iso()})
                state.save()  # persisted immediately: never re-submit
                log(f"[batch] chunk {cid} submitted: {c['job_name']}")
                running.append(cid)
                progressed = True
        if build_only:
            break
        running = [cid for cid, c in state.chunks.items() if c["status"] == "submitted"]
        pending = [cid for cid, c in state.chunks.items() if c["status"] in ("planned", "built", "uploaded")]
        if not running and not pending:
            break
        if no_wait:
            break
        if not progressed:
            sleep(poll_seconds)
    summary = {cid: c["status"] for cid, c in state.chunks.items()}
    log(f"[batch] chunk status: {summary}")
    return state


# ----------------------------------------------------------------------------------------------------------
# Pilot and OCR
# ----------------------------------------------------------------------------------------------------------
def run_pilot(store, models, cfg, client, workers=4, out_dir=MEASURES, log=print):
    outs = {}
    for model in models:
        def one(q, model=model):
            return label_unit(client, store.unit(q), model, cfg)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, store.qids))
        rows = []
        for q, (r, raw, att) in zip(store.qids, res):
            for row in r:
                row["raw_response"] = raw
            rows += r
        df = to_frame(rows).join(pd.DataFrame(rows)[["raw_response"]]) if rows else to_frame(rows)
        path = os.path.join(out_dir, f"m2_pilot_{slug(model)}.parquet")
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(path, index=False)
        log(f"[pilot] {model}: {len(store.qids)} questions, {len(df)} parts, "
            f"{(df['status'] != 'ok').sum()} invalid -> {path}")
        outs[model] = path
    return outs


OCR_PROMPT = ("Transcribe all the text in this document (a Lok Sabha question and answer, possibly a scanned image) "
              "exactly as written, in reading order. Keep question numbers, part labels such as (a), (b), and table "
              "contents as plain text. Skip Hindi text if the document repeats it in English. Output only the "
              "transcription, with no commentary.")


def ocr_targets(sel, fallback_dir=FALLBACK_DIR):
    """qids flagged needs_ocr in questions_analysis (column may not exist) or given by --qids-file."""
    qids = None
    if getattr(sel, "qids_file", None):
        qids = read_id_list(sel.qids_file)
    elif os.path.exists(QUESTIONS_PARQUET):
        import pyarrow.parquet as pq
        if "needs_ocr" in pq.ParquetFile(QUESTIONS_PARQUET).schema_arrow.names:
            d = pd.read_parquet(QUESTIONS_PARQUET, columns=["qid", "needs_ocr"])
            qids = d.loc[d["needs_ocr"].fillna(False).astype(bool), "qid"].tolist()
    if qids is None:
        return None
    out = []
    for q in sorted(qids):
        m = re.match(r"ls(\d+)-", q)
        cands = [os.path.join(d, f"ls{m.group(1)}", f"{q}.pdf") for d in (fallback_dir, DOCX_PDF_DIR)] if m else []
        p = next((c for c in cands if os.path.exists(c)), None)
        if p:
            out.append((q, p))
    return out[:sel.limit] if getattr(sel, "limit", None) else out


def run_ocr(targets, model, client, out_path=OCR_PARQUET, workers=4, max_pdf_mb=18, log=print, save_every=25):
    have = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame()
    done = set(have["qid"]) if len(have) else set()
    todo = [(q, p) for q, p in targets if q not in done]
    rows = have.to_dict("records") if len(have) else []
    log(f"[ocr] {len(todo)} PDFs to transcribe ({len(done)} already done)")

    def one(item):
        q, p = item
        if os.path.getsize(p) > max_pdf_mb * 1e6:
            return {"qid": q, "text": "", "model_id": model, "input_tokens": 0, "output_tokens": 0,
                    "status": "too_large"}
        with open(p, "rb") as f:
            res = client.generate_pdf(model, OCR_PROMPT, f.read())
        ok = res.text and not res.error and res.finish_reason in (None, "STOP")
        return {"qid": q, "text": (res.text or "").strip(), "model_id": model, "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens, "model_version": res.model_version,
                "status": "ok" if ok else f"error:{res.error or res.finish_reason}"}

    def save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path + ".part", index=False)
        os.replace(out_path + ".part", out_path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            rows.append(r)
            if i % save_every == 0:
                save()
    if rows:
        save()
    return rows


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def add_common(p):
    p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP project id")
    p.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                   help="Vertex location (default global; use e.g. us-central1 if the model/bucket need it)")
    p.add_argument("--thinking", default="auto", help="auto|off|budget:N|level:MINIMAL|LOW|MEDIUM|HIGH")
    p.add_argument("--temperature", type=float, default=None,
                   help="default auto: 0 for Gemini 2.x, 1.0 for Gemini 3+ (Google advises against <1.0 there)")
    p.add_argument("--gen-seed", type=int, default=0, dest="gen_seed", help="generation seed (reproducibility)")
    p.add_argument("--no-rationale", action="store_true", help="cheaper variant: no rationale field in the output")
    p.add_argument("--strip-property-ordering", action="store_true",
                   help="drop propertyOrdering from the response schema if Vertex rejects it")


def add_select(p):
    p.add_argument("--qids-file", help=".txt/.csv/.parquet with qid or part_id")
    p.add_argument("--ls", type=int, nargs="*", help="restrict to these Lok Sabha numbers, e.g. 18")
    p.add_argument("--sample", type=int, help="random sample of N questions")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, help="first N questions after other filters")
    p.add_argument("--skip-no-answer", action="store_true", help="drop questions with no answer/annexure text")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="offline token + cost estimate (no API calls)")
    add_select(e)
    e.add_argument("--models", default=",".join(list(BATCH_PRICES)))
    e.add_argument("--price-date", help="YYYY-MM-DD for date-dependent prices (default today)")
    e.add_argument("--measure", action="store_true",
                   help="re-measure output tokens per part on 200 real questions instead of the stored constants")
    e.add_argument("--out-tokens-per-part", type=float, help="override measured output tokens per part (both variants)")

    pl = sub.add_parser("pilot", help="synchronous labelling of a list of qids with several models")
    add_common(pl)
    add_select(pl)
    pl.add_argument("--models", required=True, help="comma-separated model ids")
    pl.add_argument("--workers", type=int, default=4)

    b = sub.add_parser("batch", help="Vertex AI batch prediction: build/submit/poll/collect (resumable)")
    add_common(b)
    add_select(b)
    b.add_argument("--model", required=True)
    b.add_argument("--gcs-bucket", help="bucket name (no gs://); required unless --build-only/--dry-run")
    b.add_argument("--gcs-prefix", default="lok_sabha_m2")
    b.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_REQUESTS,
                   help=f"requests per job (Vertex max {VERTEX_MAX_REQUESTS_PER_JOB:,})")
    b.add_argument("--max-chunk-mb", type=int, default=DEFAULT_CHUNK_MB, help="input file size cap (Vertex max 1 GB)")
    b.add_argument("--max-usd", type=float, default=150.0)
    b.add_argument("--max-concurrent-jobs", type=int, default=4)
    b.add_argument("--poll-seconds", type=int, default=60)
    b.add_argument("--no-wait", action="store_true", help="submit then exit; re-run later to poll/collect")
    b.add_argument("--build-only", action="store_true", help="plan and write JSONL locally; no upload/submit")
    b.add_argument("--dry-run", action="store_true", help="plan + cost check only; write nothing")
    b.add_argument("--max-sync-retries", type=int, default=2000,
                   help="invalid batch rows retried once synchronously (costs real-time price)")
    b.add_argument("--price-in", type=float)
    b.add_argument("--price-out", type=float)

    o = sub.add_parser("ocr", help="transcribe needs_ocr PDFs with Gemini -> data/measures/ocr_text.parquet")
    add_common(o)
    o.add_argument("--qids-file")
    o.add_argument("--limit", type=int)
    o.add_argument("--model", default="gemini-3.1-flash-lite")
    o.add_argument("--workers", type=int, default=4)
    return ap


def _cfg(a):
    return GenCfg(temperature=a.temperature, thinking=a.thinking, rationale=not a.no_rationale,
                  seed=a.gen_seed, strip_property_ordering=a.strip_property_ordering)


def _need_project(a):
    if not a.project:
        sys.exit("--project (or GOOGLE_CLOUD_PROJECT) is required. Run `gcloud auth application-default login` first.")


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "estimate":
        store = Store.load(a, use_ocr=False)
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        measured = {k: dict(v) for k, v in MEASURED_OUT_TOKENS.items()}
        if a.measure:
            for rat in (True, False):
                m = measure_output_tokens(store, GenCfg(rationale=rat))
                measured[rat] = {"per_part": m["per_part"], "overhead": m["overhead"]}
                print(f"measured output ({'with' if rat else 'no'} rationale): {m['per_part']:.1f} tokens/part + "
                      f"{m['overhead']} overhead ({m['tokens_per_response']:.1f} tokens/response, "
                      f"{m['n_questions']} questions, {m['n_parts']} parts)")
        if a.out_tokens_per_part is not None:
            for rat in (True, False):
                measured[rat] = {"per_part": a.out_tokens_per_part, "overhead": measured[rat]["overhead"]}
        print(f"{len(store.qids):,} questions selected (prices: {PRICES_SOURCE}, as of {PRICES_AS_OF}; "
              f"CONFIRM Vertex prices in the console).\n"
              f"prompt_hash with rationale {MP.PROMPT_HASH}, "
              f"without {MP.PROMPT_HASH_NR}; output tokens/part used: with {measured[True]['per_part']:.1f}, "
              f"without {measured[False]['per_part']:.1f}")
        profile = question_profile(store)
        print(f"parts {sum(p for _, p in profile):,}; user-prompt tokens {sum(t for t, _ in profile) / 1e6:,.1f}M; "
              f"system prompt {est_tokens(MP.SYSTEM_PROMPT):.0f} tokens (with rationale) / "
              f"{est_tokens(MP.SYSTEM_PROMPT_NR):.0f} (without); input tokens counted per request, no caching")
        print(f"{'model':24s} {'variant':14s} {'in M':>7s} {'out M':>7s} {'batch USD':>10s} {'real-time USD':>14s}")
        for m in models:
            for rat in (True, False):
                e = estimate_profile(profile, m, GenCfg(rationale=rat), a.price_date, None, measured)
                print(f"{m:24s} {'rationale' if rat else 'no-rationale':14s} {e['input_tokens'] / 1e6:7.1f} "
                      f"{e['output_tokens'] / 1e6:7.1f} {e['batch_usd']:10,.2f} {e['sync_usd']:14,.2f}")
        for m in models:
            if len(BATCH_PRICES.get(m, [])) > 1:
                later = BATCH_PRICES[m][-1][0]
                for rat in (True, False):
                    e = estimate_profile(profile, m, GenCfg(rationale=rat), later, None, measured)
                    print(f"{m} from {later} ({'rationale' if rat else 'no-rationale'}): batch ${e['batch_usd']:,.2f}")
        return 0

    if a.cmd == "pilot":
        _need_project(a)
        cfg = _cfg(a)
        store = Store.load(a)
        client = SyncClient(a.project, a.location)
        run_pilot(store, [m.strip() for m in a.models.split(",") if m.strip()], cfg, client, a.workers)
        return 0

    if a.cmd == "batch":
        cfg = _cfg(a)
        store = Store.load(a)
        override = (a.price_in, a.price_out) if a.price_in is not None and a.price_out is not None else None
        wd = run_dir(a.model, cfg)
        if a.dry_run or a.build_only:
            plan = plan_chunks(store, store.qids, a.model, cfg, a.chunk_size, a.max_chunk_mb * 1_000_000,
                               None, override)
            total = check_cost_cap(plan, a.max_usd)
            print(f"{len(store.qids)} questions -> {len(plan)} chunk(s), est ${total:,.2f} (cap ${a.max_usd:,.2f})")
            if a.dry_run:
                return 0
            run_batch(store, a.model, cfg, work_dir=wd, gcs=None, vertex=None, bucket=None,
                      chunk_requests=a.chunk_size, max_bytes=a.max_chunk_mb * 1_000_000, max_usd=a.max_usd,
                      build_only=True, price_override=override)
            return 0
        _need_project(a)
        if not a.gcs_bucket:
            sys.exit("--gcs-bucket is required")
        sync = SyncClient(a.project, a.location)

        def retry_fn(unit, err, res):
            r = sync.generate(a.model, unit.user_prompt(retry_error=err), cfg, len(unit.parts))
            parsed, e2 = (None, f"API error: {r.error}") if r.error else validate_text(r.text, unit.labels,
                                                                                         cfg.rationale)
            return rows_for_unit(unit, a.model, parsed, e2, [res, r], "sync_retry", 2, cfg.phash()), r.text, 2

        run_batch(store, a.model, cfg, work_dir=wd, gcs=GcsCli(a.project), vertex=VertexBatchClient(a.project, a.location),
                  bucket=a.gcs_bucket, gcs_prefix=a.gcs_prefix, chunk_requests=a.chunk_size,
                  max_bytes=a.max_chunk_mb * 1_000_000, max_usd=a.max_usd, max_concurrent=a.max_concurrent_jobs,
                  poll_seconds=a.poll_seconds, no_wait=a.no_wait, retry_fn=retry_fn,
                  max_sync_retries=a.max_sync_retries, price_override=override)
        return 0

    if a.cmd == "ocr":
        _need_project(a)
        targets = ocr_targets(a)
        if targets is None:
            sys.exit("questions_analysis.parquet has no needs_ocr column yet; pass --qids-file with the qids to OCR")
        run_ocr(targets, a.model, SyncClient(a.project, a.location), workers=a.workers)
        return 0


def cli():
    try:
        return main()
    except CostCapError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
