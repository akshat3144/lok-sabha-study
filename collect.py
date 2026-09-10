"""Resumable, polite collector for Lok Sabha questions & answers (sansad.in).

Two independent jobs, each safe to stop (Ctrl+C or `python collect.py stop`) and resume (run again):

  python collect.py api    # question + answer text via the official sansad.in API, 13th-18th Lok Sabha
  python collect.py docs   # Word (.docx, else PDF) answer files for records whose answer text is empty (18th LS)
  python collect.py status # print progress of both jobs
  python collect.py stop   # ask both running jobs to finish their current request and exit cleanly

Every page / file is written atomically (temp file + rename), so stopping never leaves partial data;
on restart, anything already on disk is skipped.

The text is copyright of the Lok Sabha Secretariat: collect for non-commercial research only and do
not redistribute the raw files.
"""
import gzip, json, os, random, re, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
API_DIR = os.path.join(DATA, "raw", "api")
DOC_DIR = os.path.join(DATA, "raw", "docs")
LOG_DIR = os.path.join(DATA, "logs")
STOP_FILE = os.path.join(DATA, "STOP")

BASE = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"
SESSIONS_URL = "https://sansad.in/api_ls/business/getAllLoksabhaAndSession"
LOK_SABHAS = [18, 17, 16, 15, 14, 13]  # 18 first so the docs job can start early
PAGE_SIZE = 100
WORKERS = 2
BASE_DELAY = 1.0  # seconds each worker waits between requests
NET_WAIT_MAX = 3600  # wait out internet outages for up to an hour per request before giving up
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; polite rate-limited collector)"}

stop_event = threading.Event()
lock = threading.RLock()  # re-entrant: progress logging happens while the stats lock is held
for d in (API_DIR, DOC_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# ---------- helpers ----------
def log(job, msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{job}] {msg}"
    print(line, flush=True)
    with lock, open(os.path.join(LOG_DIR, f"{job}.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write(path, data: bytes):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_status(job, status):
    status["updated"] = datetime.now().isoformat(timespec="seconds")
    atomic_write(os.path.join(DATA, f"status_{job}.json"), json.dumps(status, indent=2).encode())


def should_stop():
    return stop_event.is_set() or os.path.exists(STOP_FILE)


def polite_sleep(sec):
    end = time.time() + sec
    while time.time() < end and not should_stop():
        time.sleep(0.25)


def fetch(job, url, params=None, timeout=120, max_tries=8):
    """GET with exponential backoff. Returns Response or None (after max_tries or on stop).
    Only server answers (HTTP errors) count toward max_tries. Network failures (our connection
    dropping, timeouts) are waited out for up to NET_WAIT_MAX seconds without using up tries, so
    an internet outage never gets mistaken for a broken record."""
    attempt, net_waited, net_fail = 0, 0.0, 0
    while attempt < max_tries:
        if should_stop():
            return None
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            net_fail = 0
            if r.status_code in (200, 404):
                return r
            attempt += 1
            log(job, f"HTTP {r.status_code} on {url} {params or ''} (attempt {attempt})")
            polite_sleep(min(300, 5 * 2 ** (attempt - 1)) + random.uniform(0, 3))
        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema) as e:
            log(job, f"ERROR {type(e).__name__} (bad link, skipped) on {url!r}")
            return None
        except requests.RequestException as e:
            net_fail += 1
            wait = min(120, 5 * 2 ** min(net_fail - 1, 5)) + random.uniform(0, 3)
            if net_waited + wait > NET_WAIT_MAX:
                log(job, f"giving up after {int(net_waited)} s of network errors on {url} {params or ''}")
                return None
            log(job, f"ERROR {type(e).__name__} (network) on {url} {params or ''}; waiting {int(wait)} s")
            polite_sleep(wait)
            net_waited += wait
    return None


def on_sigint(*_):
    if not stop_event.is_set():
        print("\nStopping after current requests finish... (data on disk is safe)", flush=True)
    stop_event.set()


# ---------- job 1: API pages ----------
def page_path(ls, sess, page):
    return os.path.join(API_DIR, f"ls{ls}", f"s{sess:02d}_p{page:04d}.json.gz")


def build_plan():
    plan_file = os.path.join(API_DIR, "plan.json")
    if os.path.exists(plan_file):
        return json.load(open(plan_file, encoding="utf-8"))
    r = fetch("api", SESSIONS_URL)
    if r is None:
        sys.exit("Could not fetch session list; try again later.")
    sessions = {x["loksabha"]: [s["sessionNo"] for s in x["sessions"]] for x in r.json()}
    plan = []
    for ls in LOK_SABHAS:
        expected = None
        r = fetch("api", BASE, {"loksabhaNo": ls, "locale": "en", "pageNo": 1, "pageSize": 1})
        if r is not None:
            expected = r.json()[0]["totalRecordSize"]
        got = 0
        for s in sessions.get(ls, []):
            if should_stop():
                sys.exit("Stopped while planning; nothing lost. Run again to continue.")
            r = fetch("api", BASE, {"loksabhaNo": ls, "sessionNumber": s, "locale": "en", "pageNo": 1, "pageSize": 1})
            n = r.json()[0]["totalRecordSize"] if r is not None else 0
            got += n
            if n:
                plan.append({"ls": ls, "session": s, "records": n, "pages": -(-n // PAGE_SIZE)})
            time.sleep(BASE_DELAY)
        log("api", f"plan LS{ls}: {got} records across sessions (whole-LS total reported: {expected})")
    atomic_write(plan_file, json.dumps(plan, indent=1).encode())
    return plan


def run_api():
    plan = build_plan()
    tasks = [(p["ls"], p["session"], pg) for p in plan for pg in range(1, p["pages"] + 1)]
    for ls in LOK_SABHAS:
        os.makedirs(os.path.join(API_DIR, f"ls{ls}"), exist_ok=True)
    todo = [t for t in tasks if not os.path.exists(page_path(*t))]
    total, start_done = len(tasks), len(tasks) - len(todo)
    stats = {"job": "api", "pages_total": total, "pages_done": start_done, "records_saved": 0,
             "failed": [], "state": "running", "started": datetime.now().isoformat(timespec="seconds")}
    log("api", f"{total} pages planned, {start_done} already on disk, {len(todo)} to fetch")
    t0, fetched = time.time(), [0]

    def get_list(ls, sess, page_no, size, tries):
        r = fetch("api", BASE, {"loksabhaNo": ls, "sessionNumber": sess, "locale": "en",
                                "pageNo": page_no, "pageSize": size}, max_tries=tries)
        if r is None or r.status_code != 200:
            return None, None
        try:
            return r.json()[0]["listOfQuestions"], r.content
        except Exception:
            return None, None

    CHILDREN = {100: (20, 5), 20: (10, 2), 10: (5, 2), 5: (1, 5)}  # size -> (child size, how many)

    def fetch_block(ls, sess, start, size, qs, bad):
        """Fetch records [start, start+size) (1-based, aligned to size); on failure narrow down
        20 -> 10 -> 5 -> 1 so only the broken record(s) are lost. Returns False if stopped."""
        if should_stop():
            return False
        sub, _ = get_list(ls, sess, (start - 1) // size + 1, size, 2 if size == 1 else 1)
        if sub is not None:
            qs += sub
            return True
        if size == 1:
            bad.append(start)
            return True
        child, n = CHILDREN[size]
        return all(fetch_block(ls, sess, start + i * child, child, qs, bad) for i in range(n))

    def split_fetch(ls, sess, pg):
        """Some pages return HTTP 500 at size 100 because one bad record breaks the response.
        Re-fetch the same 100 records in shrinking blocks so only the broken record is lost."""
        qs, bad = [], []
        start = (pg - 1) * PAGE_SIZE + 1
        for i in range(5):
            if not fetch_block(ls, sess, start + i * 20, 20, qs, bad):
                return None, bad
        return qs, bad

    def work(t):
        if should_stop():
            return
        ls, sess, pg = t
        qs, content = get_list(ls, sess, pg, PAGE_SIZE, 1)  # one try: these 500s are deterministic, so split straight away
        if qs is None and not should_stop():
            log("api", f"page {t} failing at size {PAGE_SIZE}; splitting into smaller requests")
            qs, bad = split_fetch(ls, sess, pg)
            if qs is None:
                return  # stopped mid-split; page will be retried on the next run
            if bad:
                with lock, open(os.path.join(DATA, "api_bad_records.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ls": ls, "session": sess, "page": pg, "record_index": bad}) + "\n")
                log("api", f"page {t}: {len(bad)} record(s) unrecoverable even one-by-one (listed in data/api_bad_records.jsonl)")
            content = json.dumps([{"listOfQuestions": qs, "totalRecordSize": None, "_split": True,
                                   "_unrecoverable_record_index": bad}]).encode()
        if qs is None:
            return
        atomic_write(page_path(ls, sess, pg), gzip.compress(content))
        with lock:
            stats["pages_done"] += 1
            stats["records_saved"] += len(qs)
            fetched[0] += 1
            if fetched[0] % 25 == 0:
                rate = fetched[0] / (time.time() - t0)
                left = stats["pages_total"] - stats["pages_done"]
                stats["eta_hours"] = round(left / rate / 3600, 2) if rate else None
                log("api", f"progress {stats['pages_done']}/{total} pages ({100 * stats['pages_done'] / total:.1f}%), "
                           f"ETA {stats['eta_hours']} h, failures {len(stats['failed'])}")
                write_status("api", stats)
        polite_sleep(BASE_DELAY + random.uniform(0, 0.5))

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    remaining = sum(1 for t in tasks if not os.path.exists(page_path(*t)))
    stats["state"] = "stopped" if should_stop() else ("complete" if remaining == 0 else "incomplete")
    stats["pages_missing"] = remaining
    write_status("api", stats)
    log("api", f"FINISHED state={stats['state']} pages_done={stats['pages_done']}/{total} missing={remaining}"
               + ("  -> run `python collect.py api` again to retry missing pages" if remaining else ""))


# ---------- job 2: answer files ----------
def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def doc_targets():
    """Records with empty answer text but a downloadable answer file."""
    out = {}
    for ls in LOK_SABHAS:
        d = os.path.join(API_DIR, f"ls{ls}")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json.gz"):
                continue
            try:
                qs = json.loads(gzip.decompress(open(os.path.join(d, fn), "rb").read()))[0]["listOfQuestions"]
            except Exception:
                continue
            for q in qs:
                if (q.get("answerText") or "").strip():
                    continue
                docx, pdf = (str(q.get(k) or "").strip() for k in ("questionsDocPath", "questionsFilePath"))
                docx = docx if docx.startswith("http") else None  # some records carry a blank " " link
                pdf = pdf if pdf.startswith("http") else None
                if not (docx or pdf):
                    continue
                key = f"ls{ls}_s{safe(q.get('sessionNo'))}_{safe((q.get('type') or '').strip())}_{safe(q.get('quesNo'))}_{safe(q.get('date'))}"
                out[key] = (ls, docx, pdf)
    return out


def doc_done(key, ls):
    base = os.path.join(DOC_DIR, f"ls{ls}", key)
    return os.path.exists(base + ".docx") or os.path.exists(base + ".pdf")


def run_docs():
    stats = {"job": "docs", "files_done": 0, "files_known": 0, "failed": 0, "state": "running",
             "started": datetime.now().isoformat(timespec="seconds")}
    fail_log = os.path.join(DATA, "docs_failed.jsonl")
    t0, fetched = time.time(), [0]

    def work(item):
        key, (ls, docx, pdf) = item
        if should_stop() or doc_done(key, ls):
            return
        os.makedirs(os.path.join(DOC_DIR, f"ls{ls}"), exist_ok=True)
        for url, ext in ((docx, ".docx"), (pdf, ".pdf")):
            if not url:
                continue
            r = fetch("docs", url, timeout=90, max_tries=5)
            if r is not None and r.status_code == 200 and len(r.content) > 500:
                atomic_write(os.path.join(DOC_DIR, f"ls{ls}", key + ext), r.content)
                with lock:
                    stats["files_done"] += 1
                    fetched[0] += 1
                    if fetched[0] % 100 == 0:
                        rate = fetched[0] / (time.time() - t0)
                        left = stats["files_known"] - stats["files_done"]
                        stats["eta_hours"] = round(left / rate / 3600, 2) if rate else None
                        log("docs", f"progress {stats['files_done']}/{stats['files_known']} files, "
                                    f"ETA {stats['eta_hours']} h (of files known so far), failures {stats['failed']}")
                        write_status("docs", stats)
                polite_sleep(BASE_DELAY * 0.5 + random.uniform(0, 0.5))
                return
            polite_sleep(BASE_DELAY)
        if not should_stop():
            with lock:
                stats["failed"] += 1
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "docx": docx, "pdf": pdf}) + "\n")

    while not should_stop():
        targets = doc_targets()
        stats["files_known"] = len(targets)
        todo = [(k, v) for k, v in targets.items() if not doc_done(k, v[0])]
        stats["files_done"] = len(targets) - len(todo)
        write_status("docs", stats)
        log("docs", f"{len(targets)} answer files known, {stats['files_done']} on disk, {len(todo)} to fetch")
        if todo:
            with ThreadPoolExecutor(WORKERS) as ex:
                list(ex.map(work, todo))
            continue
        api_state = {}
        try:
            api_state = json.load(open(os.path.join(DATA, "status_api.json"), encoding="utf-8"))
        except Exception:
            pass
        if api_state.get("state") == "complete":
            break
        log("docs", "waiting for the api job to save more pages (checking again in 60 s)")
        polite_sleep(60)
    stats["state"] = "stopped" if should_stop() else "complete"
    write_status("docs", stats)
    log("docs", f"FINISHED state={stats['state']} files_done={stats['files_done']}/{stats['files_known']} "
                f"failed={stats['failed']} (failures listed in data/docs_failed.jsonl; re-run to retry)")


# ---------- recheck split pages ----------
def recheck_bad():
    """Re-test every record a split page marked unrecoverable. If any of them now loads (i.e. the
    earlier failure was really a network drop), delete that page so `api` re-fetches it in full."""
    pages = []
    for ls in LOK_SABHAS:
        d = os.path.join(API_DIR, f"ls{ls}")
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not fn.endswith(".json.gz"):
                continue
            p = os.path.join(d, fn)
            obj = json.loads(gzip.decompress(open(p, "rb").read()))[0]
            bad = obj.get("_unrecoverable_record_index") or []
            if bad:
                sess = int(fn[1:3])
                pages.append((p, ls, sess, bad))
    print(f"{len(pages)} split pages with {sum(len(b) for *_, b in pages)} unrecoverable records to re-test")
    deleted, genuine = [], 0

    def test(item):
        p, ls, sess, bad = item
        for idx in bad:
            if should_stop():
                return
            r = fetch("recheck", BASE, {"loksabhaNo": ls, "sessionNumber": sess, "locale": "en",
                                        "pageNo": idx, "pageSize": 1}, max_tries=1)
            if r is not None and r.status_code == 200:
                os.remove(p)
                with lock:
                    deleted.append(os.path.basename(p))
                return
            time.sleep(0.5)
        with lock:
            nonlocal_genuine[0] += len(bad)

    nonlocal_genuine = [0]
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(test, pages))
    log("recheck", f"deleted {len(deleted)} pages for re-fetch (records loaded fine now); "
                   f"{nonlocal_genuine[0]} records confirmed broken on the server")


# ---------- status / stop ----------
def show_status():
    for job in ("api", "docs"):
        p = os.path.join(DATA, f"status_{job}.json")
        print(f"--- {job} ---")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "not started")
    print("STOP flag present" if os.path.exists(STOP_FILE) else "no STOP flag")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "stop":
        open(STOP_FILE, "w").close()
        print("STOP flag written; running jobs will exit after their current request.")
    elif cmd == "status":
        show_status()
    elif cmd == "recheck":
        signal.signal(signal.SIGINT, on_sigint)
        recheck_bad()
    elif cmd in ("api", "docs"):
        if "--workers" in sys.argv:  # e.g. `python collect.py docs --workers 4`
            WORKERS = max(1, min(6, int(sys.argv[sys.argv.index("--workers") + 1])))
            print(f"using {WORKERS} workers")
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)  # starting a job clears an old stop request
        signal.signal(signal.SIGINT, on_sigint)
        (run_api if cmd == "api" else run_docs)()
    else:
        print(__doc__)
