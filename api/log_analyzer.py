from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import docker
import re
import os
import asyncio
import time
from typing import List, Optional, Dict

app = FastAPI()

# ==========================================
# CONFIGURATION
# ==========================================
SYSTEM_LOG_FILES = ["/var/log/syslog", "/var/log/auth.log", "/var/log/kern.log"]

# Containers to ALWAYS ignore (Noise reduction)
IGNORE_CONTAINERS = ["LogAI", "ollama", "qdrant"]

# Final output cap (what you return)
MAX_LINES_PER_SOURCE = int(os.getenv("MAX_LINES_PER_SOURCE", "1000"))

# Cap docker log retrieval at the source
DOCKER_LOG_TAIL_LINES = int(os.getenv("DOCKER_LOG_TAIL_LINES", "8000"))

# Safety: if processing takes too long, bail out
MAX_PROCESS_SECONDS = int(os.getenv("MAX_PROCESS_SECONDS", "120"))

# Safety: cap line length before regex filtering — \r-progress blobs
# (yt-dlp etc.) can otherwise form multi-MB "lines" that grind the
# noise regex for hours and wedge the worker (metube incident 2026-06-12)
MAX_LINE_CHARS = int(os.getenv("MAX_LINE_CHARS", "2000"))

# Per-container known-benign lines, dropped from /digest before any model sees them.
# Each pattern must match the WHOLE line (after timestamp/ANSI stripping), so a
# crash, bind error or any other text on the line still reaches the analyser.
# vibravid: the upstream entrypoint runs Django runserver on purpose (owner-accepted,
# LAN-only app); its fixed startup banner was re-escalated under new paraphrases.
CONTAINER_NOISE_LINES = {
    "vibravid": [
        r"Starting development server at http://0\.0\.0\.0:8000/",
        r"WARNING: This is a development server\. Do not use it in a production setting\. "
        r"Use a production WSGI or ASGI server instead\.",
        r"For more information on production servers see: "
        r"https://docs\.djangoproject\.com/en/\d{1,2}\.\d{1,2}/howto/deployment/",
    ],
}
_CONTAINER_NOISE_RE = {
    name: re.compile("|".join(f"(?:{p})" for p in pats))
    for name, pats in CONTAINER_NOISE_LINES.items()
}


def drop_container_noise(name: str, lines, info: dict, deadline=None):
    """Yield lines except the container's known-benign whole lines.

    Runs per line, before build_digest's per-template noise cache, so a template
    collision can never carry a drop decision over to a different line. Matches the
    whole (already source-bounded) line, never a truncated prefix. Checks the deadline
    itself, because dropped lines never reach build_digest's periodic check.
    """
    rx = _CONTAINER_NOISE_RE.get(name)
    if rx is None:
        yield from lines
        return
    info["container_noise_dropped"] = 0
    n = 0
    for line in lines:
        n += 1
        if deadline is not None and (n & 1023) == 0 and time.time() > deadline:
            info["noise_timed_out"] = True
            return
        # A line at or over the cap may be a truncated prefix (the iterator cuts EOF and
        # runaway lines to MAX_LINE_CHARS without a marker): never treat it as a whole line.
        if len(line) >= MAX_LINE_CHARS:
            yield line
            continue
        s = _ANSI.sub("", line).strip()
        m = _DOCKER_TS.match(s)
        if m:
            s = s[m.end():].strip()
        if rx.fullmatch(s):
            info["container_noise_dropped"] += 1
            continue
        yield line



# ==========================================
# SECRET REDACTION
# ==========================================
# Private keys must never leave this API (LLM prompts, n8n executions, mail).
# Handles single-line PEM (escaped \n) and multi-line PEM, including
# util.inspect-style dumps where each body line is its own log line.
_PEM_INLINE = re.compile(r"-----BEGIN ([A-Z0-9 ]*)PRIVATE KEY-----.*?(?:-----END \1PRIVATE KEY-----|$)")
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----")
_TS_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s")
PEM_MAX_BODY_LINES = 120
# Secret values in env dumps (Dockhand env rebase), query strings, JSON keys and bearer headers.
# Env names match upper-case only, so prose such as "bearer token: timeout" stays readable.
_SECRET_ENV = re.compile(r'\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|PASS|APIKEY|API_KEY|_KEY|AUTH)[A-Z0-9_]*\s*=\s*)([^"\\\s,;&\]\}]+)')
_SECRET_QUERY = re.compile(r'(?i)([?&](?:api_?key|apikey|token|access_token|secret|password|pass|passwd)=)[^&\s"\'#]+')
_SECRET_JSON = re.compile(r'(?i)(\\?"[a-z0-9_]*(?:token|secret|password|passwd|api_?key)\\?"\s*:\s*\\?")[^"\\]+')
_BEARER = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]{8,}")


def _redact_secrets(line):
    for rx in (_SECRET_ENV, _SECRET_QUERY, _SECRET_JSON, _BEARER):
        line = rx.sub(lambda m: m.group(1) + "[REDACTED]", line)
    return line


def redact_lines(lines):
    """Yield lines with private keys and secret values redacted."""
    in_key, body = False, 0
    for line in lines:
        if in_key:
            m = _TS_PREFIX.match(line)
            prefix = m.group(0) if m else ""
            body += 1
            if _PEM_END.search(line) or body > PEM_MAX_BODY_LINES:
                in_key = False
            yield prefix + "[REDACTED PRIVATE KEY]"
            continue
        if _PEM_BEGIN.search(line):
            red = _PEM_INLINE.sub("[REDACTED PRIVATE KEY]", line)
            last_begin = list(_PEM_BEGIN.finditer(line))[-1].end()
            if not _PEM_END.search(line, last_begin):
                in_key, body = True, 0
            yield _redact_secrets(red)
            continue
        yield _redact_secrets(line)


# ==========================================
# MODELS
# ==========================================
class LogRequest(BaseModel):
    containers: Optional[List[str]] = None
    hours: int = 24


# ==========================================
# LOGIC
# ==========================================
class LogFetcher:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(
                "CRITICAL: Could not connect to Docker socket. "
                "Ensure /var/run/docker.sock is mounted. "
                f"Error: {e}"
            )
            self.client = None

        # Precompile noise regex once (big speed win)
        noise_patterns = [
            r"^\s*$",
            r"GET /health.*200",
            r"HEAD /.*200",
            r".*heartbeat.*",
            r".*ping.*pong.*",
            r"CRON.*pam_unix.*session",
            r"nginx.*(GET|POST|HEAD).* 200 ",
            r"nginx.*(GET|POST|HEAD).* 304 ",
            r"\[Nest\].*Request.*200",
            r".*Processing job.*",
            r".*Face detection.*",
            r".*CLIP encoder.*",
            r".*Encoded.*images.*",
            r".*Loading model.*",
            r".*checkpoint starting.*",
            r".*checkpoint complete.*",
            r".*RssSyncService.*",
            r".*DownloadDecisionMaker.*",
            r".*TrackedDownloadService.*Processing.*",
            r".*HousekeepingService.*",
            r".*Scheduler.*",
            r".*Refresh.*Service.*",
            r".*flaresolverr.*GET /v1",
            r".*prowlarr.*api/v1/indexer",
            r".*IP geolocation database loaded.*",
            r".*Successfully parsed.*",
            r"query\[A\]",
            r"query\[AAAA\]",
            r".*INF.*Connection.*registered.*",
            r".*INF.*Quic.*connection.*",
            r".*INF.*Generated.*event.*",
            r".*IP address.*matches.*",
            r".*Checking.*IP.*",
            r".*healthMonitoring.*succeeded but took.*",
            r".*No active video sessions found.*",
            r".*tRPC request from.*",
            r".*Failed to get releases.*",
            r"\[download\]\s+\d{1,3}(?:\.\d+)?%",
        ]
        self._noise_re = re.compile("|".join(f"(?:{p})" for p in noise_patterns), re.IGNORECASE)

    def filter_noise(self, log_line: str) -> bool:
        # True = keep, False = drop
        return not bool(self._noise_re.search(log_line))

    def get_stats(self, container_name: str) -> str:
        if not self.client:
            return "Docker Error"
        try:
            container = self.client.containers.get(container_name)
            if container.status != "running":
                return "Status: Not Running"

            stats = container.stats(stream=False)

            # CPU
            cpu_percent = 0.0
            try:
                cpu_delta = (
                    stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                )
                system_cpu_delta = (
                    stats["cpu_stats"]["system_cpu_usage"]
                    - stats["precpu_stats"]["system_cpu_usage"]
                )
                number_cpus = stats["cpu_stats"].get("online_cpus") or 1
                if system_cpu_delta > 0.0:
                    cpu_percent = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
            except Exception:
                pass

            # Memory
            mem_usage = mem_limit = mem_percent = 0.0
            try:
                mem_usage = stats["memory_stats"]["usage"] / (1024 * 1024)
                mem_limit = stats["memory_stats"]["limit"] / (1024 * 1024)
                mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit else 0.0
            except Exception:
                pass

            return (
                f"LIVE STATS: CPU: {cpu_percent:.2f}% | "
                f"RAM: {mem_usage:.1f}MB / {mem_limit:.1f}MB ({mem_percent:.1f}%)"
            )
        except Exception as e:
            return f"Stats Error: {str(e)}"

    def get_logs_generic(self, name: str, is_file: bool, hours: int) -> Dict[str, str]:
        start_time = time.time()
        try:
            raw_lines: List[str] = []
            stats_info = ""

            if is_file:
                if os.path.exists(name):
                    # Read only the tail without loading entire file
                    raw_lines = self._tail_file_lines(name, MAX_LINES_PER_SOURCE * 5)
                    stats_info = "Source: System Log File"
                else:
                    return {"logs": "", "stats": "File not found"}

            else:
                if not self.client:
                    return {"logs": "", "stats": "Docker Client Missing"}

                container = self.client.containers.get(name)

                since = datetime.now() - timedelta(hours=hours)

                logs_bytes = container.logs(
                    since=since,
                    timestamps=True,
                    tail=DOCKER_LOG_TAIL_LINES,
                    stdout=True,
                    stderr=True,
                )

                logs = logs_bytes.decode("utf-8", errors="ignore")
                # \r-overwritten progress output (yt-dlp etc.) would otherwise
                # keep an entire progress session as ONE multi-MB line
                raw_lines = logs.replace("\r", "\n").split("\n")

                stats_info = self.get_stats(name)

            relevant_logs: List[str] = []
            for line in redact_lines(raw_lines):
                if time.time() - start_time > MAX_PROCESS_SECONDS:
                    stats_info = f"{stats_info} | NOTE: processing timed out after {MAX_PROCESS_SECONDS}s"
                    break

                # Never feed the noise regex an unbounded line — a single huge
                # line can grind the regex engine far past MAX_PROCESS_SECONDS
                # (the guard above only fires between lines)
                if len(line) > MAX_LINE_CHARS:
                    line = line[:MAX_LINE_CHARS] + " ...[line truncated]"

                if self.filter_noise(line):
                    s = line.strip()
                    if s:
                        relevant_logs.append(s)

            log_content = "\n".join(relevant_logs[-MAX_LINES_PER_SOURCE:])
            return {"logs": log_content, "stats": stats_info}

        except Exception as e:
            return {"logs": "", "stats": f"Error fetching source: {str(e)}"}

    @staticmethod
    def _tail_file_lines(path: str, max_lines: int) -> List[str]:
        """
        Efficient-ish tail: read from end in blocks, avoid loading full file.
        Good enough for large /var/log/*.
        """
        block_size = 8192
        data = b""
        lines: List[bytes] = []

        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()

            while pos > 0 and len(lines) <= max_lines:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
                lines = data.splitlines()

        # Convert last max_lines to str
        return [l.decode("utf-8", errors="ignore") for l in lines[-max_lines:]]


fetcher = LogFetcher()

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/list_targets")
async def list_targets():
    targets = [{"name": f, "type": "file", "pretty_name": f"SYSTEM: {f}"} for f in SYSTEM_LOG_FILES]

    if fetcher.client:
        try:
            containers = fetcher.client.containers.list(filters={"status": "running"})
            for c in containers:
                if c.name in IGNORE_CONTAINERS:
                    continue
                targets.append({"name": c.name, "type": "container", "pretty_name": f"DOCKER: {c.name}"})
        except Exception as e:
            print(f"Error listing containers: {e}")

    return {"targets": targets}


@app.post("/fetch_single")
async def fetch_single(request: LogRequest):
    if not request.containers:
        return {"status": "error", "content": "", "stats": "No target provided"}

    target = request.containers[0]
    is_file = target.startswith("/")

    # Offload blocking docker/file IO + parsing to a worker thread
    result = await asyncio.to_thread(fetcher.get_logs_generic, target, is_file, request.hours)

    if not result["logs"] or not result["logs"].strip():
        return {"status": "empty", "content": "", "stats": result["stats"]}

    return {"status": "success", "content": result["logs"], "stats": result["stats"]}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ==========================================
# DIGEST (added 2026-09-27): whole-window, templated, severity-first
# ==========================================
# /fetch_single returns the LAST 1000 filtered lines, so on chatty containers the
# analyser only ever saw the final hours of the 24 h window. /digest reads the full
# window (streamed, bounded by lines, bytes and time), collapses repeated lines into
# templates with counts and first/last timestamps, and spends a fixed character budget
# on WARN/ERROR templates first. `meta.complete` is False whenever anything was cut, so
# the caller never treats a partial digest as proof that an issue went away.
from concurrent.futures import ThreadPoolExecutor

DIGEST_TAIL_LINES = int(os.getenv("DIGEST_TAIL_LINES", "60000"))
DIGEST_MAX_BYTES = int(os.getenv("DIGEST_MAX_BYTES", str(48 * 1024 * 1024)))
DIGEST_BUDGET_CHARS = int(os.getenv("DIGEST_BUDGET_CHARS", "5000"))
DIGEST_SAMPLE_CHARS = int(os.getenv("DIGEST_SAMPLE_CHARS", "260"))
DIGEST_WORKERS = int(os.getenv("DIGEST_WORKERS", "3"))

_DOCKER_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z?\s")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|(?<![\w\]])\[[0-9;]{1,8}m")
_VOLATILE = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2}(?::\d{2})?(?: ?[AP]M)?", re.I), "<ts>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<ts>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b", re.I), "<hex>"),
    (re.compile(r"\d+(?:\.\d+)?"), "#"),
    (re.compile(r"\s+"), " "),
]
_SEV = [
    (3, re.compile(r"\b(fatal|panic|critical|crit|traceback|segfault|oom|out of memory|killed)\b", re.I)),
    (2, re.compile(r"\b(error|err|exception|fail(?:ed|ure|s)?|refused|denied|unauthori[sz]ed|forbidden|"
                   r"timed? ?out|timeout|unreachable|unavailable|crash(?:ed)?|exited|abort(?:ed)?)\b", re.I)),
    (1, re.compile(r"\b(warn|warning|deprecat\w*)\b", re.I)),
]
_SEV_NAME = {3: "CRIT", 2: "ERR", 1: "WARN", 0: "INFO"}
# "error=nil", "0 errors", "errors: 0" and similar are not errors
_NOT_ERR = re.compile(r"\b(error=(nil|null|none|\"\")|0 errors?|errors?: ?0\b|no errors?|without errors?)", re.I)
_EXPLICIT = re.compile(r"(?:\b(?:lvl|level|severity|loglevel)[=:]\s*[\"']?([a-z]+)|\[(trace|debug|dbg|info|inf|notice|warn|warning|wrn|error|err|eror|fatal|ftl|crit|critical)\]|(?:^|\s|\|)(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)(?:\s|:|\|))", re.I)
_EXPLICIT_MAP = {"trace": 0, "debug": 0, "dbg": 0, "info": 0, "inf": 0, "notice": 0, "warn": 1, "warning": 1,
                 "wrn": 1, "error": 2, "err": 2, "eror": 2, "fatal": 3, "ftl": 3, "crit": 3, "critical": 3}
# an explicit-INFO line is bumped to WARN only on clear failure phrases
# (media titles like "Crash" or "The Fatal Five" must not trip it)
_INFO_BUMP = re.compile(r"\b(failed|failure|error|exception|timed out|timeout|refused|denied|unable to|could not|cannot|panic|traceback)\b", re.I)


def _severity(body: str) -> int:
    m = _EXPLICIT.search(body[:200])
    if m:
        lv = _EXPLICIT_MAP.get((m.group(1) or m.group(2) or m.group(3) or "").lower())
        if lv is not None:
            if lv == 0:
                return 1 if (_INFO_BUMP.search(body) and not _NOT_ERR.search(body)) else 0
            return lv
    for level, rx in _SEV:
        if rx.search(body):
            if level == 2 and _NOT_ERR.search(body) and not _SEV[0][1].search(body):
                return 0
            return level
    return 0


_INNER_TS = re.compile(r"(?:\bts=|\btime=)?[\"']?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?[\"']?\s?")


def _template(body: str) -> str:
    t = body.lower()
    for rx, rep in _VOLATILE:
        t = rx.sub(rep, t)
    return t.strip()


def build_digest(lines, budget=DIGEST_BUDGET_CHARS, sample_chars=DIGEST_SAMPLE_CHARS, noise=None, deadline=None):
    """lines: iterable of raw log lines (docker lines may carry an RFC3339 prefix)."""
    tmpl = {}
    raw = kept = 0
    first_ts = last_ts = ""
    timed_out = False
    for line in lines:
        raw += 1
        if deadline is not None and (raw & 1023) == 0 and time.time() > deadline:
            timed_out = True
            break
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + " ...[line truncated]"
        s = _ANSI.sub("", line).strip()
        if not s:
            continue
        ts = ""
        m = _DOCKER_TS.match(s)
        if m:
            ts = m.group(1)
            s = s[m.end():].strip()
            if not s:
                continue
        sev = _severity(s)
        # key = FULL normalised content + severity: two lines never share a template just
        # because they share a long prefix, and an ERROR never inherits an INFO's level
        key = (sev, _template(s))
        e = tmpl.get(key)
        if e is None:
            # the noise regex is expensive (~10 ms/line on long lines), so it runs once per
            # TEMPLATE, not once per line; a noisy template is remembered and skipped
            tmpl[key] = e = {"n": 0, "first": ts, "last": ts, "raw": _INNER_TS.sub("", s).strip(),
                             "sev": sev, "order": kept + 1,
                             "noise": bool(noise is not None and not noise(s))}
        if e["noise"]:
            continue
        kept += 1
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
            e["first"] = e["first"] or ts
            e["last"] = ts
        e["n"] += 1
        e["recent"] = kept

    def fmt(e, chars):
        span = ""
        if e["first"]:
            a, b = e["first"][5:16].replace("T", " "), e["last"][5:16].replace("T", " ")
            span = f" {a}" if e["n"] == 1 or a == b else f" {a}..{b}"
        cnt = f" x{e['n']}" if e["n"] > 1 else ""
        r = e["raw"]
        if len(r) > chars:   # keep head AND tail: the distinguishing part is often at the end
            head = int(chars * 0.6)
            r = r[:head] + " ... " + r[-(chars - head - 5):]
        return f"[{_SEV_NAME[e['sev']]}{cnt}{span}] {r}"

    live = [e for e in tmpl.values() if not e["noise"] and e["n"] > 0]
    signal = sorted((e for e in live if e["sev"] > 0), key=lambda e: (-e["sev"], -e["n"], e["order"]))
    info = [e for e in live if e["sev"] == 0]
    rare_info = sorted((e for e in info if e["n"] <= 3), key=lambda e: -e["recent"])
    freq_info = sorted((e for e in info if e["n"] > 3), key=lambda e: -e["n"])

    header_room = 260  # the header line is part of the budget
    sig_budget = budget - header_room
    # Signal first. If it does not fit at full sample length, shorten the samples
    # (260 -> 160 -> 100 chars) before dropping anything; INFO then gets what is left.
    chars = sample_chars
    for c in (sample_chars, 160, 100):
        chars = c
        if sum(len(fmt(e, c)) + 1 for e in signal) <= sig_budget:
            break
    out, used, dropped_signal = [], 0, 0
    for e in signal:
        line = fmt(e, chars)
        if used + len(line) + 1 > sig_budget:
            dropped_signal += 1
            continue
        out.append(line)
        used += len(line) + 1
    info_cap = min(sig_budget, used + int(budget * 0.5))
    for e in freq_info[:5]:               # the chattiest INFO templates, as counts
        line = fmt(e, sample_chars)
        if used + len(line) + 1 <= info_cap:
            out.append(line)
            used += len(line) + 1
    for e in rare_info:                   # rare INFO lines (starts, config, state changes), newest first
        line = fmt(e, sample_chars)
        if used + len(line) + 1 > info_cap:
            break
        out.append(line)
        used += len(line) + 1

    header = (f"window {first_ts or '?'} .. {last_ts or '?'} | raw_lines {raw} | kept {kept} | "
              f"templates {len(live)} (signal {len(signal)}) | shown {len(out)}"
              + (f" | SIGNAL_TRUNCATED {dropped_signal}" if dropped_signal else "")
              + (" | PROCESSING_TIMED_OUT" if timed_out else ""))[:header_room - 1]
    meta = {"raw_lines": raw, "kept_lines": kept, "templates": len(live), "signal_templates": len(signal),
            "signal_dropped": dropped_signal, "shown": len(out), "window_first": first_ts, "window_last": last_ts,
            "timed_out": timed_out}
    return (header + "\n" + "\n".join(out)) if out else "", meta


def _iter_container_lines(container, since, deadline, info):
    """Stream docker logs line by line, bounded by lines, bytes and time (no full materialisation).
    follow=False is essential: docker-py defaults follow to the value of stream and would block forever."""
    stream = container.logs(since=since, timestamps=True, tail=DIGEST_TAIL_LINES, stdout=True, stderr=True, stream=True, follow=False)
    buf, total, lines = b"", 0, 0
    try:
        for chunk in stream:
            total += len(chunk)
            if total > DIGEST_MAX_BYTES:
                info["byte_capped"] = True
                break
            if time.time() > deadline:
                info["fetch_timed_out"] = True
                break
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                ln, buf = buf[:i], buf[i + 1:]
                for part in ln.decode("utf-8", errors="ignore").split("\r"):
                    lines += 1
                    yield part[:MAX_LINE_CHARS + 20]
            if len(buf) > MAX_LINE_CHARS * 4:           # a runaway line with no newline yet
                lines += 1
                yield buf[:MAX_LINE_CHARS].decode("utf-8", errors="ignore")
                buf = b""
        if buf:
            lines += 1
            yield buf[:MAX_LINE_CHARS].decode("utf-8", errors="ignore")
    finally:
        try:
            stream.close()
        except Exception:
            pass
        info["tail_capped"] = lines >= DIGEST_TAIL_LINES


def digest_one(name: str, hours: int, budget: int) -> Dict:
    t0 = time.time()
    deadline = t0 + MAX_PROCESS_SECONDS
    try:
        info = {}
        if name.startswith("/"):
            if not os.path.exists(name):
                return {"name": name, "status": "error", "digest": "", "error": "File not found"}
            lines = fetcher._tail_file_lines(name, MAX_LINES_PER_SOURCE * 5)
        else:
            if not fetcher.client:
                return {"name": name, "status": "error", "digest": "", "error": "Docker Client Missing"}
            container = fetcher.client.containers.get(name)
            st = container.attrs.get("State", {}) or {}
            info = {"started_at": st.get("StartedAt", ""), "restart_count": container.attrs.get("RestartCount", 0),
                    "health": (st.get("Health") or {}).get("Status", "")}
            lines = _iter_container_lines(container, datetime.now() - timedelta(hours=hours), deadline, info)
        lines = drop_container_noise(name, lines, info, deadline)
        digest, meta = build_digest(redact_lines(lines), budget=budget, noise=fetcher.filter_noise, deadline=deadline)
        meta.update(info)
        meta["complete"] = not (meta.get("timed_out") or meta.get("signal_dropped") or info.get("tail_capped")
                                or info.get("byte_capped") or info.get("fetch_timed_out")
                                or info.get("noise_timed_out"))
        meta["seconds"] = round(time.time() - t0, 2)
        return {"name": name, "status": "success" if digest else "empty", "digest": digest, "meta": meta}
    except Exception as e:
        return {"name": name, "status": "error", "digest": "", "error": str(e)[:300]}


class DigestRequest(BaseModel):
    containers: List[str]
    hours: int = 24
    budget_chars: int = DIGEST_BUDGET_CHARS


def _digest_many(names, hours, budget):
    with ThreadPoolExecutor(max_workers=max(1, DIGEST_WORKERS)) as ex:
        return list(ex.map(lambda n: digest_one(n, hours, budget), names))


@app.post("/digest")
async def digest(request: DigestRequest):
    names = [n for n in (request.containers or []) if n][:20]
    budget = max(1000, min(int(request.budget_chars or DIGEST_BUDGET_CHARS), 20000))
    results = await asyncio.to_thread(_digest_many, names, int(request.hours or 24), budget)
    return {"results": results}
