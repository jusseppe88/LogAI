from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import docker
import re
import os
import asyncio
import ssl
import time
from typing import List, Optional, Dict

app = FastAPI()

# ==========================================
# CONFIGURATION
# ==========================================
SYSTEM_LOG_FILES = ["/var/log/syslog", "/var/log/auth.log", "/var/log/kern.log"]

# Containers to ALWAYS ignore (Noise reduction)
IGNORE_CONTAINERS = ["LogAI", "ollama", "qdrant"]

# Final output cap (what you return to n8n)
MAX_LINES_PER_SOURCE = int(os.getenv("MAX_LINES_PER_SOURCE", "1000"))

# Critical fix: cap docker log retrieval at the source
# Start with 8000; adjust if you filter too much noise and end up with too few useful lines.
DOCKER_LOG_TAIL_LINES = int(os.getenv("DOCKER_LOG_TAIL_LINES", "8000"))

# Safety: if processing takes too long, bail out (prevents wedging on very noisy containers)
MAX_PROCESS_SECONDS = int(os.getenv("MAX_PROCESS_SECONDS", "120"))

# ==========================================
# SMTP PROXY CONFIGURATION (Port 2526)
# ==========================================
MAILCOW_IP = os.getenv("MAILCOW_IP", "127.0.0.1")
MAILCOW_PORT = int(os.getenv("MAILCOW_PORT", 465))


async def handle_smtp_client(client_reader, client_writer):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        backend_reader, backend_writer = await asyncio.open_connection(
            MAILCOW_IP, MAILCOW_PORT, ssl=ctx
        )

        async def pipe(r, w):
            try:
                while True:
                    data = await r.read(4096)
                    if not data:
                        break
                    w.write(data)
                    await w.drain()
            except Exception:
                pass

        await asyncio.gather(
            pipe(client_reader, backend_writer),
            pipe(backend_writer, client_writer),
        )
    except Exception:
        pass
    finally:
        client_writer.close()


@app.on_event("startup")
async def start_proxy():
    if MAILCOW_IP != "127.0.0.1":
        print(f"Starting SMTP Proxy forwarding to {MAILCOW_IP}:{MAILCOW_PORT}")
        server = await asyncio.start_server(handle_smtp_client, "0.0.0.0", 2526)
        asyncio.create_task(server.serve_forever())
    else:
        print("SMTP Proxy disabled (No MAILCOW_IP env var set)")


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

    def filter_noise(self, log_line: str) -> bool:
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
            # Mailcow specific
            r".*connect from unknown.*",
            r".*lost connection.*",
            r".*disconnect from unknown.*",
            r".*statistics:.*",
            r".*NOQUEUE: reject:.*",
            r".*pop3-login: Disconnected.*",
            r".*imap-login: Disconnected.*",
            r".*sieve:.*",
            r".*warning:.*hostname.*does not resolve.*",
        ]
        return not any(re.search(p, log_line, re.IGNORECASE) for p in noise_patterns)

    def get_stats(self, container_name: str) -> str:
        if not self.client:
            return "Docker Error"
        try:
            container = self.client.containers.get(container_name)
            if container.status != "running":
                return "Status: Not Running"

            stats = container.stats(stream=False)

            # CPU
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
                cpu_percent = 0.0
                if system_cpu_delta > 0.0:
                    cpu_percent = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
            except Exception:
                cpu_percent = 0.0

            # Memory
            try:
                mem_usage = stats["memory_stats"]["usage"] / (1024 * 1024)
                mem_limit = stats["memory_stats"]["limit"] / (1024 * 1024)
                mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit else 0.0
            except Exception:
                mem_usage = 0.0
                mem_limit = 0.0
                mem_percent = 0.0

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
                    with open(name, "r", errors="ignore") as f:
                        raw_lines = f.readlines()[-MAX_LINES_PER_SOURCE:]
                    stats_info = "Source: System Log File"
                else:
                    return {"logs": "", "stats": "File not found"}

            else:
                if not self.client:
                    return {"logs": "", "stats": "Docker Client Missing"}

                container = self.client.containers.get(name)

                # Important: bound log retrieval
                since = datetime.now() - timedelta(hours=hours)

                logs_bytes = container.logs(
                    since=since,
                    timestamps=True,
                    tail=DOCKER_LOG_TAIL_LINES,
                    stdout=True,
                    stderr=True,
                )

                logs = logs_bytes.decode("utf-8", errors="ignore")
                raw_lines = logs.split("\n")

                stats_info = self.get_stats(name)

            # Filter + safety timeout
            relevant_logs: List[str] = []
            for line in raw_lines:
                if time.time() - start_time > MAX_PROCESS_SECONDS:
                    # Bail out rather than wedge the API
                    stats_info = f"{stats_info} | NOTE: processing timed out after {MAX_PROCESS_SECONDS}s"
                    break

                if self.filter_noise(line):
                    relevant_logs.append(line.strip())

            log_content = "\n".join(relevant_logs[-MAX_LINES_PER_SOURCE:])
            return {"logs": log_content, "stats": stats_info}

        except Exception as e:
            return {"logs": "", "stats": f"Error fetching source: {str(e)}"}


fetcher = LogFetcher()

# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/list_targets")
async def list_targets():
    targets = []

    for f in SYSTEM_LOG_FILES:
        targets.append({"name": f, "type": "file", "pretty_name": f"SYSTEM: {f}"})

    if fetcher.client:
        try:
            containers = fetcher.client.containers.list(filters={"status": "running"})
            for c in containers:
                if c.name in IGNORE_CONTAINERS:
                    continue
                targets.append(
                    {"name": c.name, "type": "container", "pretty_name": f"DOCKER: {c.name}"}
                )
        except Exception as e:
            print(f"Error listing containers: {e}")

    return {"targets": targets}


@app.post("/fetch_single")
async def fetch_single(request: LogRequest):
    target = request.containers[0]
    is_file = target.startswith("/")

    # Critical fix: offload blocking docker/file IO + parsing to a worker thread
    result = await asyncio.to_thread(fetcher.get_logs_generic, target, is_file, request.hours)

    if not result["logs"] or not result["logs"].strip():
        return {"status": "empty", "content": "", "stats": result["stats"]}

    return {"status": "success", "content": result["logs"], "stats": result["stats"]}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

