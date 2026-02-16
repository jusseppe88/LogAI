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
                raw_lines = logs.split("\n")

                stats_info = self.get_stats(name)

            relevant_logs: List[str] = []
            for line in raw_lines:
                if time.time() - start_time > MAX_PROCESS_SECONDS:
                    stats_info = f"{stats_info} | NOTE: processing timed out after {MAX_PROCESS_SECONDS}s"
                    break

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
