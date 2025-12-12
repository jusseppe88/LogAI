from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
import docker
import re
import os
import asyncio
import ssl
from typing import List, Optional, Dict, Any

app = FastAPI()

# ==========================================
# CONFIGURATION
# ==========================================
# Files to monitor (Static)
# Modify this list if your system logs are in different locations
SYSTEM_LOG_FILES = ["/var/log/syslog", "/var/log/auth.log", "/var/log/kern.log"]

# Containers to ALWAYS ignore (Noise reduction)
# Add any container names here that you do not want to analyze
IGNORE_CONTAINERS = ["LogAI", "ollama", "qdrant"]

MAX_LINES_PER_SOURCE = 1000

# ==========================================
# SMTP PROXY CONFIGURATION (Port 2526)
# ==========================================
# These are now loaded from Environment Variables defined in docker-compose.yml
MAILCOW_IP = os.getenv("MAILCOW_IP", "127.0.0.1") 
MAILCOW_PORT = int(os.getenv("MAILCOW_PORT", 465))

async def handle_smtp_client(client_reader, client_writer):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        backend_reader, backend_writer = await asyncio.open_connection(MAILCOW_IP, MAILCOW_PORT, ssl=ctx)
        
        async def pipe(r, w):
            try:
                while True:
                    data = await r.read(4096)
                    if not data: break
                    w.write(data)
                    await w.drain()
            except: pass
        
        await asyncio.gather(pipe(client_reader, backend_writer), pipe(backend_writer, client_writer))
    except: pass
    finally: client_writer.close()

@app.on_event("startup")
async def start_proxy():
    # Only start the proxy if we have a valid target IP
    if MAILCOW_IP != "127.0.0.1":
        print(f"Starting SMTP Proxy forwarding to {MAILCOW_IP}:{MAILCOW_PORT}")
        server = await asyncio.start_server(handle_smtp_client, '0.0.0.0', 2526)
        asyncio.create_task(server.serve_forever())
    else:
        print("SMTP Proxy disabled (No MAILCOW_IP env var set)")

# ==========================================
# LOGIC
# ==========================================
class LogRequest(BaseModel):
    containers: Optional[List[str]] = None
    hours: int = 24

class LogFetcher:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"CRITICAL: Could not connect to Docker socket. Ensure /var/run/docker.sock is mounted. Error: {e}")
            self.client = None

    def filter_noise(self, log_line: str) -> bool:
        noise_patterns = [
            r'^\s*$', r'GET /health.*200', r'HEAD /.*200', r'.*heartbeat.*', r'.*ping.*pong.*',
            r'CRON.*pam_unix.*session', r'nginx.*(GET|POST|HEAD).* 200 ', r'nginx.*(GET|POST|HEAD).* 304 ',
            r'\[Nest\].*Request.*200', r'.*Processing job.*', r'.*Face detection.*', r'.*CLIP encoder.*',
            r'.*Encoded.*images.*', r'.*Loading model.*', r'.*checkpoint starting.*', r'.*checkpoint complete.*',
            r'.*RssSyncService.*', r'.*DownloadDecisionMaker.*', r'.*TrackedDownloadService.*Processing.*',
            r'.*HousekeepingService.*', r'.*Scheduler.*', r'.*Refresh.*Service.*', r'.*flaresolverr.*GET /v1',
            r'.*prowlarr.*api/v1/indexer', r'.*IP geolocation database loaded.*', r'.*Successfully parsed.*',
            r'query\[A\]', r'query\[AAAA\]', r'.*INF.*Connection.*registered.*', r'.*INF.*Quic.*connection.*',
            r'.*INF.*Generated.*event.*', r'.*IP address.*matches.*', r'.*Checking.*IP.*',
            r'.*healthMonitoring.*succeeded but took.*',
            r'.*No active video sessions found.*',
            r'.*tRPC request from.*',
            r'.*healthMonitoring.*succeeded but took.*',
            r'.*No active video sessions found.*',
            r'.*tRPC request from.*',
            r'.*Failed to get releases.*',
            # Mailcow specific
            r'.*connect from unknown.*', r'.*lost connection.*', r'.*disconnect from unknown.*',
            r'.*statistics:.*', r'.*NOQUEUE: reject:.*', r'.*pop3-login: Disconnected.*',
            r'.*imap-login: Disconnected.*', r'.*sieve:.*', r'.*warning:.*hostname.*does not resolve.*'
        ]
        return not any(re.search(pattern, log_line, re.IGNORECASE) for pattern in noise_patterns)

    def get_stats(self, container_name: str) -> str:
        """Fetch Live CPU/RAM usage"""
        if not self.client: return "Docker Error"
        try:
            container = self.client.containers.get(container_name)
            if container.status != 'running':
                return "Status: Not Running"
            
            # Get stats (snapshot)
            stats = container.stats(stream=False)
            
            # Calculate CPU
            try:
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                number_cpus = stats['cpu_stats']['online_cpus']
                cpu_percent = 0.0
                if system_cpu_delta > 0.0:
                    cpu_percent = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
            except KeyError:
                cpu_percent = 0.0
            
            # Calculate Memory
            try:
                mem_usage = stats['memory_stats']['usage'] / (1024 * 1024) # MB
                mem_limit = stats['memory_stats']['limit'] / (1024 * 1024) # MB
                mem_percent = (mem_usage / mem_limit) * 100.0
            except KeyError:
                mem_usage = 0.0
                mem_limit = 0.0
                mem_percent = 0.0
            
            return f"LIVE STATS: CPU: {cpu_percent:.2f}% | RAM: {mem_usage:.1f}MB / {mem_limit:.1f}MB ({mem_percent:.1f}%)"
        except Exception as e:
            return f"Stats Error: {str(e)}"

    def get_logs_generic(self, name: str, is_file: bool, hours: int) -> Dict[str, str]:
        try:
            raw_lines = []
            stats_info = ""
            
            if is_file:
                if os.path.exists(name):
                    with open(name, 'r', errors='ignore') as f:
                        raw_lines = f.readlines()[-MAX_LINES_PER_SOURCE:]
                    stats_info = "Source: System Log File"
            else:
                # Docker
                if not self.client: return {"logs": "", "stats": "Docker Client Missing"}
                container = self.client.containers.get(name)
                # 1. Fetch Logs
                since = datetime.now() - timedelta(hours=hours)
                logs = container.logs(since=since, timestamps=True).decode('utf-8', errors='ignore')
                raw_lines = logs.split('\n')
                # 2. Fetch Stats
                stats_info = self.get_stats(name)

            # Filter
            relevant_logs = []
            for line in raw_lines:
                if self.filter_noise(line):
                    relevant_logs.append(line.strip())
            
            log_content = "\n".join(relevant_logs[-MAX_LINES_PER_SOURCE:])
            
            return {"logs": log_content, "stats": stats_info}
        except Exception:
            return {"logs": "", "stats": "Error fetching source"}

fetcher = LogFetcher()

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/list_targets")
async def list_targets():
    targets = []
    
    # 1. Add System Files
    for f in SYSTEM_LOG_FILES:
        targets.append({"name": f, "type": "file", "pretty_name": f"SYSTEM: {f}"})
    
    # 2. Add ALL Running Docker Containers (Dynamic)
    if fetcher.client:
        try:
            containers = fetcher.client.containers.list(filters={"status": "running"})
            for c in containers:
                # Skip ignored containers
                if c.name in IGNORE_CONTAINERS:
                    continue
                
                targets.append({
                    "name": c.name, 
                    "type": "container", 
                    "pretty_name": f"DOCKER: {c.name}"
                })
        except Exception as e:
            print(f"Error listing containers: {e}")

    return {"targets": targets}

@app.post("/fetch_single")
async def fetch_single(request: LogRequest):
    target = request.containers[0]
    is_file = target.startswith("/")
    
    result = fetcher.get_logs_generic(target, is_file, request.hours) 

    if not result["logs"] or len(result["logs"].strip()) == 0:
        return {"status": "empty", "content": "", "stats": result["stats"]}
    
    return {"status": "success", "content": result["logs"], "stats": result["stats"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
