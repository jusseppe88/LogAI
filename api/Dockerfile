FROM python:3.9-slim

WORKDIR /app

# (Optional) You do NOT need docker.io CLI for docker SDK, but keeping it is okay.
# If you want a slimmer image, you can remove docker.io.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY log_analyzer.py .

EXPOSE 8765

# Health check uses stdlib (no requests dependency needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2).read()"

CMD ["uvicorn", "log_analyzer:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "2"]

