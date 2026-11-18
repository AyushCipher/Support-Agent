# Production image for the supportagent API (chat UI, reviewer desk, /metrics). See docs/deploy.md.
#  - runs as a non-root user (uid 10001); application code is root-owned and read-only to that user
#  - no secrets are baked in: every key comes from the platform's secret store at runtime (render.yaml / fly.toml / -e)
#  - SA_ENV=production makes startup REFUSE an insecure configuration (security/prodcheck.py) instead of running degraded
#  - only src/ and policies/ ship: no evals, results, tests, docs or .env (see .dockerignore)
# Pin the base image by digest for reproducible builds: docker pull python:3.12-slim && docker inspect --format '{{index .RepoDigests 0}}'
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONPATH=/app/src
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY policies ./policies
# crypto is required in production (AES-GCM at rest, EdDSA agent identities). Add anthropic/otel when you use them:
#   docker build --build-arg EXTRAS=crypto,anthropic,otel .      (Gemini needs no extra: stdlib HTTP client)
ARG EXTRAS="crypto"
RUN pip install --no-cache-dir ".[${EXTRAS}]" && rm -rf /root/.cache build *.egg-info src/*.egg-info
RUN mkdir -p /app/runs && chown app:app /app/runs
USER app
ENV PORT=8080 HOST=0.0.0.0 SA_RUNS_DIR=/app/runs SA_CONFIG=full
EXPOSE 8080
VOLUME ["/app/runs"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz',timeout=2)"
CMD ["python", "-m", "supportagent.api.server"]
