# syntax=docker/dockerfile:1.7
#
# Multi stage build.
#
# The builder installs into a virtual environment that is copied whole into the
# runtime stage, so compilers, headers and the pip cache never ship. The
# Microsoft ODBC driver is needed at both stages: the development headers to
# build pyodbc, and the runtime libraries to connect.
#
# Build with the extras a deployment actually uses:
#   docker build -f docker/Dockerfile --build-arg EXTRAS="local" -t nl2sql .

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Empty by default: the local model extra pulls torch, which adds gigabytes.
ARG EXTRAS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Microsoft ODBC Driver 18, which pyodbc needs to reach Azure SQL.
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
        https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install --no-install-recommends -y \
        msodbcsql18 \
        unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata first, so a source change does not invalidate the
# dependency layer.
COPY pyproject.toml README.md ./
COPY src/nl2sql/__init__.py ./src/nl2sql/__init__.py

RUN pip install --upgrade pip setuptools wheel \
    && if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install "."; fi

COPY src/ ./src/
RUN pip install --no-deps .

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NL2SQL_ENV=production \
    HF_HOME=/var/lib/nl2sql/models

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libodbc2 \
        odbcinst \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/microsoft /opt/microsoft
COPY --from=builder /etc/odbcinst.ini /etc/odbcinst.ini
COPY --from=builder /opt/venv /opt/venv

# An unprivileged user. Nothing here needs root.
RUN groupadd --system --gid 1001 nl2sql \
    && useradd --system --uid 1001 --gid nl2sql --create-home nl2sql

WORKDIR /app

COPY --chown=nl2sql:nl2sql configs/ ./configs/
COPY --chown=nl2sql:nl2sql prompts/ ./prompts/
COPY --chown=nl2sql:nl2sql migrations/ ./migrations/
COPY --chown=nl2sql:nl2sql evaluation/ ./evaluation/
COPY --chown=nl2sql:nl2sql alembic.ini ./

# Writable paths: the local application database in development, and the model
# cache when the local model extra is installed.
RUN mkdir -p /app/data /var/lib/nl2sql/models \
    && chown -R nl2sql:nl2sql /app/data /var/lib/nl2sql

USER nl2sql

EXPOSE 8000

# Liveness only. A readiness based health check would restart every replica
# whenever the database blipped.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# Migrations are not run here. A container that migrates on start races every
# other replica during a rolling deploy; run them as a separate job first.
CMD ["uvicorn", "nl2sql.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-access-log", "--proxy-headers"]
