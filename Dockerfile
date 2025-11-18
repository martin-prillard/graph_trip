############
# Builder  #
############

FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies for building wheels (geospatial + ML stack)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgdal-dev \
        libgeos-dev \
        libproj-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip && \
    pip install uv

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock* ./

# Create an isolated environment and install project deps into it
RUN uv sync --python /usr/local/bin/python --locked

# Install PyTorch (CPU) and PyTorch Geometric into the same environment
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install torch_geometric


############
# Runtime  #
############

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime system libs only (no build tools to keep image small)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgdal-dev \
        libgeos-dev \
        libproj-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built environment from builder stage (installed by uv)
# Keep it outside /app so bind mounts don't override it
COPY --from=builder /build/.venv /opt/venv

# Provide a stable path for tools that were installed while the env lived under /build/.venv
RUN mkdir -p /build && ln -s /opt/venv /build/.venv

# Make the virtualenv the default Python
ENV VIRTUAL_ENV="/opt/venv" \
    PATH="/opt/venv/bin:$PATH"

# Copy project files (notebooks, scripts, etc.)
COPY . .

# Expose JupyterLab default port
EXPOSE 8888

# Launch JupyterLab on container start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]

