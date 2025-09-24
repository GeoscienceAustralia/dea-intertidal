# Base image with:
# - Ubuntu 22.04
# - Python 3.10.12
# - GDAL 3.7.3, released 2023/10/30
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.7.3

# curl and certificates are required to download uv,
# build-essential, libpq-dev and python3-dev are needed for psycopg2,
# git is needed for hatchling version control
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libpq-dev \
    python3-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Download, run and remove uv installer, and ensure it is on path
ADD https://astral.sh/uv/0.8.22/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy project files into image, and set as working directory
ADD . /app
WORKDIR /app

# Sync project into a new virtual environment based on uv.lock
RUN uv sync --locked --extra datacube

# Make uv virtual environment accessible
ENV PATH="/app/.venv/bin:$PATH"

# Verify installation
RUN uv pip check && dea-intertidal --help
