# Base image with:
# - Ubuntu 22.04
# - Python 3.10.12
# - GDAL 3.7.3, released 2023/10/30
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.7.3

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install system dependencies and remove the compromised
# emSign ECC Root CA - G3 certificate from the system trust store.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      build-essential \
      libpq-dev \
      python3-dev \
      git && \
    if grep -q '^mozilla/emSign_ECC_Root_CA_-_G3.crt$' /etc/ca-certificates.conf; then \
      sed -i \
        's|^mozilla/emSign_ECC_Root_CA_-_G3.crt$|!mozilla/emSign_ECC_Root_CA_-_G3.crt|' \
        /etc/ca-certificates.conf; \
    fi && \
    update-ca-certificates --fresh && \
    ! grep -Ril "emSign ECC Root CA - G3" \
      /etc/ssl/certs \
      /usr/local/share/ca-certificates \
      2>/dev/null && \
    rm -rf /var/lib/apt/lists/*

# eMudhra/emSign CA compromise remediation.
ARG UV_VERSION=0.13.4

# Download and install the approved uv version.
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && \
    rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

# Copy project files into image.
ADD . /app
WORKDIR /app

# Sync exactly from uv.lock.
RUN uv sync --locked --extra datacube

# Make the uv virtual environment accessible.
ENV PATH="/app/.venv/bin:$PATH"

# Verify installation.
RUN uv pip check && \
    dea-intertidal --help
