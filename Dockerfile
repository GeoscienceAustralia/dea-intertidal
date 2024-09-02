# Base image with:
# - Ubuntu 22.04
# - Python 3.10.12
# - GDAL 3.7.3, released 2023/10/30
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.7.3

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Apt installation
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      git \
      python3-pip \
      libpq-dev \
    && apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Set up working directory and copy in code
WORKDIR /app
COPY . .

# Install requirements
RUN pip install uv && \
    uv pip compile requirements.in -o requirements.txt && \
    uv pip install -r requirements.txt --system

# Install DEA Intertidal and verify installation
RUN uv pip install . --system && \
    uv pip check && \
    dea-intertidal --help