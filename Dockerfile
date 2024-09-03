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

# Set up working directory
WORKDIR /app

# Install uv
# RUN pip install uv==0.4.3
COPY --from=ghcr.io/astral-sh/uv:0.4.3 /uv /bin/uv

# Copy input requirement and compile/install full requirements.txt
COPY requirements.in .
RUN uv pip compile requirements.in -o requirements.txt
RUN uv pip install -r requirements.txt --system

# Copy remainder of files, install DEA Intertidal, and verify installation
COPY . .
RUN uv pip install . --system && \
    uv pip check && \
    dea-intertidal --help
