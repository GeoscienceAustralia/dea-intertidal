# Base image with:
# - Ubuntu 22.04
# - Python 3.10.12
# - GDAL 3.7.3, released 2023/10/30
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.7.3

# The installer requires curl (and certificates) to download uv
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

# Download, run and remove uv installer
ADD https://astral.sh/uv/0.8.22/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked --extra datacube --system

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Verify installation
RUN uv pip check && \
    dea-intertidal --help


# ENV DEBIAN_FRONTEND=noninteractive \
#     LC_ALL=C.UTF-8 \
#     LANG=C.UTF-8

# # Apt installation
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       curl \
#       ca-certificates \
#       build-essential \
#       git \
#       python3-pip \
#       libpq-dev \
#     && apt-get autoclean && \
#     apt-get autoremove && \
#     rm -rf /var/lib/{apt,dpkg,cache,log}

# # Set up working directory
# WORKDIR /app








# # Accept build-time argument for requirements file
# ARG REQUIREMENTS_IN=requirements.in

# # Copy requirements file first
# COPY ${REQUIREMENTS_IN} /app/requirements.in

# # Install uv and requirements
# RUN pip install uv && \
#     uv pip compile /app/requirements.in -o /app/requirements.txt --emit-find-links && \
#     uv pip install -r /app/requirements.txt --system

# # Now copy the rest of the files
# COPY . /app

# # Install DEA Intertidal with postgres extra and verify installation
# RUN uv pip install .[datacube] --system && \
#     uv pip check && \
#     dea-intertidal --help
