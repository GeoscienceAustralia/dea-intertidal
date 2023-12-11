FROM osgeo/gdal:ubuntu-small-3.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Apt installation
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      fish \
      git \
      vim \
      htop \
      wget \
      unzip \
      python3-pip \
      libpq-dev python-dev \
    && apt-get autoclean && \
    apt-get autoremove && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Pip installation
RUN mkdir -p /conf
COPY requirements.txt /conf/
RUN pip install -r /conf/requirements.txt

# Copy source code and install it
RUN mkdir -p /code
WORKDIR /code
ADD . /code

RUN echo "Installing dea-burn-cube through the Dockerfile."
RUN pip install --extra-index-url="https://packages.dea.ga.gov.au" .

RUN pip freeze && pip check

# Make sure it's working
RUN dea-burn-cube --version
