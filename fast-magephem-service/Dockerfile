FROM python:3.9.17-slim-bookworm

WORKDIR /app

#Removing warning in Jenkins about apt-util not being installed, causing containerization to hang
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"

RUN apt-get update && apt-get install -y\
    build-essential \
    cmake \
    gfortran \
    libboost-all-dev \
    libopenmpi-dev \
    googletest \
    libhdf5-dev \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y Flask \
    && pip install Flask==2.2.3

COPY python /app/python
COPY python/lib /usr/local/lib64/
COPY test /app/test

RUN ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/libhdf5.so.103

# Run pytest tests
RUN pytest /app/test

CMD ["python", "/app/python/app.py"]
