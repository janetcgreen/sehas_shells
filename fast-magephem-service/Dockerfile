FROM e3-devops.aero.org/ssdhas/python:3.9.13-buster

WORKDIR /app

COPY python /app/python
COPY python/lib /usr/local/lib64/
COPY test /app/test
COPY requirements.txt .


RUN DOW=$(date +%a) \
    && export http_proxy=http://31180:$DOW@proxy-west.aero.org:8080 \
    && export https_proxy=http://31180:$DOW@proxy-west.aero.org:8080 \
    && apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y libhdf5-dev \
    && apt-get install -y libxerces-c-dev \
    && pip3 install --upgrade pip setuptools \
    && pip3 install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/libhdf5.so.103

# Run app.py
CMD ["python", "/app/python/app.py"]
