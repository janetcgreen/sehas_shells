FROM e3-devops.aero.org/ssdhas/python:3.9.13-buster

# Install python/pip
ENV PYTHONUNBUFFERED=1
RUN python3 -m ensurepip
RUN pip install --no-cache --upgrade pip setuptools wheel

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "/app/python/app.py"]

