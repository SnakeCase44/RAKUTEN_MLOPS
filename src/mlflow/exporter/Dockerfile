FROM python:3.10-slim

WORKDIR /app

RUN pip install prometheus_client

COPY exporter.py .

CMD ["python", "exporter.py"]
