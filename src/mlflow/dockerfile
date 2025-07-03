FROM python:3.11-slim

RUN pip install mlflow

WORKDIR /app

RUN mkdir -p /mlflow/mlruns

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "file:///mlflow/mlruns", \
     "--default-artifact-root", "file:///mlflow/mlruns"]