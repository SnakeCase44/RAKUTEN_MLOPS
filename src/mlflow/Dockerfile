FROM python:3.11-slim

# Installer les dépendances système pour psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer MLflow et psycopg2
RUN pip install mlflow psycopg2-binary

WORKDIR /app
RUN mkdir -p /mlflow/mlruns
EXPOSE 5005

CMD ["mlflow", "server", \
"--host", "0.0.0.0", \
"--port", "5005", \
"--backend-store-uri", "file:///mlflow/mlruns", \
"--default-artifact-root", "file:///mlflow/mlruns"]