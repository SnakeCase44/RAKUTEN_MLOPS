FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires depuis le contexte parent
COPY ../../requirements.txt ../../setup.py ./

# Installation des dépendances FastAPI + Airflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e . && \
    pip install apache-airflow==2.9.1

# Copier le code depuis le contexte parent
COPY ../../src/ ./src/
COPY ../../models/ ./models/
COPY ../../tests/ ./tests/
COPY ../../config.py ./

# Définir les variables d'environnement Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="/root/.local/bin:$PATH"

# Créer les répertoires nécessaires
RUN mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins

# Script de démarrage
COPY src/airflow/start-airflow.sh /start-airflow.sh
RUN chmod +x /start-airflow.sh

CMD ["/start-airflow.sh"]