FROM rakuten_fastapi

# Installer curl (et éventuellement d'autres utilitaires utiles)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer Apache Airflow
RUN pip install apache-airflow==2.9.1

# Définir les variables d'environnement
ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="/root/.local/bin:$PATH"

# Créer les répertoires nécessaires
RUN mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins

# Script de démarrage
COPY start-airflow.sh /start-airflow.sh
RUN chmod +x /start-airflow.sh

CMD ["/start-airflow.sh"]
