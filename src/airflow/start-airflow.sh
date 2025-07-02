#!/bin/bash
set -e

echo "=== Starting Airflow initialization ==="

# Attendre PostgreSQL
echo "Waiting for PostgreSQL..."
while ! pg_isready -h postgres -p 5432 -U admin; do
  sleep 2
done
echo "PostgreSQL is ready!"

# Initialiser Airflow
echo "Initializing Airflow database..."
airflow db init

# Créer l'utilisateur admin
echo "Creating admin user..."
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com || echo "User exists"

# Démarrer le scheduler
echo "Starting Airflow scheduler..."
airflow scheduler &

# Démarrer le webserver
echo "Starting Airflow webserver..."
exec airflow webserver --port 8080 --host 0.0.0.0