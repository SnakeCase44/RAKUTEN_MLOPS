FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système minimales (optionnel mais recommandé)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uniquement les bibliothèques Python nécessaires à gateway.py
RUN pip install --no-cache-dir fastapi uvicorn httpx python-jose dotenv

# Copier seulement les fichiers nécessaires
COPY src ./src

# Exposer le port de FastAPI
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "src.gateway.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
