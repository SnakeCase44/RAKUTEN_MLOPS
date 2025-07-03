FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*


# Copier les fichiers nécessaires depuis la racine
COPY requirements.txt setup.py ./

# Installation simple - laisser pip gérer les versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Copier le code ET les tests
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/
COPY config.py ./

CMD ["uvicorn", "src.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]