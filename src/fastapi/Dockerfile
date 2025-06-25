FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers nécessaires depuis la racine
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

# Copier le code
COPY src/ ./src/
COPY models/ ./models/
COPY config.py ./

CMD ["uvicorn", "src.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
