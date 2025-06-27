FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installation uniquement des dépendances Streamlit
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    streamlit \
    requests \
    pandas \
    plotly

# Copier le code Streamlit depuis la racine
COPY . ./

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]