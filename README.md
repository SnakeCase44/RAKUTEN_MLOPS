# RAKUTEN_MLOPS

---

## Installation locale (mode dev)

### 1. Cloner le dépôt

```bash
git clone <url_du_repo>
cd RAKUTEN_MLOPS
```

---

### 2. Créer et activer un environnement virtuel (facultatif en mode Docker)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Installer les dépendances (hors Docker uniquement)

```bash 
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Lancer le projet avec Docker

### 1. Construire les images

```bash
docker compose build
```

### 2. Démarrer les services

```bash
docker compose up
```

---

## Accès à l'API FastAPI

- Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
- Route test : [http://localhost:8000/test](http://localhost:8000/test) → doit répondre `Hello, Rakuten World!`

---

## Structure du projet

```
RAKUTEN_MLOPS/
├── docker-compose.yml
├── .env
├── config.py
├── requirements.txt
├── setup.py
├── dataset/                        # Contient train.csv, val.csv
├── models/
│   ├── multimodal_transformer_classifier/
│   ├── rakuten_transformer_text/
│   └── rakuten_efficientnet_image/
├── src/
│   └── fastapi/
│       ├── Dockerfile             # Dockerfile dédié à l’API
│       ├── main.py                # Entrée de l’API
│       └── endpoints/
│           ├── __init__.py
│           └── test.py            # Route GET /test
├── logs/
└── ...
```

---

## Routes disponibles

| Méthode | Endpoint     | Description                    |
|---------|--------------|--------------------------------|
| GET     | `/test`      | Vérifie que l’API fonctionne   |

---

---

## Astuces Docker

- Voir l'espace utilisé : `docker system df`
- Nettoyer les images et containers inutiles : `docker system prune -a --volumes`

---

