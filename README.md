# RAKUTEN_MLOPS

Projet de classification multimodale (texte + image) inspiré du dataset Rakuten, structuré pour un pipeline MLOps local et collaboratif.  
Modèles entraînables en local, environnement Python reproductible, code modulaire.

---

## Installation du projet

### 1. Cloner le dépôt

### 2. Créer et activer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash 
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Lancer l'entraînement des modèles

#### Texte 
```bash 
python -m models.rakuten_transformer_text.train
```

### Image
```bash
python -m models.rakuten_efficientnet_image.train
```

### Multimodal
```bash 
python -m models.multimodal_transformer_classifier.train
```
