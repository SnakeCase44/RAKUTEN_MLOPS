## Modèle de classification de texte avec XLM-RoBERTa

Ce module implémente un classifieur de texte multilingue basé sur **xlm-roberta-base** (modèle préentraîné de Facebook AI). Il est conçu pour être robuste, flexible, et adapté aux tâches complexes de classification de texte multiclasse. Le pipeline utilise PyTorch Lightning pour faciliter l'entraînement, la sauvegarde, et l'évaluation du modèle.

---

### Configuration principale (`TRANSFORMER_CONFIG`)

```python
TRANSFORMER_CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "lr": 2e-5,
    "max_epochs": 4,
    "model_name": "FacebookAI/xlm-roberta-base",
    "unfreeze_layers": 4,
    "patience": 2,
    "weight_decay": 0.015,
}
```

> Remarques :
>
> * `max_len=128` est choisi pour des raisons de performance (temps + mémoire).
> * 4 couches sont dégelées pour un compromis entre spécialisation et stabilité.
> * Un `ReduceLROnPlateau` dynamique ajuste le learning rate selon la validation.

---

### Structure du modèle (`TransformerModel`)

1. **Backbone** : `xlm-roberta-base`, dont les couches finales peuvent être dégelées.
2. **Classifieur** : architecture dense progressive avec :

   ```text
   Linear(hidden→512) → GELU → LayerNorm → Dropout(0.5)
                     → Linear(512→256) → GELU → LayerNorm → Dropout(0.50)
                     → Linear(256→128) → GELU → LayerNorm → Dropout(0.40)
                     → Linear(128→nb_classes)
   ```
3. **Optimiseur** : AdamW, avec gestion fine du `weight_decay`.
4. **Scheduler** : `ReduceLROnPlateau` sur `val_loss`, avec les paramètres suivants :
   - `factor=0.1`
   - `patience=1`
   - `threshold=0.015`
   - `threshold_mode=rel`

> Les métriques `Accuracy` et `F1-score` pondéré sont suivies à chaque étape d'entraînement et validation.

---

### Dataset (`TransformerDataset`)

Transforme les textes bruts + labels numériques en un dataset compatible PyTorch :

* Tokenisation (`padding`, `truncation`, `max_len`)
* Conversion en tenseurs
* Intégration des labels pour supervision

---

### Classe `TextClassifier`

Point d'entrée principal pour l'entraînement, la sauvegarde, l'évaluation :

#### 🔹 `fit()`

* Nettoyage des textes (via `clean_text` et une `cleaning_config` personnalisable)
* Encodage des labels avec `LabelEncoder`
* Création des `DataLoader`
* Initialisation du modèle + callbacks (`EarlyStopping`, `ModelCheckpoint`)
* Logging des métriques dans `metrics/version_*/metrics.csv`

#### 🔹 `evaluate()`

* Prédiction sur de nouveaux textes nettoyés
* Calcul du `classification_report` (avec `zero_division=0`)
* Affichage + export éventuel en `.txt`

#### 🔹 `save()` / `load()`

* Sauvegarde du modèle (`.pth`), tokenizer (`.json`) et du `LabelEncoder`

---

### Entraînement & Validation (Lightning)

À chaque batch :

* Passage du batch dans le modèle
* Calcul de la `loss` via `cross_entropy` avec label smoothing (0.1)
* Mise à jour des métriques (acc, f1)
* Logging dynamique (console + CSVLogger)

À la fin de chaque époque :

* Affichage des métriques (DEBUG)
* Reset des compteurs

---

### Récapitulatif des hyperparamètres

| Paramètre         | Valeur                       | Description                              |
| ----------------- | ---------------------------- | ---------------------------------------- |
| `max_len`         | 128                          | Longueur max d'un texte (post nettoyage) |
| `batch_size`      | 32                           | Nombre d'exemples par batch              |
| `lr`              | 2e-5                         | Learning rate initial                    |
| `max_epochs`      | 4                            | Nb d'époques max                         |
| `model_name`      | FacebookAI/xlm-roberta-base  | Modèle de base                           |
| `unfreeze_layers` | 4                            | Nb de couches dégélées du RoBERTa        |
| `patience`        | 2                            | Pour `EarlyStopping`                     |
| `weight_decay`    | 0.015                        | Poids de régularisation                  |

---

### Suivi & performances

*  `metrics.csv` : journal des pertes et scores à chaque époque
*  `checkpoints/` : sauvegardes auto du meilleur modèle (selon `val_f1`)
*  Console : DEBUG des métriques (Accuracy, F1) à chaque fin d'époque

---

### Détails techniques additionnels

1. **Early Stopping** : Arrêt de l'entrainement après 2 époques sans amélioration de `val_f1` (amélioration min: 0.015)
2. **Label Smoothing** : Réduction de l'overconfidence avec `label_smoothing=0.1` dans la fonction de perte
3. **Gradient Clipping** : Stabilisation de l'entrainement avec `gradient_clip_val=1.0`
4. **Parallelisation** : Utilisation de 4 workers pour le chargement des données (`num_workers=4`)
5. **Embeddings** : Méthode `get_embedding()` disponible pour extraire le vecteur [CLS] sans passer par le classifieur

---
![newplot](https://github.com/user-attachments/assets/8aad10fe-8602-4893-a235-96b606dd13c7)
