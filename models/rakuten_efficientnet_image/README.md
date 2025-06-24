# Modèle de classification d'images avec EfficientNet-B2

Ce module implémente un classifieur d'images basé sur **EfficientNet-B2**, conçu pour traiter un dataset de produits avec images (Rakuten, ou autre). Le pipeline inclut prétraitement, fine-tuning partiel du backbone, entraînement avec gestion des métriques, et évaluation complète. L'approche est modulaire et adaptée à une utilisation sur GPU.

---

### Configuration par défaut (`DEFAULT_IMAGE_CLASSIFIER_CONFIG`)

```python
{
    "image_size": (384, 384),
    "batch_size": 32,
    "lr": 1e-4,
    "max_epochs": 5,
    "weight_decay": 3e-4,
    "unfreeze_layers": 6,
    "model_name": "efficientnet_b2",
    "patience": 2,
    "reduce_lr_factor": 0.4,
    "reduce_lr_patience": 1,
    "min_lr": 5e-6
}
```

Ce dictionnaire contrôle l'entraînement du modèle :

* **image\_size** : redimensionnement standard
* **unfreeze\_layers** : nombre de *blocs* du modèle à dégeler pour affiner l'apprentissage
* **patience** : nombre d'époques sans amélioration avant arrêt anticipé
* **reduce\_lr\_factor** : facteur de réduction du taux d'apprentissage
* **reduce\_lr\_patience** : nombre d'époques sans amélioration avant réduction du taux d'apprentissage

---

### Classe `ImageDataset`

Transforme un DataFrame de métadonnées images (`imageid`, `productid`) et les labels associés en un dataset PyTorch :

* Charge les images depuis le disque en utilisant OpenCV (`cv2.imread`)
* Applique un pipeline d'augmentation différent selon le mode (entraînement/validation):
  
  **En mode augmentation (entraînement):**
  ```python
  transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),  # Recadrage aléatoire avec zoom léger
      transforms.RandomHorizontalFlip(p=0.5),               # Retournement horizontal (50% de chance)
      transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),          # Variations de couleur/luminosité
      transforms.ToTensor(),
      transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))    # Effacement aléatoire de zones
  ])
  ```
  
  **En mode validation:**
  ```python
  transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((384, 384)),                        # Redimensionnement simple
      transforms.ToTensor()
  ])
  ```

* Normalise avec les moyennes et écarts-types standards ImageNet:
  ```python
  mean = torch.tensor([0.485, 0.456, 0.406])
  std = torch.tensor([0.229, 0.224, 0.225])
  ```

* Retourne un tuple `(image_tensor, label_tensor)`

> Les images doivent déjà être prétraitées et disponibles dans le dossier `IMAGE_PREPROCESSED_DIR`.

---

### Classe `LitImageModel`

Cœur du modèle basé sur `EfficientNet-B2` pré-entraîné :

* **Transfer Learning Stratégique**: 
  * Gèle d'abord toutes les couches du modèle pré-entraîné
  * Dégèle sélectivement les `n` derniers blocs (définis par `unfreeze_layers`) pour fine-tuning
  * Dégèle toujours le classifieur pour permettre l'adaptation à de nouvelles classes

* **Architecture du classifieur personnalisé**:
  ```python
  new_classifier = nn.Sequential(
      nn.Linear(in_features, 1024),          # Projection vers un espace de dimension supérieure
      nn.GELU(),                            # Activation GELU (plus lisse que ReLU)
      nn.BatchNorm1d(1024),                  # Normalisation par batch pour stabilité
      nn.Dropout(0.40),                     # Dropout modéré (40%)
      nn.Linear(1024, 768),                  # Réduction de dimensionnalité
      nn.GELU(),                            # Seconde activation GELU
      nn.BatchNorm1d(768),                  # Seconde normalisation
      nn.Dropout(0.40),                     # Dropout plus fort (40%) 
      nn.Linear(768, num_classes)           # Couche de sortie (logits)
  )
  ```

* **Extracteur d'embedding dédié**:
  ```python
  # Création d'un extracteur jusqu'à la couche Linear(1024→768)
  self.embedding_extractor = nn.Sequential(*list(new_classifier.children())[:5])
  ```
  Permet d'obtenir des représentations vectorielles des images pour d'autres tâches.

* **Optimisation avancée**:
  * Optimiseur **AdamW** avec régularisation différenciée:
    ```python
    # Pas de weight decay sur biais et couches de normalisation
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in self.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in self.named_parameters() 
                    if any(nd in n for nd in no_decay)], 
         "weight_decay": 0.0},
    ]
    ```
  
  * Scheduler `ReduceLROnPlateau` configuré pour:
    ```python
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',                              # Minimisation de la perte
        factor=self.config["reduce_lr_factor"],  # Facteur de réduction (0.4)
        patience=self.config["reduce_lr_patience"], # Attente avant réduction (1)
        min_lr=self.config["min_lr"],            # LR plancher (5e-6)
        verbose=True                            # Affichage des changements
    )
    ```

* **Métriques de suivi**:
  * `Accuracy` classique (précision)
  * `F1Score` en mode pondéré (weighted) pour tenir compte des déséquilibres de classes
  * Suivi séparé pour entraînement et validation

---

### Entraînement (`training_step`, `validation_step`)

L'entraînement est orchestré par PyTorch Lightning avec des étapes structurées:

#### 🔹 `training_step`:
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)                                 # Forward pass
    
    # Cross-entropy avec label smoothing (0.05) et poids des classes
    loss = nn.functional.cross_entropy(
        y_hat, y, 
        weight=self.class_weights,                  # Poids calculés par classe (équilibrage)
        label_smoothing=0.05                        # Label smoothing (évite surconfiance)
    )
    
    # Mise à jour et logging des métriques
    self.train_acc.update(y_hat, y)
    self.train_f1.update(y_hat, y)
    
    # Logging avec PyTorch Lightning
    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log('train_acc', self.train_acc.compute(), on_step=True, on_epoch=False)
    self.log('train_f1', self.train_f1.compute(), on_step=True, on_epoch=False)
    
    return loss
```

#### 🔹 `validation_step`:
Similaire au `training_step` mais utilisé pour l'évaluation sur le dataset de validation.

#### 🔹 `on_train_epoch_end` et `on_validation_epoch_end`:
Calcul et reset des métriques à la fin de chaque époque, permettant un suivi précis des performances.

#### Gestion des classes déséquilibrées:
Le modèle calcule et utilise des poids de classe basés sur la distribution des étiquettes:
```python
class_weights = compute_class_weight(
    class_weight="balanced",                 # Mode équilibré (inversement proportionnel à la fréquence)
    classes=np.unique(y_train_enc),          # Liste des classes uniques
    y=y_train_enc                            # Distribution réelle dans le train set
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
```

#### Label Smoothing:
Le label smoothing (0.05) évite que le modèle ne devienne trop confiant dans ses prédictions,
ce qui améliore la généralisation:
```python
# Au lieu de [0,0,1,0,0] (one-hot parfait)
# On obtient [0.01,0.01,0.95,0.01,0.01] (distribue un peu de proba aux autres classes)
```

---

### Classe `ImageClassifier`

Interface haut niveau pour orchestrer tout :

#### 🔹 `fit()`

* Encode les labels avec `LabelEncoder`
* Calcule les poids des classes pour gérer les déséquilibres
* Crée les datasets PyTorch (`ImageDataset`)
* Initialise le modèle `LitImageModel`
* Configure :

  * `EarlyStopping` sur le `val_f1` (patience=2, min_delta=0.01)
  * `ModelCheckpoint` (sauvegarde du meilleur modèle)
  * `CSVLogger` pour les métriques
* Lance `trainer.fit()` avec précision 16-bit pour accélération
* Recharge automatiquement le meilleur checkpoint à la fin

#### 🔹 `predict_with_tta()`

**Test-Time Augmentation (TTA)** améliore substantiellement la robustesse des prédictions lors de l'inférence:

```python
def predict_with_tta(self, x):
    """Test-Time Augmentation: moyenne des prédictions sur plusieurs variantes."""
    preds = []

    # Image originale
    preds.append(self.model(x))

    # Flip horizontal
    preds.append(self.model(torch.flip(x, dims=[3])))

    # Flip vertical
    preds.append(self.model(torch.flip(x, dims=[2])))

    # Rotation 180° (horizontal + vertical flip)
    preds.append(self.model(torch.flip(x, dims=[2, 3])))

    # Moyenne des prédictions
    return torch.mean(torch.stack(preds), dim=0)
```

Ce mécanisme:
* Applique 4 transformations différentes à chaque image de test
* Combine les prédictions par moyenne (réduit la variance)
* Améliore typiquement le F1-score de 1-2% sans coût d'entraînement
* Ne nécessite que 4× plus de calcul à l'inférence

Le TTA est particulièrement efficace pour les classes difficiles ou peu représentées,
en offrant une "vue d'ensemble" plus complète au modèle.

#### 🔹 `evaluate()`

* Prédit les classes sur un dataset de validation avec TTA
* Affiche et sauvegarde un rapport de classification (`classification_report` de sklearn)
* Retourne le rapport sous forme texte

#### 🔹 `save()` / `load()`

* Sauvegarde :

  * Poids du modèle
  * Encoder des labels (`LabelEncoder`)
* Chargement :

  * Recharge les poids et la configuration
  * Recompile le modèle complet en mémoire

### Optimisations techniques

Le modèle intègre plusieurs optimisations pour maximiser les performances:

#### 🔹 Entraînement en précision mixte (FP16)
```python
trainer = pl.Trainer(
    max_epochs=self.classifier_config["max_epochs"],
    callbacks=callbacks,
    logger=csv_logger,
    gradient_clip_val=1.0,            # Stabilité du gradient
    precision=16,                      # Entraînement en FP16
    accelerator="gpu"                  # Utilisation du GPU
)
```
La précision 16-bit permet:
* ~2× moins de mémoire GPU utilisée
* ~2× plus de vitesse d'entraînement
* Négligeable impact sur les performances finales

#### 🔹 Clipping de gradient
Le paramètre `gradient_clip_val=1.0` limite la norme du gradient pour éviter les explosions
et stabiliser l'entraînement, particulièrement utile avec des taux d'apprentissage élevés.

#### 🔹 Stratégie d'arrêt anticipé (EarlyStopping)
```python
EarlyStopping(
    monitor='val_f1',                 # Métrique surveillée 
    patience=self.classifier_config["patience"],  # Nombre d'époques sans amélioration (2)
    verbose=True,                     # Affichage des décisions
    mode='max',                       # Maximisation de la métrique
    min_delta=0.01                    # Amélioration minimale (1%)
)
```

#### 🔹 Parallélisation des chargements de données
```python
DataLoader(ds_train, batch_size=self.classifier_config["batch_size"], shuffle=True, num_workers=4)
```
L'utilisation de `num_workers=4` permet de paralléliser le chargement des données, améliorant sensiblement la vitesse d'entraînement en préparant les batches pendant que le GPU calcule.

#### 🔹 Suivi des métriques avec CSVLogger
Les métriques d'entraînement (train_loss, train_acc, train_f1, val_loss, val_acc, val_f1) sont sauvegardées à chaque époque dans un fichier CSV pour analyse ultérieure et visualisation des courbes d'apprentissage.

---

### Récapitulatif des hyperparamètres actuels

| Paramètre           | Valeur               | Description                                    |
|---------------------|----------------------|------------------------------------------------|
| `image_size`        | (384, 384)           | Dimensions des images d'entrée                 |
| `batch_size`        | 32                   | Taille des batches                             |
| `lr`                | 1e-4                 | Taux d'apprentissage initial                   |
| `max_epochs`        | 5                    | Nombre maximal d'époques                       |
| `weight_decay`      | 3e-4                 | Régularisation L2                              |
| `unfreeze_layers`   | 6                    | Nombre de blocs EfficientNet dégelés           |
| `model_name`        | efficientnet_b2      | Modèle backbone utilisé                        |
| `patience`          | 2                    | Patience pour EarlyStopping                    |
| `reduce_lr_factor`  | 0.4                  | Facteur de réduction du LR                     |
| `reduce_lr_patience`| 1                    | Patience avant réduction du LR                 |
| `min_lr`            | 5e-6                 | Taux d'apprentissage minimal                   |

---
![newplot](https://github.com/user-attachments/assets/bb570cb6-7012-45f4-9bb2-95a27298dc3d)
