import pickle
import os
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, F1Score
from torchvision import models
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from config import IMAGE_PREPROCESSED_DIR
from models.preprocessing_image import DEFAULT_IMAGE_PREPROCESSING_CONFIG

DEFAULT_IMAGE_CLASSIFIER_CONFIG = {
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

class ImageDataset(Dataset):
    """
    Dataset PyTorch pour la classification d’images dans le cadre du projet Rakuten.

    Ce dataset :
    - Charge les images à partir de chemins définis via `imageid` et `productid`
    - Applique des transformations d’augmentation (optionnel) ou de prétraitement standard
    - Normalise les images selon les statistiques d’ImageNet
    - Retourne un couple (image_tensor, label_tensor)

    Paramètres
    ----------
    df_X : pd.DataFrame
        DataFrame contenant les colonnes 'imageid' et 'productid' pour identifier chaque image.

    y_labels : array-like
        Liste ou tableau des labels (entiers) encodés pour la classification.

    transform_config : dict, optional
        Dictionnaire de configuration de prétraitement (non utilisé directement ici, placeholder pour évolution future).

    augment : bool, default=False
        Active les augmentations aléatoires sur les images (recadrage, flip, jitter, erasing).
    """

    def __init__(self, df_X, y_labels, transform_config=None, augment=False):
        self.df = df_X
        self.labels = y_labels
        self.config = transform_config or DEFAULT_IMAGE_PREPROCESSING_CONFIG
        self.augment = augment

        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])

        else:
            self.augmentation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

    def __len__(self):
        """
        Retourne le nombre total d’échantillons dans le dataset.
        Retourne
        --------
        int
            Longueur du dataset (nombre d’images).
        """

        return len(self.df)

    def __getitem__(self, idx):
        """
        Charge et retourne une image et son label à l’indice `idx`.

        Étapes :
        - Chargement de l’image à partir de son chemin
        - Conversion en RGB
        - Application des augmentations ou redimensionnement
        - Normalisation avec les stats d’ImageNet
        - Conversion du label en tensor

        Paramètres
        ----------
        idx : int
            Index de l’échantillon à récupérer.

        Retourne
        --------
        tuple (Tensor, Tensor)
            - image : tensor de forme (3, 384, 384) normalisé
            - label : entier encodé (Tensor)
        """

        row = self.df.iloc[idx]
        image_path = IMAGE_PREPROCESSED_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image manquante : {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augmentation(image)

        # Normalisation standard
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        label = torch.tensor(self.labels[idx])
        return image, label


class LitImageModel(pl.LightningModule):
    """
    Module PyTorch Lightning pour la classification d’images avec EfficientNet.
    Ce modèle utilise EfficientNet-B2 préentraîné comme backbone. Les couches sont gelées par défaut, avec possibilité de dégeler les derniers blocs pour le fine-tuning.
    Le classifieur est personnalisé (MLP) et remplace la tête d’origine d’EfficientNet. Il fournit aussi une méthode `get_embedding()` pour récupérer les représentations intermédiaires (utile en fusion multimodale).

    Paramètres
    ----------
    num_classes : int
        Nombre de classes cibles pour la classification.

    config : dict, optional
        Dictionnaire de configuration contenant notamment :
        - "model_name" : nom du backbone EfficientNet (doit être "efficientnet_b2" ici)
        - "unfreeze_layers" : nombre de blocs à dégeler
        - "lr", "weight_decay", etc.

    class_weights : torch.Tensor, optional
        Poids des classes à utiliser dans la loss pour gérer le déséquilibre.
    """

    def __init__(self, num_classes, config=None, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.config = config or DEFAULT_IMAGE_CLASSIFIER_CONFIG
        self.class_weights = class_weights

        # Chargement de l'EfficientNet
        if self.config["model_name"] == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=True)

            # Geler tous les paramètres
            for param in self.model.parameters():
                param.requires_grad = False

            # Dé-geler les derniers blocs
            total_blocks = len(self.model.features)
            layers_to_unfreeze = min(self.config["unfreeze_layers"], total_blocks)
            for i in range(total_blocks - layers_to_unfreeze, total_blocks):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True

            # Dé-geler le classifier
            for param in self.model.classifier.parameters():
                param.requires_grad = True

            # Remplacement du classifier EfficientNet
            in_features = self.model.classifier[1].in_features
            new_classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.GELU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.40),
                nn.Linear(1024, 768),
                nn.GELU(),
                nn.BatchNorm1d(768),
                nn.Dropout(0.40),
                nn.Linear(768, num_classes)
            )
            self.model.classifier[1] = new_classifier

            # Définir un extractor d'embedding propre
            self.embedding_extractor = nn.Sequential(*list(new_classifier.children())[:5])  # jusqu’à Linear(1024→768)

        else:
            raise ValueError(f"Modèle {self.config['model_name']} non supporté")

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, x):
        """
        Passe avant (inférence) standard.
        Paramètres
        ----------
        x : Tensor
            Image batch tensor de forme (B, 3, 384, 384)
        Retourne
        --------
        Tensor
            Logits pour chaque classe (forme : B x num_classes)
        """

        return self.model(x)

    def get_embedding(self, x):
        """
        Extrait un embedding de taille 768 depuis le backbone EfficientNet.
        Utile pour la fusion multimodale.
        Paramètres
        ----------
        x : Tensor
            Image batch tensor normalisé.

        Retourne
        --------
        Tensor
            Embedding tensor de forme (B, 768)
        """

        with torch.no_grad():
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.embedding_extractor(x)  # Linear(1280→1024) → GELU → BN → Dropout → Linear(1024→768)
        return x

    def training_step(self, batch, batch_idx):
        """
        Étape d’entraînement pour une batch : forward, calcul de la loss, mise à jour des métriques.
        Paramètres
        ----------
        batch : tuple (Tensor, Tensor)
            Tuple (image, label)

        batch_idx : int
            Indice du batch

        Retourne
        --------
        loss : Tensor
            Perte calculée pour le batch
        """

        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y, weight=self.class_weights, label_smoothing=0.05)
        self.train_acc.update(y_hat, y)
        self.train_f1.update(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc.compute(), on_step=True, on_epoch=False)
        self.log('train_f1', self.train_f1.compute(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Étape de validation : forward + métriques.
        Paramètres
        ----------
        batch : tuple
            Tuple (image, label)

        batch_idx : int
            Indice du batch

        Retourne
        --------
        loss : Tensor
            Perte sur le batch de validation
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y, weight=self.class_weights, label_smoothing=0.05)
        self.val_acc.update(y_hat, y)
        self.val_f1.update(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """
        Fin d’une époque d'entraînement : log des métriques agrégées, reset des compteurs.
        """
        self.log('train_acc', self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_epoch=True, prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        """
        Fin d’une époque de validation : log des métriques agrégées, reset des compteurs.
        """
        self.log('val_acc', self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """
        Définit l’optimiseur AdamW et un scheduler ReduceLROnPlateau basé sur val_f1.
        Retourne
        --------
        dict
            Optimiseur et scheduler à passer à PyTorch Lightning.
        """
        weight_decay = self.config.get("weight_decay", 8e-4)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["lr"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config["reduce_lr_factor"],
            patience=self.config["reduce_lr_patience"],
            min_lr=self.config["min_lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "mode": "max",
                "interval": "epoch",
                "frequency": 1
            }
        }


class ImageClassifier:
    """
    Classifieur d’images basé sur PyTorch Lightning et EfficientNet.

    Cette classe encapsule tout le pipeline d’apprentissage supervisé :
    - Entraînement (`fit`)
    - Évaluation (`evaluate`)
    - Inférence avec Test-Time Augmentation (`predict_with_tta`)
    - Sauvegarde/chargement du modèle

    Elle utilise la classe `LitImageModel` pour l’architecture et la classe `ImageDataset` pour les données.

    Paramètres
    ----------
    model_dir : str or Path
        Répertoire de sauvegarde du modèle et des métadonnées.

    preprocessing_config : dict, optional
        Configuration du prétraitement des images.

    classifier_config : dict, optional
        Configuration du modèle et de l'entraînement (learning rate, batch size, etc.).
    """
    def __init__(self, model_dir, preprocessing_config=None, classifier_config=None):
        self.model_dir = Path(model_dir)
        self.model = None
        self.label_enc = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessing_config = preprocessing_config or DEFAULT_IMAGE_PREPROCESSING_CONFIG
        self.classifier_config = classifier_config or DEFAULT_IMAGE_CLASSIFIER_CONFIG

    def fit(self, df_X_train, df_X_val, y_train, y_val):
        """
        Entraîne le modèle sur les données fournies avec un split entraînement/validation.
        Applique un oversampling implicite via le calcul des `class_weights` et active l’augmentation d’image côté entraînement.
        Paramètres
        ----------
        df_X_train : pd.DataFrame
            Données d’entraînement (doit contenir les colonnes `imageid` et `productid`).

        df_X_val : pd.DataFrame
            Données de validation.

        y_train : array-like
            Labels d’entraînement.

        y_val : array-like
            Labels de validation.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        y_train_enc = self.label_enc.fit_transform(y_train)
        y_val_enc = self.label_enc.transform(y_val)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_enc), y=y_train_enc)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        ds_train = ImageDataset(df_X_train, y_train_enc, transform_config=self.preprocessing_config, augment=True)
        ds_val = ImageDataset(df_X_val, y_val_enc, transform_config=self.preprocessing_config, augment=False)

        model = LitImageModel(num_classes=len(self.label_enc.classes_), config=self.classifier_config, class_weights=class_weights_tensor)
        csv_logger = CSVLogger(save_dir=self.model_dir, name="metrics")
        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                patience=self.classifier_config["patience"],
                verbose=True,
                mode='max',
                min_delta=0.01
            ),
            ModelCheckpoint(
                dirpath=os.path.join(self.model_dir, "checkpoints"),
                filename='{epoch}-{val_f1:.4f}',
                monitor='val_f1',
                mode='max',
                save_top_k=1,
                verbose=True
            )
        ]
        trainer = pl.Trainer(
            max_epochs=self.classifier_config["max_epochs"],
            callbacks=callbacks,
            logger=csv_logger,
            gradient_clip_val=1.0,
            precision=16,
            accelerator="gpu"
        )
        train_loader = DataLoader(ds_train, batch_size=self.classifier_config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(ds_val, batch_size=self.classifier_config["batch_size"], shuffle=False, num_workers=4)
        trainer.fit(model, train_loader, val_loader)

        best_ckpt = callbacks[1].best_model_path
        if best_ckpt:
            print(f"Chargement du meilleur modèle : {best_ckpt}")
            self.model = LitImageModel.load_from_checkpoint(best_ckpt, num_classes=len(self.label_enc.classes_), config=self.classifier_config, class_weights=class_weights_tensor).to(self.device)
        else:
            self.model = model.to(self.device)

        self.save()

    def predict_with_tta(self, x):
        """
        Réalise une prédiction avec Test-Time Augmentation (TTA).
        L’image est testée sous différentes transformations (flip, rotation),
        et les prédictions sont moyennées.
        Paramètres
        ----------
        x : torch.Tensor
            Batch d’images à prédire (forme : B, 3, 384, 384).

        Retourne
        --------
        torch.Tensor
            Logits moyens issus des différentes variantes.
        """

        """Test-Time Augmentation: retourne la moyenne des prédictions sur plusieurs variantes de l'image."""
        preds = []

        # Original
        preds.append(self.model(x))

        # Horizontal flip
        preds.append(self.model(torch.flip(x, dims=[3])))

        # Vertical flip
        preds.append(self.model(torch.flip(x, dims=[2])))

        # Rotation 180° (horizontal + vertical flip)
        preds.append(self.model(torch.flip(x, dims=[2, 3])))

        return torch.mean(torch.stack(preds), dim=0)

    def evaluate(self, df_X_val, y_val, report_path=None):
        """
        Évalue le modèle sur un jeu de validation en utilisant `predict_with_tta`.
        Paramètres
        ----------
        df_X_val : pd.DataFrame
            Données à évaluer (avec `imageid` et `productid`).

        y_val : array-like
            Labels réels (non encodés).

        report_path : str or Path, optional
            Si spécifié, sauvegarde le rapport de classification à ce chemin.
        Retourne
        --------
        str
            Rapport de classification scikit-learn (format texte).
        """

        if self.model is None or self.label_enc is None:
            raise ValueError("Le modèle ou l'encodeur de labels n'est pas initialisé.")
        y_val_enc = self.label_enc.transform(y_val)
        ds = ImageDataset(df_X_val, y_val_enc, transform_config=self.preprocessing_config)
        loader = DataLoader(ds, batch_size=self.classifier_config["batch_size"], num_workers=4)
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Évaluation (avec TTA)"):
                x = x.to(self.device)
                out = self.predict_with_tta(x)
                preds += torch.argmax(out, dim=1).cpu().tolist()
                labels += y.tolist()

        y_true = self.label_enc.inverse_transform(labels)
        y_pred = self.label_enc.inverse_transform(preds)
        report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)

        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)

        print("\n=== Rapport d'évaluation ===")
        print(report_str)
        return report_str

    def save(self):
        """
        Sauvegarde le modèle entraîné (`.pth`) et l'encodeur de labels (`.pkl`) dans `model_dir`.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / "model_model.pth")
        with open(self.model_dir / "model_meta.pkl", "wb") as f:
            pickle.dump({"label_enc": self.label_enc}, f)
        print(f"Modèle sauvegardé dans : {self.model_dir}")

    def load(self):
        """
        Charge le modèle entraîné et l’encodeur de labels à partir de `model_dir`.
        """
        with open(self.model_dir / "model_meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.label_enc = meta["label_enc"]
        num_classes = len(self.label_enc.classes_)
        self.model = LitImageModel(num_classes=num_classes, config=self.classifier_config)
        self.model.load_state_dict(torch.load(self.model_dir / "model_model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()