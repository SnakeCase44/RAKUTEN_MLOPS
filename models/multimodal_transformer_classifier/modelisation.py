import os
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from models.rakuten_efficientnet_image.modelisation import LitImageModel
from models.preprocessing_text import clean_text
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torchvision.transforms as T
import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger


# Import des modules de prétraitement et d'augmentation
from models.multimodal_transformer_classifier.augmentation import DEFAULT_IMAGE_AUGMENTATION_CONFIG
from models.rakuten_transformer_text.modelisation import TransformerModel
from models.multimodal_transformer_classifier.augmentation import (
    augment_image, get_cleaning_config
)

from config import IMAGE_PREPROCESSED_DIR


# Configuration par défaut pour le modèle multimodal
DEFAULT_MULTIMODAL_CONFIG = {
    "batch_size": 48,         # Nombre d'échantillons par batch
    "max_epochs": 15,         # Nombre maximal d'epochs (early stopping peut arrêter avant)
    "lr": 5e-6,               # Learning rate de départ (sera ajusté avec ReduceLROnPlateau)
    "patience": 2,            # Nombre d'epochs sans amélioration avant action du scheduler
    "dropout": 0.4,           # Dropout appliqué dans le classifieur (pour éviter l'overfitting)
    "weight_decay": 0.01,     # Régularisation L2 (évite que les poids deviennent trop grands)
    "hidden_size": 512,       # Taille des couches cachées dans le classifieur
    "label_smoothing": 0.15   # Applique un lissage sur les labels pour régulariser
}


class MultimodalClassifier(pl.LightningModule):
    """
    Modèle de classification multimodale combinant caractéristiques d'image et de texte.

    Cette classe implémente un modèle PyTorch Lightning qui:
    1. Extrait des embeddings à partir de modèles pré-entraînés pour l'image et le texte
    2. Concatène ces embeddings pour créer une représentation multimodale
    3. Utilise un classifieur multicouche pour prédire la classe du produit

    Le modèle utilise:
    - Une architecture de fusion par concaténation plutôt que par pondération fixe
    - Un réseau de classification à 3 couches (1536→1024→512→classes)
    - Un dropout élevé (0.4 par défaut) pour prévenir le surapprentissage
    - Une optimisation via AdamW avec ReduceLROnPlateau
    - Un suivi de métriques F1 et accuracy pour l'entraînement et la validation

    Paramètres
    ----------
    model_img : torch.nn.Module
        Modèle d'encodage d'image (doit exposer une méthode `get_embedding()`)
    model_txt : torch.nn.Module
        Modèle d'encodage de texte (doit exposer une méthode `get_embedding()`)
    num_classes : int, default=27
        Nombre de classes cibles pour la classification
    lr : float, default=1e-4
        Taux d'apprentissage initial
    config : dict, optional
        Configuration des hyperparamètres (dropout, label_smoothing, etc.)
        Si None, utilise DEFAULT_MULTIMODAL_CONFIG

    Attributs
    ---------
    fusion_classifier : torch.nn.Sequential
        Réseau multicouche pour la classification sur embeddings concaténés
    train_acc, val_acc : torchmetrics.Accuracy
        Métriques d'accuracy pour l'entraînement et la validation
    train_f1, val_f1 : torchmetrics.F1Score
        Métriques F1 pondérées pour l'entraînement et la validation
    """
    def __init__(self, model_img, model_txt, num_classes=27, lr=1e-4, config=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model_img', 'model_txt'])
        self.config = config or DEFAULT_MULTIMODAL_CONFIG
        self.model_img = model_img
        self.model_txt = model_txt
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.get("label_smoothing", 0.15))

        self.img_output_dim = 768
        self.txt_output_dim = 768
        fusion_dim = self.img_output_dim + self.txt_output_dim

        hidden_size = self.config.get("hidden_size", 512)
        dropout_rate = self.config.get("dropout", 0.4)

        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

    def forward(self, image_tensor, input_ids, attention_mask):
        img_feat = self.model_img.get_embedding(image_tensor)  # [B, 768]
        txt_feat = self.model_txt.get_embedding(input_ids, attention_mask)  # [B, 768]
        fused_feat = torch.cat([img_feat, txt_feat], dim=1)  # [B, 1536]
        logits = self.fusion_classifier(fused_feat)
        return logits, {"fused_feat": fused_feat}

    def training_step(self, batch, batch_idx):
        image_tensor, input_ids, attention_mask, labels = batch
        logits, _ = self(image_tensor, input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.train_acc.update(logits, labels)
        self.train_f1.update(logits, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensor, input_ids, attention_mask, labels = batch
        logits, _ = self(image_tensor, input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.val_acc.update(logits, labels)
        self.val_f1.update(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc.compute(), on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_epoch=True, prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,  # taux d'apprentissage initial
            weight_decay=self.config.get("weight_decay", 0.01)  # régularisation L2
        )

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="max",  # maximise le F1 score
                factor=0.5,  # divise le LR par 2
                patience=1,  # attends 1 epoch sans amélioration
                threshold=0.01,  # nécessite au moins +1% d'amélioration
                threshold_mode="rel",  # interprète threshold comme relatif
                min_lr=1e-7,  # ne descend pas en dessous de ce LR
            ),
            "monitor": "val_f1",  # métrique surveillée par Lightning
            "interval": "epoch",  # mise à jour après chaque epoch
            "frequency": 1
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


class MultimodalDataset(Dataset):
    """
        Dataset PyTorch pour la classification multimodale texte + image.

        Chaque échantillon combine :
        - un texte (concaténation de `designation` et `description`, nettoyé puis tokenisé)
        - une image associée (chargée depuis un chemin formaté, transformée et normalisée)
        - un label (entier encodé avec LabelEncoder)

        Ce dataset applique optionnellement :
        - une augmentation de texte (e.g., synonymes, ajout de bruit)
        - une augmentation d'image (e.g., flip, rotation, brightness)

        Paramètres
        ----------
        df : pd.DataFrame
            DataFrame contenant les colonnes `designation`, `description`, `productid`, `imageid`.

        labels : array-like
            Liste ou tableau des labels encodés à associer aux lignes du DataFrame.

        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer compatible avec un modèle Transformer (ex. XLM-Roberta).

        max_len : int, default=128
            Longueur maximale de la séquence tokenisée.

        img_transform_config : dict, optional
            Dictionnaire de configuration pour activer/désactiver les augmentations d'image.

        text_augmentation_config : dict, optional
            Dictionnaire de configuration pour activer/désactiver les augmentations de texte.
        """

    def __init__(self, df, labels, tokenizer, max_len=128, img_transform_config=None):
        self.df = df
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.img_transform_config = img_transform_config or {}
        self.cv2 = cv2
        self.np = np
        self.os = os
        self.clean_text = clean_text  # Utilise l'import du module preprocessing_text
        self.augment_image = augment_image
        self.cleaning_config = get_cleaning_config()
        self.image_dir = IMAGE_PREPROCESSED_DIR

    def __len__(self):
        """
        Retourne le nombre d'échantillons dans le dataset.
        Retourne
        --------
        int
            Nombre de lignes dans le DataFrame source.
        """

        return len(self.df)

    def __getitem__(self, idx):
        """
        Récupère un échantillon multimodal à l’indice donné.

        Cette méthode applique :
        - le nettoyage et la tokenisation du texte (`designation` + `description`)
        - l’augmentation conditionnelle du texte (si activée)
        - le chargement et la normalisation de l’image (avec augmentation si activée)
        - la construction du tenseur image et des entrées texte pour le modèle Transformer
        Paramètres
        ----------
        idx : int
            Indice de l’échantillon dans le dataset
        Retourne
        --------
        tuple : (image_tensor, input_ids, attention_mask, label)
            - image_tensor : torch.Tensor (3, 384, 384)
            - input_ids : torch.Tensor (max_len,)
            - attention_mask : torch.Tensor (max_len,)
            - label : int
        """

        row = self.df.iloc[idx]

        # Traitement du texte
        designation = str(row.get("designation", "")).strip()
        description = str(row.get("description", "")).strip()
        text = f"{designation} {description}".strip()

        # Prétraitement du texte avec la configuration standard
        text_clean = self.clean_text(text, config=self.cleaning_config)

        # Tokenization pour le modèle transformer
        encodings = self.tokenizer(
            text_clean,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)

        # Chargement et augmentation de l'image
        try:
            imageid = row.get('imageid')
            productid = row.get('productid')

            # Construire le chemin de l'image prétraitée
            image_path = self.os.path.join(self.image_dir, f"image_{imageid}_product_{productid}.jpg")

            # Charger simplement l'image avec OpenCV
            image = self.cv2.imread(image_path)
            image = self.cv2.resize(image, (384, 384))

            if image is None:
                # Si l'image ne peut pas être chargée, utiliser une image noire
                raise ValueError(f"Image introuvable ou corrompue: {image_path}")

            # Appliquer les augmentations à l'image
            if self.img_transform_config.get("enabled", False):
                image = self.augment_image(image, config=self.img_transform_config)

        except Exception as e:
            # Gérer toutes les erreurs en créant une image noire de secours
            print(f"Erreur lors du chargement de l'image {imageid}_{productid}: {str(e)}")
            image = self.np.zeros((384, 384, 3), dtype=self.np.uint8)

        # Normalisation de l'image
        image = image.astype(self.np.float32) / 255.0
        image = self.np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image_tensor = torch.tensor(image, dtype=torch.float32)

        return image_tensor, input_ids, attention_mask, self.labels[idx]


class MultimodalTrainer:
    """
    Classe de gestion de l'entraînement, de l'évaluation et du chargement d’un modèle de classification multimodale
    combinant texte et image avec PyTorch Lightning.
    Cette classe encapsule toutes les étapes suivantes :
    - Préparation des datasets (texte + image)
    - Application d’augmentations conditionnelles
    - Entraînement d’un modèle `MultimodalClassifier` avec callbacks Lightning
    - Évaluation sur un jeu de test avec métriques détaillées
    - Sauvegarde et rechargement automatique du modèle et des métadonnées (label encoder, config)
    Attributs
    ---------
    model_save_path : str
        Répertoire où sont stockés les poids du modèle et les fichiers de configuration.
    config : dict
        Configuration des hyperparamètres du modèle multimodal (batch_size, patience, dropout, etc.).
    model : MultimodalClassifier
        Instance du modèle entraîné ou chargé.
    label_enc : sklearn.preprocessing.LabelEncoder
        Encodeur des classes cibles.
    device : torch.device
        Appareil utilisé pour l'entraînement ou l'inférence (CPU ou CUDA).
        """
    
    def __init__(self, model_save_path, config=None, mlflow_run_id=None):
        """
          Initialise l'entraîneur multimodal avec les chemins de sauvegarde et la configuration.
        Paramètres
        ----------
        model_save_path : str
            Répertoire de sauvegarde des poids du modèle et des métadonnées.
        config : dict, optional
            Dictionnaire de configuration pour les hyperparamètres du modèle multimodal.
            Si None, utilise la configuration par défaut (`DEFAULT_MULTIMODAL_CONFIG`).

        MODIFICATION: Ajout du paramètre mlflow_run_id
        """
        self.model_save_path = model_save_path
        self.config = config or DEFAULT_MULTIMODAL_CONFIG
        self.mlflow_run_id = mlflow_run_id  # NOUVEAU
        print(f"Using config: {self.config}")
        self.model = None
        self.label_enc = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df_train, df_val, y_train, y_val, img_model, txt_model, tokenizer):
        """
        Entraîne le modèle multimodal sur les données texte + image.
        Étapes réalisées :
        - Encodage des labels
        - Construction des datasets avec ou sans augmentations
        - Instanciation du modèle multimodal
        - Configuration de PyTorch Lightning avec callbacks
        - Entraînement avec suivi de validation
        - Chargement du meilleur checkpoint et sauvegarde finale
        Paramètres
        ----------
        df_train : pd.DataFrame
            Données d'entraînement contenant les colonnes `designation`, `description`, `productid`, `imageid`.
        df_val : pd.DataFrame
            Données de validation (même format que df_train).
        y_train : array-like
            Labels d'entraînement (format brut, non encodé).
        y_val : array-like
            Labels de validation (format brut, non encodé).
        img_model : torch.nn.Module
            Modèle d'encodage des images (doit exposer `get_embedding()`).
        txt_model : torch.nn.Module
            Modèle d'encodage des textes (doit exposer `get_embedding()`).
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer compatible avec le modèle texte (ex. XLM-Roberta).
            
        MODIFICATION: Remplacement du CSVLogger par MLFlowLogger
        """
        os.makedirs(self.model_save_path, exist_ok=True)

        # Encodage des labels
        self.label_enc = LabelEncoder()
        y_train_enc = self.label_enc.fit_transform(y_train)
        y_val_enc = self.label_enc.transform(y_val)

        # Images augmentation
        image_augmentation_config = DEFAULT_IMAGE_AUGMENTATION_CONFIG

        # Création des datasets
        train_dataset = MultimodalDataset(
            df_train,
            y_train_enc,
            tokenizer,
            max_len=128,
            img_transform_config=image_augmentation_config
        )

        val_dataset = MultimodalDataset(
            df_val,
            y_val_enc,
            tokenizer,
            max_len=128
        )

        # Création des dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        # Création du modèle
        self.model = MultimodalClassifier(
            model_img=img_model,
            model_txt=txt_model,
            num_classes=len(self.label_enc.classes_),
            lr=self.config["lr"],
            config=self.config
        )

        # ============= MODIFICATION PRINCIPALE =============
        # Configuration des loggers avec MLflow
        loggers = []
        
        # CSV Logger (gardé pour compatibilité)
        csv_logger = CSVLogger(save_dir=self.model_save_path, name="metrics")
        loggers.append(csv_logger)
        
        # MLflow Logger - NOUVEAU
        if self.mlflow_run_id:
            mlflow_logger = MLFlowLogger(
                experiment_name="Late Fusion Multimodal",
                run_id=self.mlflow_run_id,
                log_model=False
                
            )
            loggers.append(mlflow_logger)
        # ===============================================

        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                patience=self.config.get("patience", 2),
                verbose=True,
                mode='max',
                min_delta=0.05
            ),
            ModelCheckpoint(
                dirpath=os.path.join(self.model_save_path, "checkpoints"),
                filename='best_model_epoch_{epoch:02d}_f1_{val_f1:.4f}',
                monitor='val_f1',
                mode='max',
                save_top_k=1,
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]

        # ============= MODIFICATION =============
        # Entraînement avec les deux loggers
        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            callbacks=callbacks,
            logger=loggers,  # MODIFIÉ: utilise la liste de loggers
            log_every_n_steps=50,
            gradient_clip_val=1.0,
            default_root_dir=self.model_save_path
        )
        # ======================================

        trainer.fit(self.model, train_loader, val_loader)

        # Chargement du meilleur modèle
        best_ckpt = callbacks[1].best_model_path
        if best_ckpt:
            print(f"Chargement du meilleur modèle : {best_ckpt}")
            self.model = MultimodalClassifier.load_from_checkpoint(
                best_ckpt,
                model_img=img_model,
                model_txt=txt_model,
                num_classes=len(self.label_enc.classes_)
            ).to(self.device)

        self.save()

    def evaluate(self, df_test, y_test, tokenizer, report_path=None):
        """
              
        Évalue le modèle entraîné sur un jeu de test.
        Calcule les prédictions sur les données test, puis affiche et enregistre un rapport de classification.
        Paramètres
        ----------
        df_test : pd.DataFrame
            Données de test (même format que df_train).
        y_test : array-like
            Labels réels (non encodés) à comparer.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer utilisé pour le texte.
        report_path : str, optional
            Chemin d'enregistrement du rapport de classification. Si None, ne sauvegarde pas.
        Retourne
        --------
        str
            Rapport de classification (au format `sklearn.metrics.classification_report`)
            MODIFICATION: Ajout du logging MLflow pour les métriques d'évaluation
        """
        if self.model is None or self.label_enc is None:
            raise ValueError("Le modèle ou l'encodeur de labels n'est pas initialisé.")

        # Encodage des labels
        y_test_enc = self.label_enc.transform(y_test)

        # Création du dataset
        test_dataset = MultimodalDataset(
            df_test,
            y_test_enc,
            tokenizer,
            max_len=128
        )

        # Création du dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        # Évaluation
        self.model.eval()
        preds, refs = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Évaluation"):
                img, input_ids, attention_mask, labels = batch
                img = img.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits, _ = self.model(img, input_ids, attention_mask)
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                refs.extend(labels.cpu().tolist())

        # Conversion des indices en labels
        y_true = self.label_enc.inverse_transform(refs)
        y_pred = self.label_enc.inverse_transform(preds)

        # Génération du rapport
        report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
        report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)

        # ============= NOUVEAU: LOGGING MLFLOW =============
        if self.mlflow_run_id:
            # Log des métriques globales
            mlflow.log_metrics({
                "test_accuracy": report['accuracy'],
                "test_macro_f1": report['macro avg']['f1-score'],
                "test_weighted_f1": report['weighted avg']['f1-score'],
                "test_macro_precision": report['macro avg']['precision'],
                "test_macro_recall": report['macro avg']['recall']
            })
            
            # Log des métriques par classe (optionnel)
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    mlflow.log_metrics({
                        f"test_{class_name}_f1": metrics['f1-score'],
                        f"test_{class_name}_precision": metrics['precision'],
                        f"test_{class_name}_recall": metrics['recall']
                    })
        # =================================================

        # Écriture du rapport dans un fichier
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(report_str)

        print("\n=== Rapport de classification du modèle multimodal ===")
        print(report_str)

        return report_str

# =============================================================================
# 3. MODIFICATION DU FICHIER train.py (pour référence)
# =============================================================================

# Dans train.py, la ligne suivante doit être modifiée :
# AVANT :
# trainer = MultimodalTrainer(
#     model_save_path=MULTIMODAL_MODEL_PATH,
#     config=custom_config
# )

# APRÈS :
# trainer = MultimodalTrainer(
#     model_save_path=MULTIMODAL_MODEL_PATH,
#     config=custom_config,
#     mlflow_run_id=run.info.run_id  # Passer le run ID MLflow
# )

    def save(self):
        """
        Sauvegarde le modèle entraîné et ses métadonnées.

        - `model_model.pth` : poids du modèle multimodal
        - `model_meta.pkl` : dictionnaire contenant le label encoder et la configuration
        """
        os.makedirs(self.model_save_path, exist_ok=True)
        # Sauvegarde du modèle
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "model_model.pth"))

        # Sauvegarde des métadonnées
        with open(os.path.join(self.model_save_path, "model_meta.pkl"), "wb") as f:
            pickle.dump({"label_enc": self.label_enc, "config": self.config}, f)

        print(f"Modèle sauvegardé dans : {self.model_save_path}")

    def load(self, img_model, txt_model):
        """
        Recharge le modèle multimodal entraîné ainsi que ses métadonnées.

        Cette méthode restaure :
        - la configuration du modèle
        - l’encodeur des labels
        - les poids sauvegardés du modèle

        Paramètres
        ----------
        img_model : torch.nn.Module
            Modèle image utilisé lors de l'entraînement.
        txt_model : torch.nn.Module
            Modèle texte utilisé lors de l'entraînement.
        """
        """Charge le modèle et les métadonnées"""

        # Chargement des métadonnées
        with open(os.path.join(self.model_save_path, "model_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
            self.label_enc = meta["label_enc"]
            self.config = meta.get("config", self.config)

        # Chargement du modèle
        self.model = MultimodalClassifier(
            model_img=img_model,
            model_txt=txt_model,
            num_classes=len(self.label_enc.classes_),
            lr=self.config["lr"]
        )

        self.model.load_state_dict(
            torch.load(os.path.join(self.model_save_path, "model_model.pth"), map_location=self.device, weights_only=True)
        )

        self.model.to(self.device)
        self.model.eval()

def load_image_model(model_path, meta_path, classifier_config):
    """
    Charge un modèle d'encodage d'image (`LitImageModel`) avec ses poids et métadonnées.

    Paramètres
    ----------
    model_path : str
        Chemin vers le fichier `.pth` contenant les poids du modèle image.

    meta_path : str
        Chemin vers le fichier `.pkl` contenant les métadonnées (label encoder).

    classifier_config : dict
        Configuration utilisée pour initialiser le modèle image.

    Retourne
    --------
    model : torch.nn.Module
        Modèle image prêt pour l'inférence (`eval()` activé).

    label_enc : sklearn.preprocessing.LabelEncoder
        LabelEncoder utilisé lors de l'entraînement.
    """

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        label_enc = meta["label_enc"]
        num_classes = len(label_enc.classes_)

    model = LitImageModel(num_classes=num_classes, config=classifier_config)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle image: {str(e)}")
        print(f"Tentative de chargement avec strictness=False...")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, label_enc


def load_text_model(model_path, meta_path):
    """
    Charge un modèle de classification texte (`TransformerModel`) avec ses poids et métadonnées.

    Paramètres
    ----------
    model_path : str
        Chemin vers le fichier `.pth` contenant les poids du modèle texte.

    meta_path : str
        Chemin vers le fichier `.pkl` contenant les métadonnées (label encoder).

    Retourne
    --------
    model : torch.nn.Module
        Modèle texte prêt pour l'inférence (`eval()` activé).

    label_enc : sklearn.preprocessing.LabelEncoder
        LabelEncoder utilisé lors de l'entraînement.
    """

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        label_enc = meta["label_enc"]
        num_classes = len(label_enc.classes_)

    model = TransformerModel(num_classes=num_classes)

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle texte: {str(e)}")
        print(f"Tentative de chargement avec strictness=False...")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model, label_enc

def predict(self, text, image_path, tokenizer):
    """
    Prédit la classe d’un couple texte + image à l’aide du modèle multimodal chargé.

    Paramètres
    ----------
    text : str
        Description produit (désignation + description).

    image_path : str
        Chemin vers le fichier image (jpg, png...).

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associé au modèle texte.

    Retourne
    --------
    str
        Label prédit.
    """
    if self.model is None or self.label_enc is None:
        raise ValueError("Le modèle ou l'encodeur de labels n'est pas initialisé.")

    # Nettoyage et tokenisation du texte
    text_clean = clean_text(text)
    encoded = tokenizer(
        text_clean,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    input_ids = encoded["input_ids"].to(self.device)
    attention_mask = encoded["attention_mask"].to(self.device)

    # Chargement et prétraitement de l’image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {e}")
        image = Image.new("RGB", (384, 384))

    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(self.device)

    # Prédiction
    self.model.eval()
    with torch.no_grad():
        logits, _ = self.model(image_tensor, input_ids, attention_mask)
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = self.label_enc.inverse_transform([pred_idx])[0]

    return pred_label
