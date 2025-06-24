import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score
from torch.utils.data import DataLoader, TensorDataset
from models.preprocessing_text import clean_text
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

class TransformerDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour l’entraînement de modèles Transformers avec des textes et leurs labels associés.

    Ce dataset prépare à l’avance les tokenisations nécessaires via un `PreTrainedTokenizer` (HuggingFace),
    et fournit les entrées au format dict pour une utilisation directe dans un modèle Transformer.

    Paramètres
    ----------
    texts : list of str
        Liste des textes (pré-nettoyés ou bruts) à transformer en séquences.

    labels : list or array-like
        Liste des labels (entiers ou encodés) associés à chaque texte.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer HuggingFace, par exemple `AutoTokenizer.from_pretrained(...)`.

    max_len : int
        Longueur maximale des séquences tokenisées. Les textes plus courts sont paddés,
        les plus longs sont tronqués.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        """
        Retourne le nombre total d’échantillons dans le dataset.
        Retourne
        --------
        int
            Taille du dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retourne un échantillon indexé comprenant :
        - input_ids : séquence d’IDs du tokenizer
        - attention_mask : masque pour ignorer les paddings
        - labels : entier représentant la classe cible
        Paramètres
        ----------
        idx : int
            Index de l’échantillon à retourner.
        Retourne
        --------
        dict
            Un dictionnaire contenant les clés `input_ids`, `attention_mask` et `labels`.
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


class TransformerModel(pl.LightningModule):
    """
    Modèle de classification basé sur un Transformer préentraîné (ex : XLM-Roberta) avec fine-tuning partiel,
    implémenté avec PyTorch Lightning.

    Le modèle utilise le token [CLS] extrait de la dernière couche de l’encodeur transformer
    et le passe dans un classifieur profond à plusieurs couches pour effectuer une classification multiclasse.

    Paramètres
    ----------
    num_classes : int
        Nombre de classes cibles à prédire.

    model_name : str, optional
        Nom ou chemin du modèle transformer préentraîné à charger (par défaut : défini dans TRANSFORMER_CONFIG).

    lr : float, optional
        Taux d'apprentissage à utiliser (par défaut : défini dans TRANSFORMER_CONFIG).
    """
    def __init__(self, num_classes, model_name=None, lr=None):
        super().__init__()
        self.save_hyperparameters()
        config = TRANSFORMER_CONFIG

        model_name = model_name or config["model_name"]
        lr = lr or config["lr"]

        self.lr = lr
        self.transformer = AutoModel.from_pretrained(model_name)

        # Classifier optimisé pour 4 couches + batch 128
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.config.hidden_size, 512),
            torch.nn.GELU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.50),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.LayerNorm(256),
            torch.nn.Dropout(0.45),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(0.40),
            torch.nn.Linear(128, num_classes)
        )

        # Metrics for training and validation
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted')

        # Freezing transformer layers
        for p in self.transformer.parameters():
            p.requires_grad = False
        for i in range(1, config["unfreeze_layers"] + 1):
            for p in self.transformer.encoder.layer[-i].parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Effectue une passe avant sur le modèle.

        Paramètres
        ----------
        input_ids : torch.Tensor
            Tensor des IDs des tokens (shape : [batch_size, seq_len]).

        attention_mask : torch.Tensor
            Tensor du masque d'attention (shape : [batch_size, seq_len]).

        Retourne
        --------
        torch.Tensor
            Logits de prédiction (shape : [batch_size, num_classes]).
        """
        out = self.transformer(input_ids, attention_mask).last_hidden_state[:, 0]
        return self.classifier(out)

    def get_embedding(self, input_ids, attention_mask):
        """
        Extrait les embeddings (vecteur [CLS]) du modèle transformer sans les passer dans le classifieur.

        Retourne
        --------
        torch.Tensor
            Vecteurs d’embedding [CLS] (shape : [batch_size, hidden_dim]).
        """
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def training_step(self, batch, batch_idx):
        """
        Étape d'entraînement sur un batch. Calcule la perte et met à jour les métriques.

        Paramètres
        ----------
        batch : dict
            Batch contenant les `input_ids`, `attention_mask`, et `labels`.

        batch_idx : int
            Index du batch dans l'époque.

        Retourne
        --------
        torch.Tensor
            La valeur de la perte du batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        # Update metrics
        self.train_acc.update(logits, labels)
        self.train_f1.update(logits, labels)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', self.train_acc.compute(), on_step=True, on_epoch=False)
        self.log('train_f1', self.train_f1.compute(), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Étape de validation sur un batch. Calcule la perte et les métriques.

        Retourne
        --------
        torch.Tensor
            La valeur de la perte du batch de validation.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        # Update metrics
        self.val_acc.update(logits, labels)
        self.val_f1.update(logits, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """
        Log les métriques agrégées à la fin d’une époque d'entraînement et réinitialise les compteurs.
        """
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True)
        self.log('train_f1', train_f1, on_epoch=True, prog_bar=True)

        # Reset metrics for next epoch
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        """
        Log les métriques agrégées à la fin d’une époque de validation et réinitialise les compteurs.
        """
        # Calculer les métriques
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()

        # Convertir en float si c'est un tensor
        if torch.is_tensor(val_acc):
            val_acc_float = val_acc.item()
            val_f1_float = val_f1.item()
            print(f"DEBUG - Converted val_acc: {val_acc_float:.4f}")
            print(f"DEBUG - Converted val_f1: {val_f1_float:.4f}")

        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', val_f1, on_epoch=True, prog_bar=True)

        # Reset metrics for next epoch
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """
        Configure l'optimiseur AdamW et le scheduler ReduceLROnPlateau.

        Retourne
        --------
        dict
            Dictionnaire contenant l’optimiseur et le scheduler de learning rate.
        """
        weight_decay = TRANSFORMER_CONFIG.get("weight_decay", 0.025)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        reduce_scheduler = ReduceLROnPlateau(
            optimizer=opt,
            factor=0.1,
            patience=1,
            mode="min",
            threshold=0.015,
            threshold_mode="rel",
        )

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': reduce_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class TextClassifier:
    """
    Pipeline de classification de texte utilisant un modèle Transformer avec PyTorch Lightning.

    Cette classe encapsule l'entraînement, l'évaluation, la sauvegarde et le rechargement d’un modèle
    de classification multilingue basé sur un Transformer (ex : XLM-Roberta).

    Paramètres
    ----------
    model_save_path : str or Path
        Chemin vers le dossier de sauvegarde du modèle (poids, tokenizer, label encoder...).

    cleaning_config : dict, optional
        Dictionnaire de configuration pour le nettoyage des textes (via `clean_text`).
    """
    def __init__(self, model_save_path, cleaning_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_CONFIG["model_name"])
        self.label_enc = None
        self.model = None
        self.model_save_path = model_save_path
        self.cleaning_config = cleaning_config or {}

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        """
        Entraîne le modèle Transformer sur des textes et labels fournis.

        Paramètres
        ----------
        texts : list of str
            Textes d'entraînement bruts.

        labels : list
            Labels cibles associés aux textes.

        val_texts : list of str, optional
            Textes de validation.

        val_labels : list, optional
            Labels de validation.

        Notes
        -----
        Applique le nettoyage (`clean_text`) et encode les labels via `LabelEncoder`.
        Utilise un scheduler et l’early stopping basés sur la métrique `val_f1`.
        """
        print("Nettoyage des textes d'entraînement...\n")
        texts_clean = [clean_text(t, config=self.cleaning_config) for t in tqdm(texts, desc="Clean train")]

        self.label_enc = LabelEncoder()
        y_enc = self.label_enc.fit_transform(labels)

        train_dataset = TransformerDataset(texts_clean, y_enc, self.tokenizer, TRANSFORMER_CONFIG["max_len"])
        train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_CONFIG["batch_size"], shuffle=True,
                                  num_workers=4)

        val_loader = None
        if val_texts is not None and val_labels is not None:
            print("Nettoyage des textes de validation...")
            val_texts_clean = [clean_text(t, config=self.cleaning_config) for t in tqdm(val_texts, desc="Clean val")]
            val_labels_enc = self.label_enc.transform(val_labels)
            val_dataset = TransformerDataset(val_texts_clean, val_labels_enc, self.tokenizer,
                                             TRANSFORMER_CONFIG["max_len"])
            val_loader = DataLoader(val_dataset, batch_size=TRANSFORMER_CONFIG["batch_size"], shuffle=False,
                                    num_workers=4)

        self.model = TransformerModel(num_classes=len(self.label_enc.classes_), lr=TRANSFORMER_CONFIG["lr"])

        csv_logger = CSVLogger(save_dir=self.model_save_path, name="metrics")

        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                patience=TRANSFORMER_CONFIG["patience"],
                verbose=True,
                mode='max',
                min_delta=0.015
            ),
            ModelCheckpoint(
                dirpath=os.path.join(self.model_save_path, "checkpoints"),
                filename='{epoch}-{val_f1:.4f}',
                monitor='val_f1',
                mode='max',
                save_top_k=1,
                verbose=True,
            )
        ]

        trainer = pl.Trainer(
            max_epochs=TRANSFORMER_CONFIG["max_epochs"],
            callbacks=callbacks,
            logger=csv_logger,
            gradient_clip_val=1.0,
        )

        if val_loader:
            trainer.fit(self.model, train_loader, val_loader)
        else:
            trainer.fit(self.model, train_loader)

        best_ckpt = callbacks[1].best_model_path
        if best_ckpt:
            print(f"Chargement du meilleur modèle : {best_ckpt}")
            self.model = TransformerModel.load_from_checkpoint(
                best_ckpt,
                num_classes=len(self.label_enc.classes_)
            )

        self.save(self.model_save_path)

    def save(self, path):
        """
        Sauvegarde les poids du modèle, le tokenizer et le label encoder dans le dossier spécifié.

        Paramètres
        ----------
        path : str or Path
            Chemin du dossier de sauvegarde.
        """
        path_str = str(path)
        os.makedirs(path_str, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path_str, "model_model.pth"))
        self.tokenizer.save_pretrained(path_str)
        with open(os.path.join(path_str, "model_meta.pkl"), "wb") as f:
            pickle.dump({"label_enc": self.label_enc}, f)
        print(f"Modèle sauvegardé dans : {path_str}")

    def evaluate(self, texts, labels, report_path=None):
        """
        Évalue le modèle sur un jeu de données texte + labels.

        Paramètres
        ----------
        texts : list of str
            Textes à évaluer.

        labels : list
            Labels réels pour comparaison.

        report_path : str, optional
            Chemin pour enregistrer le rapport de classification (format texte).

        Retourne
        --------
        str
            Rapport de classification (precision, recall, f1, etc.).

        Raises
        ------
        ValueError
            Si le modèle ou le label encoder n'a pas été initialisé.
        """
        if self.model is None or self.label_enc is None:
            raise ValueError("Le modèle ou l'encodeur de labels n'est pas initialisé.")

        texts_clean = [clean_text(t, config=self.cleaning_config) for t in texts]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        loader = self._dataloader(texts_clean, self.label_enc.transform(labels))
        preds, y_refs = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Évaluation"):
                input_ids, attention_mask, labels_batch = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels_batch = labels_batch.to(device)
                out = self.model(input_ids, attention_mask)
                preds.extend(torch.argmax(out, dim=1).cpu().tolist())
                y_refs.extend(labels_batch.cpu().tolist())

        y_true_labels = self.label_enc.inverse_transform(y_refs)
        y_pred_labels = self.label_enc.inverse_transform(preds)

        report = classification_report(y_true_labels, y_pred_labels, digits=4, zero_division=0)

        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(report)

        print("\n=== Rapport d'évaluation ===")
        print(report)
        return report

    def _dataloader(self, texts, labels=None):
        """
        Crée un DataLoader à partir de textes (et labels si fournis).

        Paramètres
        ----------
        texts : list of str
            Textes à encoder et loader.

        labels : list or torch.Tensor, optional
            Labels à associer aux textes.

        Retourne
        --------
        DataLoader
            DataLoader prêt à l'emploi pour l'évaluation ou la prédiction.
        """
        enc = self.tokenizer(texts, truncation=True, padding='max_length', max_length=TRANSFORMER_CONFIG["max_len"],
                             return_tensors="pt")
        labels = torch.tensor(labels) if labels is not None else torch.zeros(len(texts)).long()
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
        return DataLoader(dataset, batch_size=TRANSFORMER_CONFIG["batch_size"], shuffle=False)