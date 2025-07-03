import os
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd
from models.multimodal_transformer_classifier.modelisation import DEFAULT_MULTIMODAL_CONFIG
from models.rakuten_efficientnet_image.modelisation import DEFAULT_IMAGE_CLASSIFIER_CONFIG
from models.rakuten_transformer_text.modelisation import TRANSFORMER_CONFIG
import argparse
import mlflow
import mlflow.pytorch
import pickle
from sklearn.metrics import classification_report
import json

# Import des modèles
from models.multimodal_transformer_classifier.modelisation import (
    MultimodalTrainer,
    load_image_model,
    load_text_model
)

from config import (
    SPLIT_DIR,
    MULTIMODAL_MODEL_PATH,
    MULTIMODAL_REPORT_PATH,
    EFFICIENTNET_IMAGE_MODEL_DIR,
    TRANSFORMER_MODEL_PATH
)

# Chemins des modèles
IMG_MODEL_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "model_model.pth"
IMG_META_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "model_meta.pkl"
TXT_MODEL_PATH = TRANSFORMER_MODEL_PATH / "model_model.pth"
TXT_META_PATH = TRANSFORMER_MODEL_PATH / "model_meta.pkl"


# === Chargement des données ===
def load_data(split_dir: Path):
    df_X_train = pd.read_csv(split_dir / "X_train_split.csv")
    df_X_val = pd.read_csv(split_dir / "X_test_split.csv")
    df_y_train = pd.read_csv(split_dir / "y_train_split.csv")
    df_y_val = pd.read_csv(split_dir / "y_test_split.csv")
    return df_X_train, df_X_val, df_y_train, df_y_val


def main():

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Late Fusion Multimodal")  

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=DEFAULT_MULTIMODAL_CONFIG["batch_size"])
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MULTIMODAL_CONFIG["max_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_MULTIMODAL_CONFIG["lr"])
    parser.add_argument("--patience", type=int, default=DEFAULT_MULTIMODAL_CONFIG["patience"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_MULTIMODAL_CONFIG["dropout"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_MULTIMODAL_CONFIG["weight_decay"])
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_MULTIMODAL_CONFIG["hidden_size"])
    parser.add_argument("--label_smoothing", type=float, default=DEFAULT_MULTIMODAL_CONFIG["label_smoothing"])
    

    args = parser.parse_args()

    with mlflow.start_run() as run:
            # Log des hyperparamètres
            mlflow.log_params({
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "lr": args.lr,
                "patience": args.patience,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "hidden_size": args.hidden_size,
                "label_smoothing": args.label_smoothing
            })

            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"Using batch_size: {args.batch_size}")
            print(f"Using max_epochs: {args.max_epochs}")
            print(f"Using lr: {args.lr}")
            print(f"Using patience: {args.patience}")
            print(f"Using dropout: {args.dropout}")
            print(f"Using weight_decay: {args.weight_decay}")
            print(f"Using hidden_size: {args.hidden_size}")
            print(f"Using label_smoothing: {args.label_smoothing}")

            # Vérifier que les chemins existent
            os.makedirs(MULTIMODAL_MODEL_PATH, exist_ok=True)

            # Chargement des modèles pré-entraînés
            print("Chargement du modèle image...")
            model_img, img_label_enc = load_image_model(
                IMG_MODEL_PATH,
                IMG_META_PATH,
                DEFAULT_IMAGE_CLASSIFIER_CONFIG
            )

            print("Chargement du modèle texte...")
            model_txt, txt_label_enc = load_text_model(
                TXT_MODEL_PATH,
                TXT_META_PATH
            )

            # Chargement du tokenizer pour le texte
            tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_CONFIG["model_name"])

            # Chargement des données depuis les splits existants
            print(f"Chargement des données depuis {SPLIT_DIR}")
            df_X_train, df_X_val, df_y_train, df_y_val = load_data(SPLIT_DIR)

            # Extraction des labels
            y_train = df_y_train["prdtypecode"].values
            y_val = df_y_val["prdtypecode"].values

            # Création du dictionnaire de configuration à partir des arguments
            custom_config = {
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "lr": args.lr,
                "patience": args.patience,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "hidden_size": args.hidden_size,
                "label_smoothing": args.label_smoothing
            }

            # Utilisation de la configuration personnalisée avec MLflow
            trainer = MultimodalTrainer(
                model_save_path=MULTIMODAL_MODEL_PATH,
                config=custom_config,
                mlflow_run_id=run.info.run_id  # Passer le run ID MLflow
            )

            # Entraînement
            print("Début de l'entraînement du modèle multimodal...")
            trainer.fit(
                df_train=df_X_train,
                df_val=df_X_val,
                y_train=y_train,
                y_val=y_val,
                img_model=model_img,
                txt_model=model_txt,
                tokenizer=tokenizer
            )

            # Évaluation
            print("Évaluation du modèle multimodal...")
            metrics = trainer.evaluate(
                df_test=df_X_val,
                y_test=y_val,
                tokenizer=tokenizer,
                report_path=MULTIMODAL_REPORT_PATH
            )                   

            print(f"Entraînement et évaluation terminés. Rapport sauvegardé dans {MULTIMODAL_REPORT_PATH}")

            # Log du modèle multimodal
            mlflow.pytorch.log_model(trainer.model, artifact_path="multimodal_model")

            # Log du label encoder
            label_path = MULTIMODAL_MODEL_PATH / "label_encoder.pkl"
            with open(label_path, "wb") as f:
                pickle.dump(trainer.label_enc, f)
            mlflow.log_artifact(str(label_path), artifact_path="artifacts")

            # Log du rapport texte
            mlflow.log_artifact(MULTIMODAL_REPORT_PATH, artifact_path="reports")

if __name__ == "__main__":
    main()