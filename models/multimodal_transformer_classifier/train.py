import os
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd
from models.multimodal_transformer_classifier.modelisation import DEFAULT_MULTIMODAL_CONFIG
from models.rakuten_efficientnet_image.modelisation import DEFAULT_IMAGE_CLASSIFIER_CONFIG
from models.rakuten_transformer_text.modelisation import TRANSFORMER_CONFIG

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

    # Initialisation du trainer
    trainer = MultimodalTrainer(
        model_save_path=MULTIMODAL_MODEL_PATH,
        config=DEFAULT_MULTIMODAL_CONFIG
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
    trainer.evaluate(
        df_test=df_X_val,  # Utilisation des données de validation pour l'évaluation
        y_test=y_val,
        tokenizer=tokenizer,
        report_path=MULTIMODAL_REPORT_PATH
    )

    print(f"Entraînement et évaluation terminés. Rapport sauvegardé dans {MULTIMODAL_REPORT_PATH}")


if __name__ == "__main__":
    main()