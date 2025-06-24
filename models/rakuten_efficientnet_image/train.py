from config import SPLIT_DIR, EFFICIENTNET_IMAGE_MODEL_PATH, EFFICIENTNET_IMAGE_REPORT_PATH
from models.rakuten_efficientnet_image.modelisation import ImageClassifier
import pandas as pd
from pathlib import Path
from models.preprocessing_image import pretraiter_dataset


def load_data(split_dir: Path):
    df_X_train = pd.read_csv(split_dir / "X_train_split.csv")
    df_X_val = pd.read_csv(split_dir / "X_test_split.csv")
    df_y_train = pd.read_csv(split_dir / "y_train_split.csv")
    df_y_val = pd.read_csv(split_dir / "y_test_split.csv")
    return df_X_train, df_X_val, df_y_train, df_y_val

def prepare_labels(df_y_train, df_y_val):
    y_train_labels = df_y_train["prdtypecode"].tolist()
    y_val_labels = df_y_val["prdtypecode"].tolist()
    return y_train_labels, y_val_labels

def get_train_preprocessing_config():
    return {
        "nettoyage_bords": False,
        "taille_cible": (384, 384),
        "preserve_aspect": False,
        "padding_color": [255, 255, 255],
        "normalize_01": True,
        "sauvegarder": False,
        "supprimer_doublons": False,
        "ameliorer_contrast": False,
        "contraste_methode": "clahe",
        "reduire_le_bruit": False,
        "bruit_methode": "bilateral",
        "bruit_taille": 5
    }

def train_model(df_X_train, df_X_val, y_train_labels, y_val_labels, model_dir, report_path, preprocessing_config):
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    classifier = ImageClassifier(model_dir=model_dir, preprocessing_config=preprocessing_config)
    classifier.fit(df_X_train, df_X_val, y_train_labels, y_val_labels)
    classifier.evaluate(df_X_val, y_val_labels, report_path)
    classifier.save()
    return classifier

def main():
    print(f"Chargement des données depuis {SPLIT_DIR}")

    df_X_train, df_X_val, df_y_train, df_y_val = load_data(SPLIT_DIR)

    print("Prétraitement des images d'entraînement")
    pretraiter_dataset(df_X_train, config=get_train_preprocessing_config(), is_train=True)

    print("Prétraitement des images de validation")
    pretraiter_dataset(df_X_val, config=get_train_preprocessing_config(), is_train=False)

    print("Préparation des labels")
    y_train_labels, y_val_labels = prepare_labels(df_y_train, df_y_val)

    model_dir = EFFICIENTNET_IMAGE_MODEL_PATH.parent
    print(f"Entraînement du modèle image dans {model_dir}")

    classifier = train_model(
        df_X_train, df_X_val, y_train_labels, y_val_labels,
        model_dir, EFFICIENTNET_IMAGE_REPORT_PATH,
        get_train_preprocessing_config()
    )

    print(f"Modèle sauvegardé dans {model_dir}")
    print(f"Rapport d'évaluation sauvegardé dans {EFFICIENTNET_IMAGE_REPORT_PATH}")

if __name__ == "__main__":
    main()

