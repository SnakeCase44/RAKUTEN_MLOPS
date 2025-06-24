from config import SPLIT_DIR, TRANSFORMER_MODEL_PATH, TRANSFORMER_REPORT_PATH
from models.rakuten_transformer_text.modelisation import TextClassifier
import pandas as pd
from pathlib import Path


# === Chargement des données ===
def load_data(split_dir: Path):
    df_X_train = pd.read_csv(split_dir / "X_train_split.csv")
    df_X_val = pd.read_csv(split_dir / "X_test_split.csv")
    df_y_train = pd.read_csv(split_dir / "y_train_split.csv")
    df_y_val = pd.read_csv(split_dir / "y_test_split.csv")
    return df_X_train, df_X_val, df_y_train, df_y_val


# === Préparation des champs texte ===
def prepare_text_fields(df_X_train, df_X_val, df_y_train, df_y_val):
    X_train_texts = (df_X_train["designation"].fillna("") + " " + df_X_train["description"].fillna("")).tolist()
    X_val_texts = (df_X_val["designation"].fillna("") + " " + df_X_val["description"].fillna("")).tolist()
    y_train_labels = df_y_train["prdtypecode"].tolist()
    y_val_labels = df_y_val["prdtypecode"].tolist()
    return X_train_texts, X_val_texts, y_train_labels, y_val_labels


# === Configuration du nettoyage ===
def get_cleaning_config():
    return {
        "fix_encoding": True,
        "remove_html": True,
        "normalize_spaces": True,
        "replace_commas": False,
        "truncate_length": None,
        "remove_short_words": False,
        "min_word_length": 2,
        "max_words": 500,
        "detect_language": False,
        "filter_exotic_languages": False,
        "remove_stopwords": False,
        "remove_punct": False,
        "normalize_numbers": False,
        "remove_units": False,
        "remove_blacklist": False
    }


# === Fonction d'entraînement ===
def train_model(X_train_texts, X_val_texts, y_train_labels, y_val_labels, model_save_path, report_save_path):
    # Obtenir la configuration de nettoyage
    cleaning_config = get_cleaning_config()

    # Initialiser le classifier avec la configuration de nettoyage
    classifier = TextClassifier(model_save_path, cleaning_config=cleaning_config)

    # Entraînement avec les données de validation
    classifier.fit(X_train_texts, y_train_labels, X_val_texts, y_val_labels)

    # Évaluation sur validation
    classifier.evaluate(X_val_texts, y_val_labels, report_path=report_save_path)

    # Sauvegarde du modèle
    classifier.save(model_save_path)


# === Programme principal ===
def main():
    # Chargement des données
    print(f"Chargement des données depuis {SPLIT_DIR}")
    df_X_train, df_X_val, df_y_train, df_y_val = load_data(SPLIT_DIR)

    # Préparation des champs texte
    print("Préparation des champs texte")
    X_train_texts, X_val_texts, y_train_labels, y_val_labels = prepare_text_fields(
        df_X_train, df_X_val, df_y_train, df_y_val
    )

    print(f"Entraînement du modèle et sauvegarde dans {TRANSFORMER_MODEL_PATH}")
    # Entraînement et sauvegarde du modèle
    train_model(
        X_train_texts, X_val_texts, y_train_labels, y_val_labels,
        TRANSFORMER_MODEL_PATH, TRANSFORMER_REPORT_PATH
    )

    print(f"Le modèle a été entraîné et sauvegardé dans {TRANSFORMER_MODEL_PATH}")
    print(f"Rapport d'évaluation sauvegardé dans {TRANSFORMER_REPORT_PATH}")


if __name__ == "__main__":
    main()