from pathlib import Path
import tensorflow as tf

# === Répertoire racine du projet ===
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "dataset"

# === Dossier de split (utilisé pour tous les modèles) ===
SPLIT_DIR = DATA_DIR / "split"

# === Modèles ===
MODEL_DIR = PROJECT_ROOT / "models"


# === EFFICIENTNET Image Model ===
EFFICIENTNET_IMAGE_MODEL_DIR = MODEL_DIR / "rakuten_efficientnet_image" / "model"
EFFICIENTNET_IMAGE_MODEL_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "model_model.pth"
EFFICIENTNET_IMAGE_MODEL_META_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "model_meta.pkl"
EFFICIENTNET_IMAGE_REPORT_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "efficientnet_classification_report.txt"
EFFICIENTNET_IMAGE_METRICS_PATH = EFFICIENTNET_IMAGE_MODEL_DIR / "metrics" / "version_0" / "metrics.csv"
EFFICIENTNET_IMAGE_CONFUSION_DATA_PATH = EFFICIENTNET_IMAGE_MODEL_DIR /"metrics" / "version_0" / "confusion_data_image.csv"

# === Multimodal Model ===
MULTIMODAL_MODEL_DIR = MODEL_DIR / "multimodal_transformer_classifier"
MULTIMODAL_MODEL_PATH = MULTIMODAL_MODEL_DIR / "model"
MULTIMODAL_REPORT_PATH = MULTIMODAL_MODEL_DIR / "model" / "multimodal_classification_report.txt"
MULTIMODAL_METRICS_PATH = MULTIMODAL_MODEL_DIR / "model" / "metrics" / "version_12" / "metrics.csv"

# === Preprocessed Images ===
IMAGE_PREPROCESSED_DIR = DATA_DIR / "images" / "images_preprocessed"

# === Transformer Model ===
TRANSFORMER_MODEL_PATH = MODEL_DIR / "rakuten_transformer_text" / "model"
TRANSFORMER_REPORT_PATH = MODEL_DIR / "rakuten_transformer_text" / "model" / "transformer_classification_report.txt"
TRANSFORMER_METRICS_PATH = MODEL_DIR / "rakuten_transformer_text" / "model" /   "metrics" / "version_0" / "metrics.csv"
TRANSFORMER_CONFUSION_DATA_PATH = MODEL_DIR / "rakuten_transformer_text" / "model" /   "metrics" / "version_0" / "confusion_data.csv"

# === Données brutes ===
X_TRAIN_PATH = DATA_DIR / "X_train_update.csv"
Y_TRAIN_PATH = DATA_DIR / "Y_train_CVw08PX.csv"
X_TEST_PATH  = DATA_DIR / "X_test_update.csv"

# === Données filtrées ===
X_TRAIN_FILTERED = DATA_DIR / "X_train_filtered.csv"
X_TEST_FILTERED  = DATA_DIR / "X_test_filtered.csv"

# ======================================================================================
# Images
# ======================================================================================
IMAGE_TRAIN_DIR        = DATA_DIR / "images" / "image_train"
IMAGE_TEST_DIR         = DATA_DIR / "images" / "image_test"

# ======================================================================================
# Modèle et device
# ======================================================================================
DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
