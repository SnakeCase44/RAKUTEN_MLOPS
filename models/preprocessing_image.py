import os
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from config import IMAGE_TRAIN_DIR, IMAGE_PREPROCESSED_DIR
from PIL import Image
from functools import partial

DEFAULT_IMAGE_PREPROCESSING_CONFIG = {
    "nettoyage_bords": True,
    "taille_cible": (500, 500),
    "preserve_aspect": True,
    "padding_color": [255, 255, 255],
    "ameliorer_contrast": False,
    "contraste_methode": "clahe",
    "reduire_le_bruit": False,
    "bruit_methode": "bilateral",
    "bruit_taille": 5,
    "supprimer_doublons": False,
    "normalize_01": False,
    "sauvegarder": False
}

def find_identical_images(df):
    print("Recherche d'images identiques...")
    signatures = {}
    duplicates = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = IMAGE_TRAIN_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"
        try:
            file_size = os.path.getsize(img_path)
            img = np.array(Image.open(img_path).convert("RGB"))
            pixel_sum = img.sum()
            signature = (file_size, img.shape[0], img.shape[1], pixel_sum)
            if signature in signatures:
                duplicates.append(idx)
            else:
                signatures[signature] = idx
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {img_path}: {e}")
    print(f"Nombre d'images dupliquées trouvées: {len(duplicates)}")
    return duplicates

def charger(chemin):
    image = cv2.imread(str(chemin))
    if image is None:
        print(f"[❌ ERREUR LECTURE] Impossible de lire : {chemin}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def sauvegarder(image, chemin):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(chemin), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def nettoyer_bords(image, seuil=0.95, padding=10):
    image_gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image_gris, int(seuil * 255), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    return image[y:y + h, x:x + w]

def redimensionner(image, largeur=None, hauteur=None, preserve_aspect=True, padding_color=[255, 255, 255]):
    h, w = image.shape[:2]
    if largeur is None and hauteur is None:
        return image
    if preserve_aspect and largeur and hauteur:
        ratio = min(largeur / w, hauteur / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = np.full((hauteur, largeur, 3), padding_color, dtype=np.uint8)
        y_offset = (hauteur - new_h) // 2
        x_offset = (largeur - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return result
    largeur = int(w * (hauteur / h)) if largeur is None else largeur
    hauteur = int(h * (largeur / w)) if hauteur is None else hauteur
    return cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_AREA)

def ameliorer_contraste(image, methode="clahe"):
    try:
        if methode == "clahe":
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        elif methode == "equalize":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    except Exception as e:
        print(f"[❌ CONTRASTE] Erreur avec {methode}: {e}")
    return image

def reduire_bruit(image, methode="bilateral", taille=5):
    if taille % 2 == 0:
        taille += 1
    try:
        if methode == "gaussian":
            return cv2.GaussianBlur(image, (taille, taille), 0)
        elif methode == "median":
            return cv2.medianBlur(image, taille)
        elif methode == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
    except Exception as e:
        print(f"[❌ BRUIT] Erreur méthode {methode}: {e}")
    return image


def pretraiter_image(image_path, image_output_path=None, config=None):
    # Charger l'image
    image = charger(image_path)
    if image is None:
        return None

    try:
        # Appliquer les transformations sur l'image (recadrage, amélioration du contraste, etc.)
        if config.get("nettoyage_bords"):
            image = nettoyer_bords(image)
        if config.get("ameliorer_contrast"):
            image = ameliorer_contraste(image, methode=config.get("contraste_methode", "clahe"))
        if config.get("reduire_le_bruit"):
            image = reduire_bruit(image, methode=config.get("bruit_methode", "bilateral"),
                                  taille=config.get("bruit_taille", 5))
        if config.get("taille_cible"):
            largeur, hauteur = config["taille_cible"]
            image = redimensionner(image, largeur, hauteur, config.get("preserve_aspect", True),
                                   config.get("padding_color", [255, 255, 255]))

        # Normalisation après toutes les autres transformations
        if config.get("normalize_01", True):
            image = image.astype(np.float32) / 255.0

        # Sauvegarder l'image si nécessaire
        if config.get("sauvegarder") and image_output_path:
            image_to_save = (image * 255).astype(np.uint8) if config.get("normalize_01") else image
            sauvegarder(image_to_save, image_output_path)
    except Exception as e:
        print(f"[❌ PRETRAITEMENT] {image_path} : {e}")
        return None

    return image



def process_chunk(chunk, dossier_output, config):
    return pretraiter_batch(chunk, dossier_output, config)

def pretraiter_batch(image_infos, dossier_output, config):
    resultats = []
    for imageid, productid in image_infos:
        chemin_source = IMAGE_TRAIN_DIR / f"image_{imageid}_product_{productid}.jpg"
        chemin_sortie = dossier_output / f"image_{imageid}_product_{productid}.jpg"
        if chemin_source.exists():
            # Appeler la fonction de prétraitement centralisée
            image = pretraiter_image(chemin_source, chemin_sortie, config)
            resultats.append((imageid, productid, image is not None))
        else:
            print(f"[❌ ABSENT] {chemin_source}")
            resultats.append((imageid, productid, False))
    return resultats


def pretraiter_dataset(df, config=None, batch_size=1000, n_jobs=None, is_train=True):
    if config is None:
        config = DEFAULT_IMAGE_PREPROCESSING_CONFIG.copy()

    dossier_output = IMAGE_PREPROCESSED_DIR
    os.makedirs(dossier_output, exist_ok=True)

    # Appliquer la suppression des doublons uniquement si is_train est True
    if is_train and config.get("supprimer_doublons"):
        doublons = find_identical_images(df)
        df = df.drop(doublons).reset_index(drop=True)
        print(f"Dataset après suppression des doublons : {len(df)} lignes")

    infos = list(zip(df["imageid"], df["productid"]))
    print(f"Nombre d'images à prétraiter : {len(infos)}")

    process_with_args = partial(process_chunk, dossier_output=dossier_output, config=config)

    resultats = process_map(
        process_with_args,
        [infos[i:i + batch_size] for i in range(0, len(infos), batch_size)],
        max_workers=n_jobs,
        chunksize=1,
        desc="Traitement en parallèle"
    )

    resultats = [item for sublist in resultats for item in sublist]
    nb_success = sum(1 for _, _, s in resultats if s)
    print(f"Images traitées avec succès : {nb_success}/{len(resultats)}")
    return resultats
