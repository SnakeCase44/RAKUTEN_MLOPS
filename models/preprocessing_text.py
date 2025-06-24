import re
import ftfy
import numpy as np
import nltk
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.corpus import stopwords


# Téléchargement des stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

SUPPORTED_LANGS = [
    'it', 'fr', 'de', 'ro', 'en', 'sk', 'ca', 'nl', 'es', 'no', 'pl', 'pt', 'id',
    'et', 'hr', 'sv', 'da', 'sl', 'fi', 'cs', 'lv', 'lt', 'tr', 'hu', 'vi'
]
EXCLUDED_LANGS = ['tl', 'af', 'sq', 'cy', 'so', 'sw']

NLTK_LANG_MAP = {
    'fr': 'french', 'en': 'english', 'de': 'german', 'es': 'spanish', 'it': 'italian',
    'pt': 'portuguese', 'nl': 'dutch', 'no': 'norwegian', 'da': 'danish', 'sv': 'swedish',
    'fi': 'finnish', 'ro': 'romanian', 'hu': 'hungarian', 'cs': 'czech', 'pl': 'polish'
}

manual_blacklist = [
    "couleur", "haute", "qualité", "100", "dimensions", "facile", "matériel", "tout",
    "produit", "sans", "bois", "acier", "eacute", "type", "comme", "strong", "taille",
    "plus", "peut", "être", "caractéristiques", "plaît", "lumière", "cette", "style",
    "comprend", "très", "poids", "led", "hauteur", "blanc", "inclus", "main", "ans",
    "protection", "description", "amp", "neuf", "raison", "marque", "temps", "mesure",
    "anti", "utiliser", "sol", "design", "longueur", "contenu", "paquet", "plastique",
    "forme", "emballage", "forfait", "kit", "durable", "bleu", "div", "environ",
    "cadeau", "avant", "noël", "couleurs", "tous", "polyester", "utilisation",
    "pouvez", "hors", "pvc", "noir", "modèle", "coton", "matériau", "lot", "système",
    "sécurité", "espace", "fonction", "intérieur", "surface", "grand", "structure",
    "grande", "parfait", "permettre", "fait", "diamètre", "différence", "vie",
    "accessoires", "charge", "sable", "facilement", "matière", "1pc", "usb", "pouces",
    "doux", "span", "confortable", "faire", "sous", "requise", "élégant", "tendance", "nouveau",
    "meilleur", "populaire", "top", "offre", "taille", "mesure", "long", "large", "épais", "épaisseur",
    "volume", "http", "www", "html", "jpeg", "png", "jpg", "livraison", "rapide", "garantie", "retour", "contact"
]

DEFAULT_CLEANING_CONFIG = {
    "fix_encoding": True,
    "remove_html": True,
    "normalize_spaces": True,
    "replace_commas": False,
    "truncate_length": None,
    "remove_short_words": False,
    "min_word_length": 3,
    "max_words": 20,
    "detect_language": False,
    "filter_exotic_languages": False,
    "remove_stopwords": True,
    "remove_punct": False,
    "normalize_numbers": False,
    "remove_units": False,
    "remove_blacklist": False
}

def remove_units(text):
    return re.sub(
        r"\b\d+(?:[\.,]\d+)?\s?(kg|g|cm|mm|m²|m3|v|hz|w|°c|cl|l|ml|pcs|x\d*|x|xl|xxl|s|m|l)\b",
        "", text, flags=re.IGNORECASE
    )

def clean_text(text, config=None):
    if config is None:
        config = DEFAULT_CLEANING_CONFIG

    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""

    if not isinstance(text, str):
        text = str(text)

    lang = None
    if config.get("detect_language", True):
        try:
            lang = detect(text)
        except:
            return ""
        if config.get("filter_exotic_languages", True):
            if lang in EXCLUDED_LANGS or lang not in SUPPORTED_LANGS:
                return ""

    if config.get("fix_encoding", True):
        text = ftfy.fix_text(text)

    if config.get("remove_html", True):
        text = BeautifulSoup(text, "html.parser").get_text()

    if config.get("normalize_spaces", True):
        text = re.sub(r'\s+', ' ', text)

    if config.get("replace_commas", True):
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)

    if config.get("remove_punct", False):
        text = re.sub(r"[^\w\s]", " ", text)

    if config.get("normalize_numbers", False):
        text = re.sub(r"\b\d+\b", "<NUM>", text)

    trunc_len = config.get("truncate_length")
    if trunc_len and len(text) > trunc_len:
        text = text[:trunc_len]

    if config.get("remove_short_words", True):
        min_len = config.get("min_word_length", 3)
        text = ' '.join([w for w in text.split() if len(w) >= min_len])

    if config.get("remove_stopwords", True) and lang in NLTK_LANG_MAP:
        try:
            stop_words = set(stopwords.words(NLTK_LANG_MAP[lang]))
            text = ' '.join([w for w in text.split() if w.lower() not in stop_words])
        except:
            pass

    if config.get("remove_units", True):
        text = remove_units(text)

    if config.get("remove_blacklist", True):
        text = ' '.join([word for word in text.split() if word.lower() not in manual_blacklist])

    max_words = config.get("max_words")
    if max_words:
        words = text.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])
            last_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            text = truncated[:last_end+1] if last_end > 0 else truncated

    return text.strip()
