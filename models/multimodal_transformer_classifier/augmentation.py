"""
Module d'augmentation de données pour le modèle multimodal.
Applique des augmentations légères aux images.
"""

import cv2
import numpy as np
import random

# --------------------------------
# Augmentation d'image
# --------------------------------

def random_rotate(image, max_degrees=10):
    if image is None:
        return None
    height, width = image.shape[:2]
    angle = random.uniform(-max_degrees, max_degrees)
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def random_horizontal_flip(image, prob=0.5):
    if image is None:
        return None
    if random.random() < prob:
        return cv2.flip(image, 1)
    return image

def random_contrast_brightness(image, brightness_range=(0.95, 1.05), contrast_range=(0.95, 1.05)):
    if image is None:
        return None
    alpha = random.uniform(*contrast_range)
    beta = int(255 * (random.uniform(*brightness_range) - 1.0))
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def augment_image(image, config=None):
    if image is None or config is None or not config.get("enabled", False):
        return image
    result = image.copy()
    if config.get("random_rotate", False):
        max_degrees = config.get("rotate_degrees", 10)
        result = random_rotate(result, max_degrees)
    if config.get("random_flip", False):
        result = random_horizontal_flip(result)
    if config.get("random_contrast", False):
        result = random_contrast_brightness(result)
    return result


# --------------------------------
# Configuration des augmentations
# --------------------------------

DEFAULT_IMAGE_AUGMENTATION_CONFIG = {
    "enabled": True,
    "random_flip": True,
    "random_rotate": True,
    "random_contrast": True,
    "rotate_degrees": 15
}


def get_cleaning_config():
    return {
        "fix_encoding": True,
        "remove_html": True,
        "normalize_spaces": True,
        "replace_commas": False,
        "truncate_length": None,
        "remove_short_words": False,
        "min_word_length": 3,
        "max_words": 150,
        "detect_language": False,
        "filter_exotic_languages": False,
        "remove_stopwords": True,
        "remove_punct": False,
        "normalize_numbers": False,
        "remove_units": False,
        "remove_blacklist": False
    }
