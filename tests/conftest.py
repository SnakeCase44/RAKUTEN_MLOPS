import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from src.fastapi1.main import app


@pytest.fixture
def client():
    """Client de test FastAPI"""
    return TestClient(app)


@pytest.fixture
def test_image():
    """Crée une image de test temporaire"""
    # Créer une image RGB de test (100x100 pixels)
    img = Image.new('RGB', (100, 100), color='red')

    # Sauvegarder dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        img.save(tmp_file.name, 'JPEG')
        yield tmp_file.name

    # Nettoyer après le test
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def test_text():
    """Texte de test pour les prédictions"""
    return "jeu de plateau de société pour enfants"


@pytest.fixture
def invalid_file():
    """Fichier non-image pour tester la validation"""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp_file:
        tmp_file.write("Ce n'est pas une image")
        tmp_file.flush()
        yield tmp_file.name

    # Nettoyer après le test
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture(scope="session")
def test_data():
    """Données de test réutilisables"""
    return {
        "valid_texts": [
            "smartphone Samsung Galaxy",
            "livre de cuisine française",
            "jeu de plateau Monopoly",
            "robe d'été légère"
        ],
        "empty_texts": ["", "   ", "\n\t"],
        "expected_classes": ["40", "2920", "2280", "1140"]  # Classes possibles
    }