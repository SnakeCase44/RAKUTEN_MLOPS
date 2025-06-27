import pytest
import json
import tempfile
import os
from PIL import Image


class TestPredictionSuccess:
    """Tests de prédictions réussies"""

    def test_valid_prediction(self, client, test_image, test_text):
        """Test de prédiction avec données valides"""
        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": test_text},
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()

        # Vérifier la structure de la réponse
        assert data["success"] is True
        assert "predicted_class" in data
        assert "confidence" in data
        assert "input_text" in data
        assert "image_filename" in data
        assert "device_used" in data

        # Vérifier les types et valeurs
        assert isinstance(data["predicted_class"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["input_text"] == test_text
        assert data["image_filename"] == "test.jpg"

    def test_prediction_with_different_image_formats(self, client, test_text):
        """Test avec différents formats d'image"""
        formats = [
            ("test.jpg", "image/jpeg", "JPEG"),
            ("test.png", "image/png", "PNG")
        ]

        for filename, content_type, pil_format in formats:
            # Créer une image de test dans le format spécifié
            img = Image.new('RGB', (100, 100), color='blue')
            with tempfile.NamedTemporaryFile(suffix=f'.{pil_format.lower()}', delete=False) as tmp_file:
                img.save(tmp_file.name, pil_format)

                try:
                    with open(tmp_file.name, 'rb') as img_file:
                        response = client.post(
                            "/predict/multimodal",
                            data={"text": test_text},
                            files={"image": (filename, img_file, content_type)}
                        )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["image_filename"] == filename

                finally:
                    os.unlink(tmp_file.name)

    def test_prediction_with_various_texts(self, client, test_image, test_data):
        """Test avec différents types de textes"""
        for text in test_data["valid_texts"]:
            with open(test_image, 'rb') as img_file:
                response = client.post(
                    "/predict/multimodal",
                    data={"text": text},
                    files={"image": ("test.jpg", img_file, "image/jpeg")}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["input_text"] == text


class TestPredictionValidation:
    """Tests de validation des entrées"""

    def test_empty_text(self, client, test_image):
        """Test avec texte vide"""
        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": ""},
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        assert response.status_code == 400
        assert "ne peut pas être vide" in response.json()["detail"]

    def test_whitespace_only_text(self, client, test_image):
        """Test avec texte contenant seulement des espaces"""
        whitespace_texts = ["   ", "\n\t", "   \n\t   "]

        for text in whitespace_texts:
            with open(test_image, 'rb') as img_file:
                response = client.post(
                    "/predict/multimodal",
                    data={"text": text},
                    files={"image": ("test.jpg", img_file, "image/jpeg")}
                )

            assert response.status_code == 400
            assert "ne peut pas être vide" in response.json()["detail"]

    def test_invalid_file_type(self, client, invalid_file, test_text):
        """Test avec fichier non-image"""
        with open(invalid_file, 'rb') as txt_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": test_text},
                files={"image": ("test.txt", txt_file, "text/plain")}
            )

        assert response.status_code == 400
        assert "doit être une image" in response.json()["detail"]

    def test_missing_image(self, client, test_text):
        """Test sans fichier image"""
        response = client.post(
            "/predict/multimodal",
            data={"text": test_text}
        )

        assert response.status_code == 422  # Validation error
        error_detail = response.json()["detail"]
        assert any("image" in str(error).lower() for error in error_detail)

    def test_missing_text(self, client, test_image):
        """Test sans texte"""
        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        assert response.status_code == 422  # Validation error
        error_detail = response.json()["detail"]
        assert any("text" in str(error).lower() for error in error_detail)

    def test_very_long_text(self, client, test_image):
        """Test avec texte très long"""
        long_text = "produit très intéressant " * 100  # ~2500 caractères

        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": long_text},
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()

        # Le texte devrait être tronqué dans la réponse
        assert len(data["input_text"]) <= 103  # 100 chars + "..."
        assert data["input_text"].endswith("...")


class TestPredictionErrorHandling:
    """Tests de gestion d'erreurs spécifiques aux prédictions"""

    def test_wrong_http_method(self, client):
        """Test avec mauvaise méthode HTTP"""
        response = client.get("/predict/multimodal")  # POST attendu
        assert response.status_code == 405

    def test_corrupted_image_handling(self, client, test_text):
        """Test avec image corrompue"""
        # Créer un fichier qui semble être une image mais est corrompu
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"This is not a valid image file")
            tmp_file.flush()

            try:
                with open(tmp_file.name, 'rb') as corrupted_file:
                    response = client.post(
                        "/predict/multimodal",
                        data={"text": test_text},
                        files={"image": ("corrupted.jpg", corrupted_file, "image/jpeg")}
                    )

                # L'API devrait soit gérer gracieusement l'erreur (200 avec fallback)
                # soit retourner une erreur appropriée (400/500)
                assert response.status_code in [200, 400, 500]

                if response.status_code == 200:
                    # Si la prédiction réussit, vérifier la structure
                    data = response.json()
                    assert "success" in data
                    assert "predicted_class" in data

            finally:
                os.unlink(tmp_file.name)

    def test_large_image_handling(self, client, test_text):
        """Test avec image de grande taille"""
        # Créer une image plus grande (1000x1000)
        large_img = Image.new('RGB', (1000, 1000), color='green')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            large_img.save(tmp_file.name, 'JPEG')

            try:
                with open(tmp_file.name, 'rb') as img_file:
                    response = client.post(
                        "/predict/multimodal",
                        data={"text": test_text},
                        files={"image": ("large.jpg", img_file, "image/jpeg")}
                    )

                # L'API devrait gérer les grandes images
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

            finally:
                os.unlink(tmp_file.name)


class TestPredictionResponseFormat:
    """Tests du format des réponses de prédiction"""

    def test_success_response_structure(self, client, test_image, test_text):
        """Test de la structure complète d'une réponse réussie"""
        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": test_text},
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        if response.status_code == 200:
            data = response.json()

            # Vérifier tous les champs requis
            required_fields = [
                "success", "predicted_class", "confidence",
                "input_text", "image_filename", "device_used"
            ]

            for field in required_fields:
                assert field in data, f"Champ manquant: {field}"

            # Vérifier les types de données
            assert isinstance(data["success"], bool)
            assert isinstance(data["predicted_class"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["input_text"], str)
            assert isinstance(data["image_filename"], str)
            assert isinstance(data["device_used"], str)

            # Vérifier les contraintes
            assert data["success"] is True
            assert 0.0 <= data["confidence"] <= 1.0
            assert len(data["predicted_class"]) > 0
            assert len(data["device_used"]) > 0

    def test_error_response_structure(self, client, test_text):
        """Test de la structure des réponses d'erreur"""
        # Test avec fichier manquant
        response = client.post(
            "/predict/multimodal",
            data={"text": test_text}
        )

        assert response.status_code in [400, 422, 500]
        data = response.json()

        # Les erreurs FastAPI ont généralement un champ "detail"
        assert "detail" in data
        assert isinstance(data["detail"], (str, list))

    def test_content_type_header(self, client, test_image, test_text):
        """Test du header Content-Type pour les prédictions"""
        with open(test_image, 'rb') as img_file:
            response = client.post(
                "/predict/multimodal",
                data={"text": test_text},
                files={"image": ("test.jpg", img_file, "image/jpeg")}
            )

        assert "application/json" in response.headers["content-type"]


class TestPredictionPerformance:
    """Tests de performance et charge"""

    def test_multiple_predictions(self, client, test_image, test_data):
        """Test de prédictions multiples consécutives"""
        texts = test_data["valid_texts"]

        for i, text in enumerate(texts):
            with open(test_image, 'rb') as img_file:
                response = client.post(
                    "/predict/multimodal",
                    data={"text": text},
                    files={"image": (f"test_{i}.jpg", img_file, "image/jpeg")}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Vérifier que chaque prédiction est indépendante
            assert data["input_text"] == text
            assert data["image_filename"] == f"test_{i}.jpg"

    def test_prediction_idempotency(self, client, test_image, test_text):
        """Test que les prédictions sont cohérentes"""
        responses = []

        # Faire plusieurs prédictions identiques
        for _ in range(3):
            with open(test_image, 'rb') as img_file:
                response = client.post(
                    "/predict/multimodal",
                    data={"text": test_text},
                    files={"image": ("test.jpg", img_file, "image/jpeg")}
                )
            responses.append(response)

        # Toutes les réponses devraient avoir le même status code
        status_codes = [r.status_code for r in responses]
        assert len(set(status_codes)) == 1

        # Si les prédictions réussissent, elles devraient être identiques
        if responses[0].status_code == 200:
            predicted_classes = [r.json()["predicted_class"] for r in responses]
            # Les prédictions devraient être identiques (modèle déterministe)
            assert len(set(predicted_classes)) == 1, "Prédictions incohérentes pour les mêmes données"