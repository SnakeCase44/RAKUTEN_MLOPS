import pytest
import json
from fastapi.testclient import TestClient


class TestRootEndpoints:
    """Tests des endpoints racine et de base"""

    def test_test_endpoint(self, client):
        """Test de l'endpoint de test /test"""
        response = client.get("/test")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello, Rakuten World!"

    def test_endpoints_return_json(self, client):
        """Test que les endpoints de base retournent du JSON"""
        endpoints = ["/test"]  # Supprimé "/" de la liste

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.headers["content-type"] == "application/json"

            # Vérifier que c'est du JSON valide
            try:
                response.json()
            except json.JSONDecodeError:
                pytest.fail(f"Réponse de {endpoint} n'est pas du JSON valide")


class TestHealthEndpoints:
    """Tests des endpoints de santé et informations"""

    def test_health_endpoint(self, client):
        """Test de l'endpoint de santé des modèles"""
        response = client.get("/predict/health")

        assert response.status_code == 200
        data = response.json()

        # Vérifier la structure de la réponse
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data

        # Le statut devrait être "healthy" ou "unhealthy"
        assert data["status"] in ["healthy", "unhealthy"]

        # models_loaded devrait être un booléen
        assert isinstance(data["models_loaded"], bool)

        # device devrait être une string
        assert isinstance(data["device"], str)

    def test_model_info_endpoint(self, client):
        """Test de l'endpoint d'informations sur les modèles"""
        response = client.get("/predict/info")

        assert response.status_code == 200
        data = response.json()

        # Vérifier que les informations sont présentes
        assert "models_loaded" in data

        if data["models_loaded"]:
            assert "model_paths" in data
            assert "device" in data
            assert "model_type" in data

            # Vérifier la structure des chemins de modèles
            model_paths = data["model_paths"]
            expected_keys = ["image_model", "text_model", "multimodal_model"]
            for key in expected_keys:
                assert key in model_paths
                assert isinstance(model_paths[key], str)


class TestErrorHandling:
    """Tests de gestion d'erreurs pour les endpoints de base"""

    def test_nonexistent_endpoint(self, client):
        """Test d'endpoint inexistant"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method_on_test(self, client):
        """Test avec mauvaise méthode HTTP sur /test"""
        response = client.post("/test")  # GET attendu
        assert response.status_code == 405

    def test_wrong_method_on_health(self, client):
        """Test avec mauvaise méthode HTTP sur /predict/health"""
        response = client.post("/predict/health")  # GET attendu
        assert response.status_code == 405


class TestResponseFormat:
    """Tests du format des réponses des endpoints de base"""

    def test_content_type_headers(self, client):
        """Test des headers Content-Type"""
        endpoints = ["/test", "/predict/health", "/predict/info"]  # Supprimé "/"

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "application/json" in response.headers["content-type"]

    def test_response_structure_consistency(self, client):
        """Test de la cohérence de structure des réponses"""
        # Test que toutes les réponses sont des objets JSON
        endpoints = ["/test", "/predict/health", "/predict/info"]  # Supprimé "/"

        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict), f"Réponse de {endpoint} n'est pas un objet JSON"

    def test_health_response_fields(self, client):
        """Test des champs spécifiques de la réponse health"""
        response = client.get("/predict/health")
        data = response.json()

        required_fields = ["status", "models_loaded", "device"]
        for field in required_fields:
            assert field in data, f"Champ manquant dans /predict/health: {field}"

        # Test des types de données
        assert isinstance(data["status"], str)
        assert isinstance(data["models_loaded"], bool)
        assert isinstance(data["device"], str)


class TestEdgeCases:
    """Tests de cas limites pour les endpoints de base"""

    def test_endpoints_with_trailing_slash(self, client):
        """Test des endpoints avec slash final"""
        # Certains endpoints peuvent accepter ou rediriger les slash finaux
        response = client.get("/test/")
        # Soit 200 (accepté), soit 307/308 (redirection), soit 404 (non géré)
        assert response.status_code in [200, 307, 308, 404]

    def test_case_sensitivity(self, client):
        """Test de la sensibilité à la casse"""
        # Les URLs devraient être sensibles à la casse
        response = client.get("/TEST")
        assert response.status_code == 404

        response = client.get("/Test")
        assert response.status_code == 404

    def test_health_endpoint_multiple_calls(self, client):
        """Test d'appels multiples à l'endpoint de santé"""
        # L'endpoint de santé devrait être idempotent
        responses = [client.get("/predict/health") for _ in range(3)]

        # Tous les appels devraient avoir le même status code
        status_codes = [r.status_code for r in responses]
        assert len(set(status_codes)) == 1, "L'endpoint de santé n'est pas idempotent"

        # Si les modèles sont chargés, ils devraient le rester
        if responses[0].status_code == 200:
            models_loaded_states = [r.json()["models_loaded"] for r in responses]
            if models_loaded_states[0]:  # Si chargés la première fois
                assert all(models_loaded_states), "État des modèles incohérent"