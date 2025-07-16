import streamlit as st
from PIL import Image

def run(role=None):
    
    st.title("Accueil")
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
        # Projet de Classification Multimodale Rakuten

        ### 🧭 Démarche globale

        Notre projet a été structuré selon les principes du cycle de vie **MLOps**, avec une forte attention portée à la **modularité**, la **reproductibilité**, et l’**automatisation** :

        1. 🔍 **Exploration des données** textuelles et visuelles, identification des déséquilibres, valeurs manquantes et bruit.
        2. 🧪 Développement de **modèles unimodaux** :
            - `XLM-RoBERTa` pour le texte,
            - `EfficientNet-B2` pour les images.
        3. 🔀 Construction de **modèles multimodaux** via une stratégie de fusion `mid-level`, en combinant les représentations issues des deux modalités.
        4. ⚙️ **Pipeline MLOps** :
            - Entraînement automatisé via **FastAPI**,
            - Intégration dans une interface **Streamlit** pour le pilotage,
            - Orchestration avec **Airflow**, packaging avec **Docker**, versioning, et logs structurés.

        ---

        ### ✅ Objectifs et bénéfices

        Notre architecture vise à :
        - **Tirer parti des deux modalités** (texte et image) de façon complémentaire,
        - **S’adapter aux cas réels** où certaines informations sont manquantes ou bruitées,
        - **Garantir la scalabilité** du projet pour de futurs déploiements ou tests de modèles.

        ---

        Ce projet constitue une mise en pratique complète de la démarche **Machine Learning orientée production**, intégrant à la fois des choix algorithmiques pertinents et des **bonnes pratiques MLOps**.
    """)
    st.title("🧩 Architecture des Microservices")
    st.subheader("📌 Schéma de l'architecture")
    image = Image.open("schema_microservices.png")
    st.image(image, caption="Schéma global des microservices", use_container_width=True)  
    st.subheader("📘 Détail des services")
    st.markdown("""
        ## 🔵 1. Base de Données (PostgreSQL)

        **`postgres`**  
        Conteneur principal. Stocke `rakuten_auth` et `airflow_db`. Port **5432**.

        **`postgres_init`**  
        Crée la base `airflow_db` à l'initialisation.

        **`init_db`**  
        Initialise `rakuten_auth` avec un script Python (`dbcrypt.py`).

        ---

        ## 🟢 2. API et Interface

        **`api` (FastAPI)**  
        Exporte les endpoints REST, charge un modèle ML, connectée à PostgreSQL et MLFlow. Port **8000**.

        **`streamlit`**  
        Interface utilisateur → interagit avec l’API via `nginx`. Port **8501**.

        ---

        ## 🟠 3. Machine Learning

        **`mlflow`**  
        Suivi des expérimentations. Port **5005**.

        ---

        ## 🟡 4. Orchestration

        **`airflow`**  
        Planifie les tâches et pipelines ML. Port **8080**.

        ---

        ## 🟣 5. Monitoring

        **`prometheus`** : collecte les métriques  
        **`grafana`** : visualisation des dashboards (port **3000**)  
        **`node_exporter`**, **`mlflow_exporter`** : sources de métriques

        ---

        ## 🟧 6. Proxy

        **`gateway`** : sécurise et route les requêtes  
        **`nginx`** : reverse proxy pour `/proxy/api`, `/proxy/mlflow`, etc.
    """)