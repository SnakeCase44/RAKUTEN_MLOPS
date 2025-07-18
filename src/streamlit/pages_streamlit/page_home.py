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
        # 🧩 Présentation de l’architecture des microservices RAKUTEN MLOps

        ---

        ## 🟦 1. Base de Données – PostgreSQL

        - Stocke deux bases :
        - `rakuten_auth` : pour l’authentification utilisateur
        - `airflow_db` : pour les métadonnées d’Airflow
        - `postgres_init` : crée `airflow_db` au démarrage
        - `init_db` : exécute un script Python (`dbcrypt.py`) pour initialiser les tables

        ---

        ## 🟧 2. API Principale – FastAPI (`rakuten_api`)

        - Noyau métier de l’application
        - Connectée à PostgreSQL (`rakuten_auth`)
        - Charge un modèle ML local depuis `MODEL_PATH`
        - Communique avec MLFlow pour tracer les expériences
        - Expose des endpoints pour la prédiction et l'entraînement
        - Compatible GPU via NVIDIA Docker runtime

        ---

        ## 🟩 3. Interface Utilisateur – Streamlit

        - Front-end interactif
        - Connectée à FastAPI via NGINX (`/proxy/api`)
        - Accès aux modèles et logs via volumes partagés
        - Permet de tester les prédictions et visualiser les résultats

        ---

        ## 🟦 4. Suivi des Expériences – MLflow

        - Serveur MLflow exposé à `/mlflow UI`
        - Stocke les métriques, hyperparamètres et artefacts dans `mlruns`
        - Connecté à l’API FastAPI pour tracer les entraînements

        ---

        ## 🔁 5. Export des Métriques – MLflow Exporter API

        - Convertit les logs MLflow pour Prometheus
        - Accessible sur le port `8001`
        - Permet à Prometheus de suivre l’évolution des expériences ML

        ---

        ## 📊 6. Monitoring – Prometheus & Grafana

        ### Prometheus
        - Récupère des métriques système, API, MLflow
        - Utilise `node_exporter` et `mlflow_exporter`

        ### Grafana
        - Visualise les métriques Prometheus
        - Dashboards préconfigurés (FastAPI, MLFlow, système)
        - Accessible sur le port `3000`

        ---

        ## ⚙️ 7. Orchestration – Airflow

        - Planifie des workflows ML (entraînement, nettoyage, etc.)
        - DAGs définis dans `src/airflow/dags/`
        - Utilise PostgreSQL (`airflow_db`) comme backend
        - Accessible sur le port `8080`

        ---

        ## 🔐 8. API Gateway – `rakuten_gateway`

        - Gère :
        - Authentification JWT
        - Redirection vers :
            - `/proxy/api`
            - `/streamlit`
            - `/mlflow`
            - `/prometheus`
        - Joue un rôle central dans la sécurité et l’agrégation des services

        ---

        ## 🌐 9. Reverse Proxy – NGINX

        - Expose tous les services vers Internet via le port `8088`
        - Gère les chemins `/proxy/*`
        - Sert un front HTML statique si besoin

        ---

        ## 🔚 Conclusion

        Cette architecture est :
        - **Modulaire** : chaque service a une responsabilité claire
        - **Robuste** : monitoring, gestion des erreurs, dépendances gérées
        - **MLOps-ready** : tracking, automatisation, reproductibilité

        ---


    """)