import streamlit as st
import requests
from datetime import datetime, timezone


def run(role=None):
    st.title("📊 Monitoring Grafana & Prometheus")

    grafana_url = "http://localhost:3000"
    prometheus_url = "http://localhost:9090"

    if role:
        st.info(f"Connecté en tant que {role}.")

    # Section accès rapide
    st.markdown("### 🚀 Accès rapide aux interfaces")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Grafana Dashboard")
        st.markdown(
            f"""<a href="{grafana_url}" target="_blank">
                    <button style="padding:15px 25px; font-size:16px; background-color:#f46800; color:white; border:none; border-radius:5px;">
                        📈 Ouvrir Grafana
                    </button>
                </a>""",
            unsafe_allow_html=True
        )
        st.markdown("**Identifiants :** admin / admin")

    with col2:
        st.markdown("#### Prometheus Metrics")
        st.markdown(
            f"""<a href="{prometheus_url}" target="_blank">
                    <button style="padding:15px 25px; font-size:16px; background-color:#e6522c; color:white; border:none; border-radius:5px;">
                        🔍 Ouvrir Prometheus
                    </button>
                </a>""",
            unsafe_allow_html=True
        )
        st.markdown("**Interface de requête directe**")

    st.markdown("---")

    # Section dashboards disponibles
    st.markdown("### 📋 Dashboards disponibles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🤖 MLflow Metrics")
        st.markdown("""
        - **F1, Precision, Recall** en temps réel
        - **Historique des entraînements**
        - **Métriques du dernier run**
        - **Paramètres et durées**
        """)
        st.markdown(
            f"""<a href="{grafana_url}/d/mlflow-monitor" target="_blank">
                    <button style="padding:8px 16px; font-size:14px; background-color:#52c41a; color:white; border:none; border-radius:3px;">
                        Voir Dashboard MLflow
                    </button>
                </a>""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("#### ⚡ FastAPI Monitoring")
        st.markdown("""
        - **Statut de l'API** (up/down)
        - **Nombre de requêtes totales**
        - **Temps de réponse moyen**
        - **Requêtes par endpoint**
        """)
        st.markdown(
            f"""<a href="{grafana_url}/d/fastapi-metrics" target="_blank">
                    <button style="padding:8px 16px; font-size:14px; background-color:#1890ff; color:white; border:none; border-radius:3px;">
                        Voir Dashboard FastAPI
                    </button>
                </a>""",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("#### 🖥️ Node Exporter Full")
        st.markdown("""
        - **CPU, Mémoire, Disque**
        - **Charge système**
        - **I/O disque et réseau**
        - **Santé des containers**
        """)
        st.markdown(
            f"""<a href="{grafana_url}/d/rYdddlPWk" target="_blank">
                    <button style="padding:8px 16px; font-size:14px; background-color:#722ed1; color:white; border:none; border-radius:3px;">
                        Voir Dashboard Système
                    </button>
                </a>""",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Section architecture
    st.markdown("### 🏗️ Architecture de monitoring")

    st.markdown("""
    Notre stack de monitoring se compose de plusieurs composants interconnectés :

    #### **1. Collecte des métriques**
    - **FastAPI** : Expose automatiquement des métriques HTTP (requêtes, latence, statuts)
    - **MLflow Exporter** : Lit les fichiers MLflow et les convertit en métriques Prometheus
    - **Node Exporter** : Métriques système (CPU, mémoire, disque)

    #### **2. Stockage des métriques - Prometheus**
    - **Base de données temporelle** pour stocker toutes les métriques
    - **Scraping automatique** toutes les 15 secondes
    - **Rétention** de 15 jours par défaut
    - **Langage de requête PromQL** pour l'analyse

    #### **3. Visualisation - Grafana**
    - **Dashboards interactifs** avec graphiques en temps réel
    - **Alerting** (configurable) sur seuils personnalisés
    - **Export** possible en PDF/PNG
    - **Provisioning automatique** des datasources et dashboards

    ---

    #### **Configuration actuelle :**
    """)

    # Section configuration
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📡 Sources de données")
        st.code("""
FastAPI         → :8000/metrics
MLflow Exporter → :8001/metrics  
Node Exporter   → :9101/metrics
Prometheus      → :9090
        """, language="text")

    with col2:
        st.markdown("##### ⏱️ Intervalles")
        st.code("""
Scraping    : 15s
Refresh     : 10s  
Rétention   : 15 jours
Timeout     : 5s
        """, language="text")

    st.markdown("---")

    # Section métriques clés
    st.markdown("### 📊 Métriques clés surveillées")

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 MLflow", "⚡ FastAPI", "🖥️ Système", "🐳 Docker"])

    with tab1:
        st.markdown("""
        **Métriques d'entraînement :**
        - `mlflow_run_metric{name="test_60_f1"}` - Score F1 du dernier test
        - `mlflow_run_metric{name="test_60_precision"}` - Précision du modèle
        - `mlflow_run_metric{name="test_60_recall"}` - Rappel du modèle
        - `mlflow_run_metric{name="epoch"}` - Progression des époques
        - `mlflow_run_metric{name="lr-AdamW"}` - Taux d'apprentissage

        **Métriques de run :**
        - `mlflow_run_duration_seconds` - Durée d'entraînement
        - `mlflow_run_status` - Statut (0=Running, 1=Finished)
        - `mlflow_run_param{name="batch_size"}` - Paramètres d'entraînement
        """)

    with tab2:
        st.markdown("""
        **Métriques HTTP :**
        - `http_requests_total` - Nombre total de requêtes
        - `http_request_duration_seconds` - Temps de réponse
        - `up{job="fastapi"}` - Statut de l'API (1=UP, 0=DOWN)

        **Exemples de requêtes PromQL :**
        ```
        # Taux de requêtes par minute
        rate(http_requests_total[1m])

        # Temps de réponse moyen
        rate(http_request_duration_seconds_sum[1m]) / 
        rate(http_request_duration_seconds_count[1m])
        ```
        """)

    with tab3:
        st.markdown("""
        **Métriques CPU :**
        - `node_cpu_seconds_total` - Temps CPU par mode
        - `node_load1` / `node_load5` / `node_load15` - Charge système
        - `100 - (rate(node_cpu_seconds_total{mode="idle"}[5m]) * 100)` - % CPU utilisé

        **Métriques Mémoire :**
        - `node_memory_MemTotal_bytes` - Mémoire totale
        - `node_memory_MemAvailable_bytes` - Mémoire disponible
        - `node_memory_Buffers_bytes` / `node_memory_Cached_bytes` - Cache

        **Métriques Disque :**
        - `node_filesystem_avail_bytes` - Espace disque disponible
        - `node_disk_io_time_seconds_total` - I/O disque
        """)

    with tab4:
        st.markdown("""
        **Conteneurs surveillés :**
        - `rakuten_api` - API FastAPI principale
        - `rakuten_mlflow` - Serveur MLflow  
        - `rakuten_postgres` - Base de données
        - `rakuten_streamlit` - Interface utilisateur
        - `rakuten_airflow` - Orchestrateur de workflows

        **Métriques réseau :**
        - `node_network_receive_bytes_total` - Données reçues
        - `node_network_transmit_bytes_total` - Données transmises
        """)

    st.markdown("---")

    # Section utilisation
    st.markdown("### 💡 Comment utiliser")

    st.markdown("""
    #### **Pour suivre un entraînement :**
    1. Lancez votre entraînement MLflow
    2. Ouvrez le **Dashboard MLflow** dans Grafana
    3. Surveillez les métriques en temps réel (F1, loss, etc.)
    4. Les **gros chiffres** affichent toujours le dernier run

    #### **Pour surveiller l'API :**
    1. Ouvrez le **Dashboard FastAPI**
    2. Vérifiez que l'API est **UP** (indicateur vert)
    3. Surveillez le **temps de réponse** et le **nombre de requêtes**

    #### **Pour surveiller le système :**
    1. Ouvrez le **Dashboard Node Exporter Full**
    2. Vérifiez l'utilisation **CPU, RAM et disque**
    3. Surveillez les **I/O** et la **charge système**
    4. Contrôlez la **santé des containers Docker**

    #### **Pour analyser les performances :**
    - Utilisez l'**historique complet** dans les graphiques
    - **Comparez** les différents runs d'entraînement
    - **Exportez** les dashboards pour vos rapports
    """)

    st.markdown("---")

    # Section footer
    st.markdown("### 🔧 Administration")
    if role == "admin":
        st.success("En tant qu'admin, vous avez accès à toutes les fonctionnalités de monitoring.")
        st.markdown("""
        **Actions disponibles :**
        - Configuration des alertes Grafana
        - Gestion des utilisateurs et permissions
        - Export des données de monitoring
        - Configuration des rétentions Prometheus
        """)
    else:
        st.info("Accès en lecture seule aux dashboards de monitoring.")