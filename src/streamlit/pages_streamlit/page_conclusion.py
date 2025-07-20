import streamlit as st

def run(role=None):
    st.title("✅ Conclusion du Projet MLOps Rakuten")

    st.markdown("### 📦 Ce que nous avons accompli")

    st.markdown("""
Nous avons mis en place une architecture **MLOps complète** en utilisant Docker Compose, incluant :

- 🔐 Une API FastAPI sécurisée avec **OAuth2 + JWT**
- 🧠 Un modèle **multimodal (texte + image)** intégré et traçable
- 📊 Un serveur **MLflow** connecté à tous les runs d’entraînement
- 🧪 Des **tests unitaires** garantissant la fiabilité du système
- 📈 Un dashboard **Streamlit** pour la visualisation et l’interaction
- 📡 Une solution de **monitoring** avec **Prometheus & Grafana**
- 🔁 Une gateway et un reverse proxy **centralisant les accès**
- ⏳ Une base **Airflow prête** pour orchestrer de futurs pipelines
    """)

    st.markdown("---")
    st.markdown("### 🧠 Et si nous avions eu plus de temps...")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ☁️ Scalabilité avec Kubernetes")
        st.markdown("""
        - Déploiement sur **Kubernetes**
        - **Autoscaling** des services API/Streamlit
        - Gestion des **volumes persistants** pour MLflow et Postgres
        """)

        st.markdown("#### 🔁 Automatisation de l’ingestion")
        st.markdown("""
        - Activation du **scheduler Airflow**
        - Détection automatique de nouvelles données
        - Réentraînement conditionnel du modèle
        """)

    with col2:
        st.markdown("#### 🚀 Pipeline CI/CD avancé")
        st.markdown("""
        - Tests automatisés
        - Build et push d’images Docker
        - Déploiement automatique via **GitHub Actions**
        """)

        st.markdown("#### 📣 Monitoring enrichi")
        st.markdown("""
        - Alertes sur les métriques critiques (F1, latence API)
        """)

    st.markdown("---")
    st.markdown("### 🎯 Notre vision à long terme")

    st.markdown("""
Construire une **plateforme intelligente et scalable**, capable de :

- S'adapter aux **variations de charge**
- Suivre l’évolution des **données** et des **modèles**
- Permettre aux utilisateurs de **piloter leur IA** via des interfaces simples
- Offrir une **observabilité complète** de bout en bout

""")

    st.markdown("---")
    st.markdown("### 🙌 Remerciements")

    st.success("Merci à tous pour votre attention")

if __name__ == "__main__":
    run()
