import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st
import requests

# Lire les query params
query_params = st.query_params
token = query_params.get("token", None)

# Si token dans URL, on l'ajoute à la session
if token and "access_token" not in st.session_state:
    st.session_state.access_token = token
    st.query_params.clear()

# Vérifie si déconnexion demandée
if "logout" in query_params:
    st.session_state.clear()
    st.success("Déconnexion réussie. Cliquez [ici](http://127.0.0.1:8000/) pour vous reconnecter.")
    st.stop()

token = st.session_state.get("access_token")

if token:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get("http://rakuten_api:8000/users/me", headers=headers)
        if response.status_code == 200:
            user = response.json()
            username = user['username']
            role = user.get("role", "inconnu")

            # ➕ Bienvenue + lien logout alignés
            col1, col2 = st.columns([5, 1])
            with col1:
                st.success(f"Bienvenue {username} !")
            with col2:
                st.markdown(
                    "<div style='text-align: right;'>"
                    "<a href='?logout=true' style='color:red;text-decoration:none;'>🚪 Logout</a>"
                    "</div>",
                    unsafe_allow_html=True
                )

            # Définir les pages accessibles en fonction du rôle
            if role == "admin":
                accessible_pages = ["Accueil", "Entraînement modèle", "Classification produit", "Dashboard MLflow", "Monitoring", "Airflow", "Conclusion"]
            elif role == "dev":
                accessible_pages = ["Accueil", "Entraînement modèle", "Classification produit", "Monitoring", "Conclusion"]
            elif role == "client":
                accessible_pages = ["Accueil", "Classification produit"]
            else:
                st.warning("Rôle non reconnu.")
                accessible_pages = ["Accueil"]

            st.set_page_config(page_title="RAKUTEN", layout="wide")

            # Navigation latérale
            st.sidebar.title("📁 Navigation")
            page = st.sidebar.radio("Aller à :", accessible_pages)

            # Exécuter la page sélectionnée
            if page == "Accueil":
                from pages_streamlit import page_home
                page_home.run(role)
            elif page == "Entraînement modèle":
                from pages_streamlit import page_train
                page_train.run(role)
            elif page == "Classification produit":
                from pages_streamlit import page_predict
                page_predict.run(role)
            elif page == "Dashboard MLflow":
                from pages_streamlit import page_MLflow
                page_MLflow.run(role)
            elif page == "Airflow":
                from pages_streamlit import page_airflow
                page_airflow.run(role)
            elif page == "Monitoring":
                from pages_streamlit import page_monitoring
                page_monitoring.run(role)
            elif page == "Conclusion":
                from pages_streamlit import page_conclusion
                page_conclusion.run(role)
        else:
            st.error("Token invalide ou expiré.")
            st.markdown("[Connectez-vous ici](http://127.0.0.1:8000/)")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}")
        st.markdown("[Connectez-vous ici](http://127.0.0.1:8000/)")
else:
    st.warning("Connectez-vous via FastAPI d'abord.")
    st.markdown("[Connectez-vous ici](http://127.0.0.1:8000/)")
