import os
import streamlit as st
import requests
from streamlit.components.v1 import html

# Lire les query params
query_params = st.query_params
token = query_params.get("token", None)

if token and "access_token" not in st.session_state:
    st.session_state.access_token = token
    st.query_params.clear()  # Nettoie l’URL

token = st.session_state.get("access_token")

if token:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get("http://host.docker.internal:8000/users/me", headers=headers)
        if response.status_code == 200:
            user = response.json()
            st.success(f"Bienvenue {user['username']} !")
            role = user.get("role", "inconnu")
            st.info(f"Rôle : {role}")

            # Définir les pages accessibles en fonction du rôle
            if role == "admin":
                st.header("👑 Admin Panel")
                accessible_pages = ["Accueil", "Entraînement modèle", "Classification produit"]
            elif role == "dev":
                st.header("👨‍💻 Espace Dev")
                accessible_pages = ["Accueil", "Entraînement modèle", "Classification produit"]
            elif role == "client":
                st.header("🛒 Espace Client")
                accessible_pages = ["Accueil", "Classification produit"]
            else:
                st.warning("Rôle non reconnu.")
                accessible_pages = ["Accueil"]

            # Configuration de la page
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
        else:
            st.error("Token invalide ou expiré.")
            # Redirection en cas d'erreur
            redirect_script = """
            <script>
            window.location.replace("http://host.docker.internal:8000/");
            </script>
            """
            st.markdown(redirect_script, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}")
        # Redirection en cas d'erreur
        redirect_script = """
        <script>
        window.location.replace("http://host.docker.internal:8000/");
        </script>
        """
        st.markdown(redirect_script, unsafe_allow_html=True)
else:
    st.warning("Connectez-vous via FastAPI d’abord.")
    # Redirection si non connecté
    redirect_script = """
    <script>
    window.location.replace("http://host.docker.internal:8000/");
    </script>
    """
    st.markdown(redirect_script, unsafe_allow_html=True)
