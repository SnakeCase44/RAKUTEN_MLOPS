import streamlit as st
import requests
import os

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/train"

DEFAULT_HYPERPARAMS = {
    "batch_size": 48,
    "max_epochs": 15,
    "lr": 5e-6,
    "patience": 2,
    "dropout": 0.4,
    "weight_decay": 0.01,
    "hidden_size": 512,
    "label_smoothing": 0.15,
}

def run(role=None):
    st.set_page_config(page_title="Multimodal Trainer", page_icon="🧠")
    st.title("🧠 Entraînement du modèle multimodal")
    st.write("Ajustez les hyperparamètres ci-dessous puis lancez l'entraînement.")

    if role:
        st.info(f"Vous êtes connecté en tant que {role}.")

    with st.form("train_form"):
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", value=DEFAULT_HYPERPARAMS["batch_size"], min_value=1)
            max_epochs = st.number_input("Max Epochs", value=DEFAULT_HYPERPARAMS["max_epochs"], min_value=1)
            lr = st.number_input("Learning Rate", value=DEFAULT_HYPERPARAMS["lr"], format="%.6f")
            patience = st.number_input("Patience", value=DEFAULT_HYPERPARAMS["patience"], min_value=0)
        with col2:
            dropout = st.slider("Dropout", min_value=0.0, max_value=1.0, value=DEFAULT_HYPERPARAMS["dropout"])
            weight_decay = st.number_input("Weight Decay", value=DEFAULT_HYPERPARAMS["weight_decay"], format="%.5f")
            hidden_size = st.number_input("Hidden Size", value=DEFAULT_HYPERPARAMS["hidden_size"], min_value=1)
            label_smoothing = st.slider("Label Smoothing", min_value=0.0, max_value=1.0, value=DEFAULT_HYPERPARAMS["label_smoothing"])

        submitted = st.form_submit_button("🚀 Lancer l'entraînement")

        if submitted:
            payload = {
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "lr": lr,
                "patience": patience,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "hidden_size": hidden_size,
                "label_smoothing": label_smoothing
            }

            st.info("📨 Envoi des hyperparamètres à l'API...")
            with st.spinner("Entraînement en cours..."):
                try:
                    response = requests.post(API_URL, data=payload, timeout=600)
                    response.raise_for_status()
                    res = response.json()

                    if res["status"] == "success":
                        st.success("✅ Entraînement terminé avec succès !")
                        with st.expander("📄 Voir le log de sortie"):
                            st.text(res.get("stdout", "Aucun output retourné"))
                    else:
                        st.error(f"❌ Erreur : {res.get('message', 'Erreur inconnue')}")
                        st.code(res.get("stderr", ""), language="bash")

                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Erreur de connexion à l'API : {e}")

                except ValueError:
                    st.error("⚠️ La réponse de l'API n'était pas au format JSON valide.")

