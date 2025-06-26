import streamlit as st
import requests

API_URL = "http://localhost:8000/train" 

DEFAULT_HYPERPARAMS = {
    "batch_size": 48,
    "max_epochs": 15,
    "lr": 5e-6,
    "patience": 2,
    "dropout": 0.4,
    "weight_decay": 0.01,
    "hidden_size": 512,
    "label_smoothing": 0.15
}

def run():
    st.title("🧠 Entraînement du modèle Multimodal")

    with st.form("train_form"):
        batch_size = st.number_input("Batch Size", value=DEFAULT_HYPERPARAMS["batch_size"])
        max_epochs = st.number_input("Max Epochs", value=DEFAULT_HYPERPARAMS["max_epochs"])
        lr = st.number_input("Learning Rate", value=DEFAULT_HYPERPARAMS["lr"], format="%.6f")
        patience = st.number_input("Patience", value=DEFAULT_HYPERPARAMS["patience"])
        dropout = st.number_input("Dropout", value=DEFAULT_HYPERPARAMS["dropout"])
        weight_decay = st.number_input("Weight Decay", value=DEFAULT_HYPERPARAMS["weight_decay"], format="%.5f")
        hidden_size = st.number_input("Hidden Size", value=DEFAULT_HYPERPARAMS["hidden_size"])
        label_smoothing = st.number_input("Label Smoothing", value=DEFAULT_HYPERPARAMS["label_smoothing"])

        submitted = st.form_submit_button("Lancer l'entraînement")

    if submitted:
        with st.spinner("Entraînement en cours..."):
            response = requests.post(API_URL, data={
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "lr": lr,
                "patience": patience,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "hidden_size": hidden_size,
                "label_smoothing": label_smoothing
            })

        if response.status_code == 200:
            res = response.json()
            if res["status"] == "success":
                st.success(res["message"])
                st.text_area("Résultat brut", res.get("output", "Aucun output"))
            else:
                st.error(res["message"])
        else:
            st.error(f"Erreur HTTP {response.status_code} : {response.text}")