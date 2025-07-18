import streamlit as st
import requests
import os
from PIL import Image

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL}/predict/multimodal"

# Dictionnaire code -> libellé
CODE2LABEL = {
    10: "Livre occasion",
    40: "Jeu vidéo, accessoire tech.",
    50: "Accessoire Console",
    60: "Console de jeu",
    1140: "Figurine",
    1160: "Carte Collection",
    1180: "Jeu Plateau",
    1280: "Jouet enfant, déguisement",
    1281: "Jeu de société",
    1300: "Jouet tech",
    1301: "Paire de chaussettes",
    1302: "Jeu extérieur, vêtement",
    1320: "Autour du bébé",
    1560: "Mobilier intérieur",
    1920: "Chambre",
    1940: "Cuisine",
    2060: "Décoration intérieure",
    2220: "Animal",
    2280: "Revues et journaux",
    2403: "Magazines, livres et BDs",
    2462: "Jeu occasion",
    2522: "Bureautique et papeterie",
    2582: "Mobilier extérieur",
    2583: "Autour de la piscine",
    2585: "Bricolage",
    2705: "Livre neuf",
    2905: "Jeu PC",
}

def run(role=None):
    st.title("🎯 Classification Multimodale")
    st.write("Uploadez une image et saisissez une description pour classifier le produit.")

    if role:
        st.info(f"Vous êtes connecté en tant que {role}.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Description du produit")
            text_input = st.text_area(
                "Texte",
                placeholder="Ex: Jeu de plateau Monopoly édition classique",
                height=200
            )

        with col2:
            st.subheader("🖼️ Image du produit")
            uploaded_file = st.file_uploader(
                "Sélectionnez une image",
                type=['png', 'jpg', 'jpeg'],
                help="Formats: PNG, JPG, JPEG"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image: {uploaded_file.name}", use_container_width=True)

        submitted = st.form_submit_button("🚀 Prédire")

        if submitted:
            if not text_input or not text_input.strip():
                st.error("❌ Veuillez saisir une description")
                return

            if not uploaded_file:
                st.error("❌ Veuillez sélectionner une image")
                return

            st.info("📨 Envoi à l'API pour classification...")

            with st.spinner("Classification en cours... (peut prendre du temps si premier chargement)"):
                try:
                    files = {'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {'text': text_input}

                    response = requests.post(PREDICT_URL, files=files, data=data)
                    response.raise_for_status()
                    result = response.json()

                    if result.get("success"):
                        pred_code = result["predicted_class"]
                        confidence = result["confidence"]

                        # Conversion du code en libellé
                        try:
                            pred_code_int = int(pred_code)
                            pred_label_text = CODE2LABEL.get(pred_code_int, f"Classe inconnue ({pred_code_int})")

                            st.success(f"✅ Classe prédite : **{pred_code_int} - {pred_label_text}**")

                        except ValueError:
                            st.success(f"✅ Classe prédite : **{pred_code}**")

                        with st.expander("📋 Détails de la prédiction"):
                            st.json({
                                "Classe prédite": pred_code,
                                "Libellé": pred_label_text if 'pred_label_text' in locals() else "N/A",
                                "Confiance": f"{confidence:.4f}",
                                "Device utilisé": result.get("device_used", "N/A"),
                                "Nom fichier": result.get("image_filename", "N/A")
                            })
                    else:
                        st.error(f"❌ Erreur : {result.get('error', 'Erreur inconnue')}")

                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Erreur de connexion : {e}")
                except ValueError:
                    st.error("⚠️ Réponse API invalide")