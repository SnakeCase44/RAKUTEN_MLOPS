import streamlit as st


def run(role=None):
    st.title("🤖 Dashboard MLflow")

    mlflow_url = "http://localhost:8088/proxy/mlflow/"

    # Section accès rapide
    st.markdown("### 🚀 Accès à l'interface MLflow")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### MLflow Tracking Server")
        st.markdown(
            f"""<a href="{mlflow_url}" target="_blank">
                    <button style="padding:15px 25px; font-size:16px; background-color:#0194e2; color:white; border:none; border-radius:5px;">
                        📊 Ouvrir MLflow
                    </button>
                </a>""",
            unsafe_allow_html=True
        )
        st.markdown("**Interface de tracking des expériences**")

    with col2:
        st.markdown(f"**URL :** `{mlflow_url}`")

    st.markdown("---")

    # Section fonctionnalités
    st.markdown("### 📋 Fonctionnalités disponibles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔍 Expériences")
        st.markdown("""
        - **Historique complet** des runs
        - **Comparaison** des expériences
        - **Filtrage** par statut/date
        - **Tags** et métadonnées
        """)

    with col2:
        st.markdown("#### 📈 Métriques")
        st.markdown("""
        - **Métriques par classe** (27 classes)
        - **F1, Precision, Recall** par classe
        - **Métriques globales** (Macro/Micro/Weighted)
        - **Comparaison** multi-runs
        """)

    with col3:
        st.markdown("#### 🗂️ Modèles")
        st.markdown("""
        - **Registre** des modèles
        - **Versions** et stages
        - **Artefacts** sauvegardés
        - **Déploiement** simplifié
        """)

    st.markdown("---")

    # Section utilisation
    st.markdown("### 💡 Comment utiliser MLflow")

    st.markdown("""
    #### **Pour consulter vos expériences :**
    1. Cliquez sur le bouton **"Ouvrir MLflow"** ci-dessus
    2. Naviguez dans la liste des **expériences**
    3. Sélectionnez un **run** pour voir les détails
    4. Consultez les **métriques** et **paramètres**

    #### **Pour comparer des runs :**
    1. Sélectionnez plusieurs runs avec les **checkboxes**
    2. Cliquez sur **"Compare"** 
    3. Analysez les **graphiques comparatifs**
    4. Exportez les résultats si nécessaire

    #### **Pour analyser les performances par classe :**
    1. Dans un run, consultez les **métriques** de test
    2. Visualisez les **F1 scores** des 31 classes
    3. Comparez les **précisions** et **rappels** par classe
    4. Analysez les **métriques globales** (macro/micro/weighted)

    #### **Pour gérer les modèles :**
    1. Allez dans l'onglet **"Models"**
    2. Sélectionnez un modèle dans le **registre**
    3. Gérez les **versions** et **stages**
    4. Téléchargez les **artefacts**
    """)

    st.markdown("---")

    # Section informations
    st.markdown("### ℹ️ Informations système")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔧 Configuration")
        st.code("""
MLflow Server  : localhost:5005
Backend Store  : PostgreSQL
Artifact Store : Local filesystem
Default Path   : ./mlruns
        """, language="text")

    with col2:
        st.markdown("##### 📊 Métriques suivies")
        st.code("""
Classes (27) :
• test_1140_f1_score
• test_1320_f1_score
• test_*_precision
• test_*_recall

Globales :
• test_macro_f1
• test_weighted_f1
• test_accuracy
• epoch, lr-AdamW
        """, language="text")

    st.markdown("---")

    # Section détails classes
    st.markdown("### 🎯 Classes surveillées")

    st.markdown("""
    **27 classes de produits Rakuten :**

    **Gaming & Tech :** Console (60), Jeu vidéo (40), Accessoire Console (50), Jeu PC (2905)

    **Jouets & Enfants :** Figurine (1140), Jeu Plateau (1180), Jouet enfant (1280), 
    Jeu de société (1281), Jouet tech (1300), Autour du bébé (1320)

    **Livres & Médias :** Livre occasion (10), Livre neuf (2705), Magazines/BDs (2403), 
    Revues et journaux (2280)

    **Maison & Décoration :** Mobilier intérieur (1560), Chambre (1920), Cuisine (1940), 
    Décoration (2060), Mobilier extérieur (2582)

    **Et 8 autres classes** incluant cartes collection, vêtements, bricolage, bureautique...
    """)

    st.markdown("---")

    # Section footer
    st.markdown("### 🔧 Administration")
    if role == "admin":
        st.success("En tant qu'admin, vous avez accès à toutes les fonctionnalités MLflow.")
        st.markdown("""
        **Actions disponibles :**
        - Gestion des expériences et runs
        - Configuration du serveur MLflow
        - Nettoyage des anciens runs
        - Backup des données d'expériences
        - Accès aux métriques détaillées (27 classes)
        """)
    else:
        st.info("Accès en lecture aux expériences et modèles MLflow.")


if __name__ == "__main__":
    run()