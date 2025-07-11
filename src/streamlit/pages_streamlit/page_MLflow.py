import streamlit as st

def run(role=None):
    st.title("🤖 Dashboard MLflow")
    
    mlflow_url = "http://localhost:8088/proxy/mlflow/"
    
    if role:
        st.info(f"Vous êtes connecté en tant que {role}.")
    
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
        st.markdown("#### Statut")
        st.success("🟢 Serveur actif")
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
        - **Graphiques** d'évolution
        - **Métriques** F1, Precision, Recall
        - **Courbes** de loss et accuracy
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
MLflow Server  : localhost:5000
Backend Store  : PostgreSQL
Artifact Store : Local filesystem
Default Path   : ./mlruns
        """, language="text")
    
    with col2:
        st.markdown("##### 📊 Métriques suivies")
        st.code("""
• test_60_f1
• test_60_precision  
• test_60_recall
• epoch
• lr-AdamW
        """, language="text")
    
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
        """)
    else:
        st.info("Accès en lecture aux expériences et modèles MLflow.")

if __name__ == "__main__":
    run()