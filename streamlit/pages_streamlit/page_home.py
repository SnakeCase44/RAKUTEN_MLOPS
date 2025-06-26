import streamlit as st
from PIL import Image


def run():
    col1, col2 = st.columns([1, 4])
    with col1:
        
        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        ## Menu
        - [Accueil](./page_home)
        - [Entraînement modèle](./page_train)
        """)
    with col2:
        st.subheader("Accueil")
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

