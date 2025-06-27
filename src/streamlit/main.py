import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st

st.set_page_config(page_title="RAKUTEN", layout="wide")

from pages_streamlit import (
    page_home,
    page_train 
)

if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Aller à :", [
    "Accueil",
    "Entraînement modèle",
    
])

st.session_state.page = page

page_switch = {
    "Accueil": page_home.run,
    "Entraînement modèle": page_train.run,

}

page_switch[st.session_state.page]()
