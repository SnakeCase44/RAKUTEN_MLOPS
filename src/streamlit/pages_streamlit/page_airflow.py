import streamlit as st

def run(role=None):
    st.title("Airflow")

    airflow_url = "http://localhost:8080"

    if role:
        st.info(f"Vous êtes connecté en tant que {role}.")

    st.components.v1.iframe(airflow_url, height=800, scrolling=True)
