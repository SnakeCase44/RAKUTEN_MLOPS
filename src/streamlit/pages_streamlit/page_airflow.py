import streamlit as st

def run():
    st.title("Airflow")

    airflow_url = "http://localhost:8080"

    st.components.v1.iframe(airflow_url, height=800, scrolling=True)
