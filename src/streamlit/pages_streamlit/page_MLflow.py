import streamlit as st

def run(role=None):
    st.title("Dashboard MLflow")

    mlflow_url = "http://localhost:5005"

    if role:
        st.info(f"Vous êtes connecté en tant que {role}.")

    st.markdown(f"""
    <iframe src="{mlflow_url}" width="100%" height="800" frameborder="0"></iframe>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()
