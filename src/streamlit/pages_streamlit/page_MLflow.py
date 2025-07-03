import streamlit as st





def run():
    st.title("Dashboard MLflow")

    mlflow_url = "http://localhost:5000"


    st.markdown(f"""
    <iframe src="{mlflow_url}" width="100%" height="800" frameborder="0"></iframe>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    run()