import streamlit as st

def run(role=None):
    st.title("Dashboard MLflow")
    mlflow_url = "http://localhost:5005"
    
    if role:
        st.success(f"✅ Connecté en tant que **{role}**")
    
    st.markdown("---")
    st.subheader("Accéder au tableau de bord MLflow")
    
    st.markdown(
        f"""
        <div style='text-align: center; padding-top: 30px;'>
            <a href="{mlflow_url}" target="_blank" style="
                display: inline-block;
                background-color: #0e76a8;
                color: white;
                padding: 1em 2em;
                text-decoration: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
                transition: background-color 0.3s ease;
            ">
                🚀 Ouvrir MLflow dans un nouvel onglet
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    


if __name__ == "__main__":
    run()