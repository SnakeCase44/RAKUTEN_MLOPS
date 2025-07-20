import streamlit as st
import requests
from datetime import datetime, timezone


def run(role=None):
    st.title("Airflow")

    airflow_url = "http://localhost:8080"

    st.markdown("### Accès à l'interface Airflow")

    st.markdown(
        f"""<a href="{airflow_url}" target="_blank">
                <button style="padding:10px 20px; font-size:16px;">Ouvrir l'interface Airflow dans un nouvel onglet</button>
            </a>""",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Description détaillée du pipeline `rakuten_pipeline`")

    st.markdown("""
Ce pipeline automatisé géré par Apache Airflow orchestre l'ensemble du processus MLOps autour d’un modèle multimodal (texte + image). Il se compose de plusieurs étapes successives et utilise différentes fonctionnalités natives d’Airflow.

---

#### **1. `authenticate`** (PythonOperator)
Cette tâche s’authentifie auprès de l’API FastAPI.  
Elle utilise un identifiant et un mot de passe encodé en base64, fournis dans le fichier `.env`.  
Une fois les identifiants validés, un **token JWT** est obtenu et stocké dans les **XComs** d’Airflow pour être utilisé par les tâches suivantes.

---

#### **2. `check_environment`** (BashOperator)
Cette tâche exécute quelques commandes shell pour vérifier que les principales librairies de l’environnement sont disponibles et correctement configurées :
- PyTorch
- Transformers (Hugging Face)
- Pytest (pour les tests unitaires)

---

#### **3. `run_tests`** (BashOperator)
Lance les tests unitaires définis dans le dossier `tests/` via `pytest`.  
Cela garantit que l’API est stable avant de déclencher l'entraînement.

---

#### **4. `train_model`** (PythonOperator)
Déclenche l'entraînement du modèle en envoyant une requête POST vers `/train` sur l'API FastAPI.  
Les hyperparamètres sont passés dynamiquement via le champ `params` du DAG.  
Cette tâche démarre **un entraînement asynchrone**, en tâche de fond.

---

#### **5. `wait_training_completion`** (PythonSensor)
Ce **capteur (sensor)** vérifie périodiquement l’état d’avancement de l'entraînement.  
Il lit un fichier `train_status.json` et attend que son champ `"state"` passe à `"done"` ou `"error"`.  
Le **mode `poke`** est utilisé, avec une interrogation toutes les 10 secondes pendant un maximum de 45 minutes.

---

#### **6. `evaluate_model`** (PythonOperator)
Une fois l’entraînement terminé, cette tâche appelle `/evaluate` sur l’API.  
Les métriques (accuracy, F1-score, etc.) sont extraites et affichées, puis stockées dans MLflow.  
Un rapport complet est aussi sauvegardé dans un fichier `.txt`.

---

### Fonctionnalités Airflow exploitées dans ce DAG :

- **`PythonOperator`** : pour les appels API (`authenticate`, `train_model`, `evaluate_model`)
- **`BashOperator`** : pour les vérifications d’environnement et les tests
- **`PythonSensor`** : pour attendre une condition personnalisée (fin de l'entraînement)
- **`params`** : permet de passer dynamiquement les hyperparamètres à la tâche d’entraînement
- **`XCom`** : utilisé pour transférer le token JWT entre tâches
- **`execution_timeout`** : définit une durée maximale par tâche pour éviter les blocages
- **`tags` et `description`** : pour améliorer la lisibilité dans l'interface Airflow
- **`trigger_rule=all_success`** (par défaut) : chaque tâche ne s’exécute que si toutes les précédentes sont un succès

---

Ce DAG constitue un exemple concret de pipeline CI/CD dans un contexte MLOps.  
Il automatise la chaîne : **authentification → validation → entraînement → attente → évaluation**.

""")
