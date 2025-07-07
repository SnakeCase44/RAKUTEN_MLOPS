from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime, timedelta
import os
import base64
from dotenv import load_dotenv

# Charger les variables d’environnement
load_dotenv(dotenv_path="/opt/airflow/.env")

API_USER = os.getenv("RAKUTEN_API_USER")
API_PASSWORD_B64 = os.getenv("RAKUTEN_API_PASSWORD_B64")
API_PASSWORD = base64.b64decode(API_PASSWORD_B64).decode() if API_PASSWORD_B64 else None
TRAIN_STATUS_PATH = "/app/models/multimodal_transformer_classifier/model/train_status.json"

default_args = {
    'start_date': datetime(2025, 7, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def authenticate_and_push_token(**context):
    import requests

    response = requests.post(
        "http://api:8000/token",
        data={"username": API_USER, "password": API_PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code != 200:
        raise Exception(f"❌ Auth échouée : {response.text}")
    token = response.json().get("access_token")
    context['ti'].xcom_push(key='access_token', value=token)

def trigger_training(**context):
    import requests
    import time

    token = context['ti'].xcom_pull(task_ids='authenticate', key='access_token')
    headers = {"Authorization": f"Bearer {token}"}
    params = context["params"]
    files = {k: (None, str(v)) for k, v in params.items()}

    response = requests.post("http://api:8000/train", headers=headers, files=files)
    if response.status_code != 200:
        raise Exception(f"❌ Échec lancement entraînement : {response.text}")

    time.sleep(5)

def check_train_status_file():
    import json
    print("🔍 Vérification du statut d'entraînement...")

    if not os.path.exists(TRAIN_STATUS_PATH):
        print("📁 Fichier de statut non trouvé")
        return False
    with open(TRAIN_STATUS_PATH, "r") as f:
        try:
            status = json.load(f).get("state")
            return status == "done"
        except json.JSONDecodeError:
            return False

def evaluate_model_func(**context):
    import requests

    token = context['ti'].xcom_pull(task_ids='authenticate', key='access_token')
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get("http://api:8000/evaluate", headers=headers)
    if response.status_code != 200:
        raise Exception(f"❌ Échec évaluation : {response.text}")

    data = response.json()
    metrics = data.get("metrics", {})
    print("📊 Résultats du modèle :")
    for key, val in metrics.items():
        print(f"{key:25}: {val:.4f}")

with DAG(
    "rakuten_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Pipeline Rakuten avec FileSensor',
    tags=['rakuten', 'ml', 'sensor'],
    params={
        "batch_size": 48,
        "max_epochs": 1,
        "lr": 5e-6,
        "patience": 2,
        "dropout": 0.4,
        "weight_decay": 0.01,
        "hidden_size": 512,
        "label_smoothing": 0.15
    }
) as dag:

    authenticate = PythonOperator(
        task_id='authenticate',
        python_callable=authenticate_and_push_token,
    )

    check_env = BashOperator(
        task_id='check_environment',
        bash_command="""
        echo "=== Environment Check ==="
        python -c "import torch; print(f'PyTorch: {torch.__version__}')"
        python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
        python -c "import pytest; print(f'Pytest: {pytest.__version__}')"
        """
    )

    run_tests = BashOperator(
        task_id='run_tests',
        bash_command="cd /app && python -m pytest tests/ --disable-warnings -v --tb=short"
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=trigger_training,
        execution_timeout=timedelta(hours=2)
    )

    wait_training = PythonSensor(
        task_id="wait_training_completion",
        python_callable=check_train_status_file,
        timeout=60 * 45,  # 45 minutes
        poke_interval=10,
        mode="poke",
    )

    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_func,
        execution_timeout=timedelta(minutes=5),
    )

    authenticate >> check_env >> run_tests >> train_model >> wait_training >> evaluate_model
