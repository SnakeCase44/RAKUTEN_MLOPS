from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'start_date': datetime(2025, 7, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    "rakuten_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Pipeline de tests et formation du modèle Rakuten',
    tags=['rakuten', 'ml', 'tests'],
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

    # Vérifier l'environnement
    check_env = BashOperator(
        task_id='check_environment',
        bash_command='''
        echo "=== Environment Check ==="
        echo "Current directory: $(pwd)"
        echo "Python version: $(python --version)"
        python -c "import torch; print(f'PyTorch: {torch.__version__}')"
        python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
        python -c "import pytest; print(f'Pytest: {pytest.__version__}')"
        ls -la /app/
        ls -la /app/tests/ || echo "Tests directory not found"
        ls -la /app/models/ || echo "Models directory not found"
        '''
    )

    # Lancer les tests
    run_tests = BashOperator(
        task_id='run_tests',
        bash_command='''
        cd /app &&
        echo "=== Running Unit Tests ===" &&
        python -m pytest tests/ --disable-warnings -v --tb=short
        '''
    )

    # Entraînement via API avec hyperparamètres dynamiques
    train_model = BashOperator(
        task_id='train_model',
        bash_command="""
        echo "=== Triggering training via FastAPI ===" &&
        curl -X POST http://api:8000/train \
          -F "batch_size={{ params.batch_size }}" \
          -F "max_epochs={{ params.max_epochs }}" \
          -F "lr={{ params.lr }}" \
          -F "patience={{ params.patience }}" \
          -F "dropout={{ params.dropout }}" \
          -F "weight_decay={{ params.weight_decay }}" \
          -F "hidden_size={{ params.hidden_size }}" \
          -F "label_smoothing={{ params.label_smoothing }}" &&
        echo "=== Training request sent ==="
        """,
        execution_timeout=timedelta(hours=2)
    )

    check_env >> run_tests >> train_model
