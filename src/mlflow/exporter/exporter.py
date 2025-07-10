import os
import time
from prometheus_client import start_http_server, Gauge

MLFLOW_DIR = os.getenv("MLFLOW_DIR", "/mlflow/mlruns")
PORT = int(os.getenv("PORT", "8001"))
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 10))

# Une seule gauge réutilisée pour toutes les métriques
metric_gauge = Gauge('mlflow_run_metric', 'MLflow metric', ['experiment_id', 'run_id', 'name'])
param_gauge = Gauge('mlflow_run_param', 'MLflow param (numeric only)', ['experiment_id', 'run_id', 'name'])

# Gauges supplémentaires
duration_gauge = Gauge('mlflow_run_duration_seconds', 'MLflow run duration (if available)', ['experiment_id', 'run_id'])
status_gauge = Gauge('mlflow_run_status', 'MLflow run status (1 = finished)', ['experiment_id', 'run_id'])

def log(msg):
    print(f"[MLFLOW_EXPORTER] {msg}")

def parse_metrics(run_path, experiment_id, run_id):
    metrics_dir = os.path.join(run_path, "metrics")
    if not os.path.isdir(metrics_dir):
        return

    for fname in os.listdir(metrics_dir):
        fpath = os.path.join(metrics_dir, fname)
        try:
            with open(fpath, "r") as f:
                last_val = None
                for line in f:
                    if line.strip():
                        parts = line.strip().split(" ")
                        if len(parts) >= 2:
                            last_val = float(parts[1])  # Deuxième colonne = valeur réelle
                if last_val is None:
                    continue
                # Utilise la gauge unique pour toutes les métriques
                metric_gauge.labels(experiment_id, run_id, fname).set(last_val)
        except Exception as e:
            log(f"Error parsing metric {fpath}: {e}")

def parse_params(run_path, experiment_id, run_id):
    params_dir = os.path.join(run_path, "params")
    if not os.path.isdir(params_dir):
        return

    for fname in os.listdir(params_dir):
        fpath = os.path.join(params_dir, fname)
        try:
            # Skip directories
            if os.path.isdir(fpath):
                continue
            with open(fpath, "r") as f:
                val = f.read().strip()
                float_val = float(val)
                # Utilise la gauge unique pour tous les paramètres
                param_gauge.labels(experiment_id, run_id, fname).set(float_val)
        except ValueError:
            continue  # Non numérique
        except Exception as e:
            log(f"Error parsing param {fpath}: {e}")

def parse_extra(run_path, experiment_id, run_id):
    meta_path = os.path.join(run_path, "meta.yaml")
    if not os.path.isfile(meta_path):
        return

    try:
        with open(meta_path, "r") as f:
            lines = f.readlines()
            start, end, status = None, None, None
            for line in lines:
                if "start_time:" in line:
                    start = int(line.split(":")[1].strip())
                elif "end_time:" in line:
                    end = int(line.split(":")[1].strip())
                elif "status:" in line:
                    status = line.split(":")[1].strip()
            if start and end:
                duration = (end - start) / 1000
                duration_gauge.labels(experiment_id, run_id).set(duration)
            status_val = 1 if status == "FINISHED" else 0
            status_gauge.labels(experiment_id, run_id).set(status_val)
    except Exception as e:
        log(f"Error parsing meta.yaml: {e}")

def scan_mlruns():
    for experiment_id in os.listdir(MLFLOW_DIR):
        exp_path = os.path.join(MLFLOW_DIR, experiment_id)
        if not os.path.isdir(exp_path) or experiment_id.startswith("."):
            continue
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path):
                continue
            parse_metrics(run_path, experiment_id, run_id)
            parse_params(run_path, experiment_id, run_id)
            parse_extra(run_path, experiment_id, run_id)

def main():
    start_http_server(PORT)
    log(f"Exporter started on port {PORT}, reading from {MLFLOW_DIR}")
    while True:
        scan_mlruns()
        time.sleep(SCRAPE_INTERVAL)

if __name__ == "__main__":
    main()