from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess
import os
import json
from config import API_DIR, MULTIMODAL_MODEL_DIR

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(API_DIR, "templates"))

DEFAULT_HYPERPARAMS = {
    "batch_size": 48,
    "max_epochs": 1,
    "lr": 5e-6,
    "patience": 2,
    "dropout": 0.4,
    "weight_decay": 0.01,
    "hidden_size": 512,
    "label_smoothing": 0.15
}

TRAIN_STATUS_PATH = os.path.join(MULTIMODAL_MODEL_DIR, "train_status.json")
PROCESS_PID_PATH = os.path.join(MULTIMODAL_MODEL_DIR, "train_pid.txt")

@router.get("/train", response_class=HTMLResponse)
def get_train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request, "hyperparams": DEFAULT_HYPERPARAMS})

@router.post("/train")
def train_model(
    batch_size: int = Form(DEFAULT_HYPERPARAMS["batch_size"]),
    max_epochs: int = Form(DEFAULT_HYPERPARAMS["max_epochs"]),
    lr: float = Form(DEFAULT_HYPERPARAMS["lr"]),
    patience: int = Form(DEFAULT_HYPERPARAMS["patience"]),
    dropout: float = Form(DEFAULT_HYPERPARAMS["dropout"]),
    weight_decay: float = Form(DEFAULT_HYPERPARAMS["weight_decay"]),
    hidden_size: int = Form(DEFAULT_HYPERPARAMS["hidden_size"]),
    label_smoothing: float = Form(DEFAULT_HYPERPARAMS["label_smoothing"]),
):
    if os.path.exists(TRAIN_STATUS_PATH):
        os.remove(TRAIN_STATUS_PATH)

    hyperparams = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": lr,
        "patience": patience,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "label_smoothing": label_smoothing,
    }

    train_script_path = os.path.join(MULTIMODAL_MODEL_DIR, "train.py")
    cmd_parts = ["python", train_script_path]
    for key, value in hyperparams.items():
        cmd_parts.extend([f"--{key}", str(value)])

    try:
        process = subprocess.Popen(cmd_parts, cwd=MULTIMODAL_MODEL_DIR)
        with open(PROCESS_PID_PATH, "w") as f:
            f.write(str(process.pid))
        return {"status": "started", "message": "Training lancé en tâche de fond."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/train/status")
def train_status():
    if os.path.exists(TRAIN_STATUS_PATH):
        with open(TRAIN_STATUS_PATH, "r") as f:
            status = json.load(f)
        return status
    elif os.path.exists(PROCESS_PID_PATH):
        try:
            pid = int(open(PROCESS_PID_PATH).read())
            os.kill(pid, 0)
            return {"state": "running"}
        except Exception:
            return {"state": "not running"}
    return {"state": "not started"}
