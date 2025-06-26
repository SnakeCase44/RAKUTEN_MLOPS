from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess
from pathlib import Path
from config import API_DIR , MULTIMODAL_MODEL_DIR
import os

from fastapi import APIRouter

router = APIRouter()


templates = Jinja2Templates(directory=os.path.join(API_DIR, "templates"))

# Valeurs par défaut des hyperparamètres
DEFAULT_HYPERPARAMS = {
    "batch_size": 48,
    "max_epochs": 15,
    "lr": 5e-6,
    "patience": 2,
    "dropout": 0.4,
    "weight_decay": 0.01,
    "hidden_size": 512,
    "label_smoothing": 0.15
}

@router.get("/train", response_class=HTMLResponse)
async def get_train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request, "hyperparams": DEFAULT_HYPERPARAMS})

@router.post("/train")
async def train_model(
    batch_size: int = Form(DEFAULT_HYPERPARAMS["batch_size"]),
    max_epochs: int = Form(DEFAULT_HYPERPARAMS["max_epochs"]),
    lr: float = Form(DEFAULT_HYPERPARAMS["lr"]),
    patience: int = Form(DEFAULT_HYPERPARAMS["patience"]),
    dropout: float = Form(DEFAULT_HYPERPARAMS["dropout"]),
    weight_decay: float = Form(DEFAULT_HYPERPARAMS["weight_decay"]),
    hidden_size: int = Form(DEFAULT_HYPERPARAMS["hidden_size"]),
    label_smoothing: float = Form(DEFAULT_HYPERPARAMS["label_smoothing"]),
):
    print(f"Received batch_size: {batch_size}")
    print(f"Received max_epochs: {max_epochs}")
    print(f"Received lr: {lr}")
    print(f"Received patience: {patience}")
    print(f"Received dropout: {dropout}")
    print(f"Received weight_decay: {weight_decay}")
    print(f"Received hidden_size: {hidden_size}")
    print(f"Received label_smoothing: {label_smoothing}")

    try:
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

        # Chemin complet vers train.py
        train_script_path = os.path.join(MULTIMODAL_MODEL_DIR, "train.py") 

        # Construire la commande pour exécuter le script train.py avec les hyperparamètres
        cmd_parts = ["python", train_script_path]
        for key, value in hyperparams.items():
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))
        
        print("Constructed command:", " ".join(cmd_parts))

        # Exécuter le script train.py avec les hyperparamètres
        result = subprocess.run(cmd_parts, capture_output=True, text=True)

        if result.returncode != 0:
            return {"status": "error", "message": f"Training failed: {result.stderr}"}

        return {"status": "success", "message": "Training completed successfully", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)
