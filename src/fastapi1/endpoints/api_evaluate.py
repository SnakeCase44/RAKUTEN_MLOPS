from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
import re

from config import MULTIMODAL_REPORT_PATH

router = APIRouter()
REPORT_PATH = MULTIMODAL_REPORT_PATH

@router.get("/evaluate")
async def evaluate_model():
    if not os.path.exists(REPORT_PATH):
        return JSONResponse(status_code=404, content={"message": "Report file not found."})

    metrics = {}
    try:
        with open(REPORT_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                # Match accuracy line
                if line.lower().startswith("accuracy"):
                    match = re.search(r"\d+\.\d+", line)
                    if match:
                        metrics["accuracy"] = float(match.group())

                # Match macro avg and weighted avg
                elif line.lower().startswith("macro avg") or line.lower().startswith("weighted avg"):
                    parts = re.split(r"\s{2,}", line)
                    if len(parts) >= 4:
                        label = parts[0].lower().replace(" ", "_")  # macro_avg or weighted_avg
                        try:
                            metrics[f"{label}_precision"] = float(parts[1])
                            metrics[f"{label}_recall"] = float(parts[2])
                            metrics[f"{label}_f1"] = float(parts[3])
                        except ValueError:
                            continue  # ignore badly formatted lines

        # Impression dans les logs pour Airflow
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            full_report = f.read()
            print("=== Rapport complet de classification ===")
            print(full_report)

        if not metrics:
            return JSONResponse(status_code=500, content={"message": "No valid metrics found in report."})

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error reading report: {str(e)}"})
