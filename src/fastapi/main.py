from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.fastapi.endpoints import test_api, prediction_api, api_train, api_evaluate

app = FastAPI(
    title="Rakuten Multimodal API",
    root_path="/api"
)

# --- Prometheus Instrumentation ---
Instrumentator().instrument(app).expose(app)

# --- Register routers ---
app.include_router(test_api.router)
app.include_router(api_train.router)
app.include_router(prediction_api.router)
app.include_router(api_evaluate.router)
