from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.fastapi1.endpoints import test_api, prediction_api, api_train, api_evaluate

app = FastAPI(title="Rakuten Multimodal API")

# --- Prometheus Instrumentation CORRIGÉE ---
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True
)

instrumentator.instrument(app).expose(app)

# --- Register routers APRÈS l'instrumentation ---
app.include_router(test_api.router)
app.include_router(api_train.router)
app.include_router(prediction_api.router)
app.include_router(api_evaluate.router)