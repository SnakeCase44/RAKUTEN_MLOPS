from fastapi import FastAPI
from src.fastapi.endpoints import test_api, api_train

app = FastAPI(title="Rakuten Multimodal API")

app.include_router(test_api.router)
app.include_router(api_train.router)

