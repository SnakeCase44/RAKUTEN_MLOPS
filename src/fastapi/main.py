from fastapi import FastAPI
from src.fastapi.endpoints import test_api

app = FastAPI(title="Rakuten Multimodal API")

app.include_router(test_api.router)

