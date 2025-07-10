from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response, RedirectResponse
from jose import JWTError, jwt
import httpx
import os

# Auth constants
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"

# App
app = FastAPI()

# Vérification JWT
def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant")

    token = auth.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # tu peux extraire 'role', etc.
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

# Proxy vers API
@app.api_route("/proxy/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_api(path: str, request: Request, user=Depends(verify_token)):
    async with httpx.AsyncClient() as client:
        proxy_url = f"http://api:8000/{path}"
        proxy_req = await client.request(
            method=request.method,
            url=proxy_url,
            headers=request.headers.raw,
            content=await request.body()
        )
        return Response(content=proxy_req.content, status_code=proxy_req.status_code, headers=proxy_req.headers)

# Proxy vers MLflow
@app.api_route("/proxy/mlflow/{path:path}", methods=["GET", "POST"])
async def proxy_mlflow(path: str, request: Request, user=Depends(verify_token)):
    async with httpx.AsyncClient() as client:
        proxy_url = f"http://mlflow:5005/{path}"
        proxy_req = await client.request(
            method=request.method,
            url=proxy_url,
            headers=request.headers.raw,
            content=await request.body()
        )
        return Response(content=proxy_req.content, status_code=proxy_req.status_code, headers=proxy_req.headers)

# Idem pour /proxy/streamlit et /proxy/airflow...
