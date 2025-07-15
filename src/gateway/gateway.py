from fastapi import FastAPI, Request
from fastapi.responses import Response, RedirectResponse
import httpx
import asyncio
import os
import base64
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit
import re




# 🔒 Charger les identifiants depuis .env
load_dotenv(dotenv_path="/opt/gateway/.env")
API_USER        = os.getenv("RAKUTEN_API_USER", "")
API_PASSWORD_B64 = os.getenv("RAKUTEN_API_PASSWORD_B64", "")
API_PASSWORD    = base64.b64decode(API_PASSWORD_B64).decode() if API_PASSWORD_B64 else None

# 🌍 FastAPI app
app = FastAPI()

# 💾 Jeton d’accès stocké globalement
AUTO_TOKEN: str | None = None

# 🚫 En-têtes “hop-by-hop” à supprimer
EXCLUDED_HEADERS = {
    b"content-encoding",
    b"transfer-encoding",
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailers",
    b"upgrade",
    b"date",
    b"server",
    b"content-length",
}

async def fetch_token_with_retry(retries: int = 10, delay: float = 2.0):
    """
    Tente plusieurs fois de récupérer un JWT depuis /token.
    """
    global AUTO_TOKEN
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://api:8000/token",
                    data={"username": API_USER, "password": API_PASSWORD},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                resp.raise_for_status()
                AUTO_TOKEN = resp.json().get("access_token")
                print(f"✅ Token récupéré (tentative {attempt})")
                return
        except Exception as e:
            print(f"⚠️  Échec auth (tentative {attempt}): {e}")
            await asyncio.sleep(delay)
    print("❌ Échec final de récupération du token après plusieurs essais")

@app.on_event("startup")
async def on_startup():
    await fetch_token_with_retry()


def prepare_headers(raw: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """
    1) Supprime les en-têtes hop-by-hop.
    2) Injecte le Bearer token si disponible.
    """
    # Filtrage
    headers = [(k, v) for k, v in raw if k.lower() not in EXCLUDED_HEADERS]

    # Inject token
    global AUTO_TOKEN
    if AUTO_TOKEN:
        # retirer un éventuel ancien Authorization
        headers = [(k, v) for (k, v) in headers if k.lower() != b"authorization"]
        headers.append((b"authorization", f"Bearer {AUTO_TOKEN}".encode()))

    return headers


@app.api_route(
    "/proxy/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_api(path: str, request: Request):
    """
    Reverse-proxy vers l’API interne sécurisée par JWT.
    """
    upstream_url = f"http://api:8000/{path}"
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=upstream_url,
            headers=prepare_headers(request.headers.raw),
            content=await request.body(),
            follow_redirects=True
        )

    # Filtrer les headers de réponse
    filtered = {
        k: v for k, v in resp.headers.items()
        if k.lower().encode() not in EXCLUDED_HEADERS
    }
    return Response(content=resp.content, status_code=resp.status_code, headers=filtered)


@app.get("/proxy/mlflow")
async def redirect_mlflow_root():
    """
    Redirige proprement /proxy/mlflow vers /proxy/mlflow/.
    """
    return RedirectResponse(url="/proxy/mlflow/")


@app.api_route(
    "/proxy/mlflow/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_mlflow(path: str, request: Request):
    """
    Reverse-proxy vers MLflow, avec --static-prefix /proxy/mlflow activé.
    Tout chemin est préfixé par /proxy/mlflow/ en amont.
    """
    # 1) Construire le chemin complet vers MLflow
    upstream_path = f"/proxy/mlflow/{path}"
    split = urlsplit(str(request.url))
    upstream_url = urlunsplit((
        "http", "mlflow:5005",
        upstream_path,
        split.query,
        ""
    ))

    print(f"🔀 Proxy MLflow → {upstream_url}")

    # 2) Appel upstream
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            upstream_url,
            headers=prepare_headers(request.headers.raw),
            content=await request.body(),
            follow_redirects=True
        )

    # 3) Filtrer les headers de réponse
    filtered = {
        k: v for k, v in resp.headers.items()
        if k.lower().encode() not in EXCLUDED_HEADERS
    }
    return Response(content=resp.content, status_code=resp.status_code, headers=filtered)

@app.get("/proxy/prometheus")
async def redirect_prometheus_root():
    """
    Redirige /proxy/prometheus vers /proxy/prometheus/
    """
    return RedirectResponse(url="/proxy/prometheus/")

@app.api_route(
    "/proxy/prometheus/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_prometheus(path: str, request: Request):
    """
    Reverse-proxy vers Prometheus (pas d’auth JWT ici).
    """
    # Construire le chemin upstream
    src = urlsplit(str(request.url))
    upstream_path = f"/{path}" if path else "/"
    upstream_url = urlunsplit((
        "http",
        "prometheus:9090",
        upstream_path,
        src.query,
        ""
    ))
    print(f"🔀 Proxy Prometheus → {upstream_url}")

    # Appel upstream
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            request.method,
            upstream_url,
            headers=[
                (k, v) for k, v in request.headers.raw
                if k.lower() not in EXCLUDED_HEADERS and k.lower() != b"authorization"
            ],
            content=await request.body(),
            follow_redirects=True,
        )

    # Filtrer les hop-by-hop headers
    filtered = {
        k: v for k, v in resp.headers.items()
        if k.lower().encode() not in EXCLUDED_HEADERS
    }
    return Response(content=resp.content, status_code=resp.status_code, headers=filtered)
