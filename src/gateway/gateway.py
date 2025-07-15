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


'''
# 1) Redirection de base pour Streamlit
@app.get("/proxy/streamlit")
async def redirect_streamlit_root():
    return RedirectResponse(url="/proxy/streamlit/")

# 2) Reverse-proxy Streamlit
@app.api_route(
    "/proxy/streamlit/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_streamlit(path: str, request: Request):
    # normaliser le path ("/" si vide)
    upstream_path = f"/{path}" if path else "/"
    src = urlsplit(str(request.url))
    upstream = urlunsplit((
        "http", "streamlit:8501", upstream_path, src.query, ""
    ))

    print(f"🔀 Proxy Streamlit → {upstream}")
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            upstream,
            headers=prepare_headers(request.headers.raw),
            content=await request.body(),
            follow_redirects=True
        )

    # filtrer les hop-by-hop headers
    out_hdrs = {
        k: v for k, v in resp.headers.items()
        if k.lower().encode() not in EXCLUDED_HEADERS
    }
    return Response(content=resp.content, status_code=resp.status_code, headers=out_hdrs)


# 3) Redirection de base pour Airflow
@app.get("/proxy/airflow")
async def redirect_airflow_root():
    return RedirectResponse(url="/proxy/airflow/")

@app.api_route(
    "/proxy/airflow/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_airflow(path: str, request: Request):
    """
    Reverse-proxy Airflow derrière /proxy/airflow,
    avec rewriting complet des URLs et injection JWT.
    """
    # 1) Choix du chemin upstream
    if request.method == "POST" and path == "":
        # le login POST form par défaut vise "/", on cible /login/
        upstream_path = "/login/"
    else:
        upstream_path = f"/{path}" if path else "/"

    # 2) Construire l'URL upstream (avec query string)
    src = urlsplit(str(request.url))
    upstream_url = urlunsplit((
        "http", "airflow:8080",  # nom du service et port
        upstream_path,
        src.query,
        ""
    ))
    print(f"🔀 Proxy Airflow → {upstream_url}")

    # 3) Appel au serveur Airflow
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            upstream_url,
            headers=prepare_headers(request.headers.raw),
            content=await request.body(),
            follow_redirects=False  # on gère les redirects nous-mêmes
        )

    # 4) Rewriting du body si c'est du HTML
    content = resp.content
    ctype   = resp.headers.get("content-type", "")
    if "text/html" in ctype.lower():
        text = content.decode("utf-8", errors="ignore")

        # 4.1) Réécrit tous les href/src="/..." → "/proxy/airflow/..."
        text = re.sub(
            r'(href|src)="\/(?!proxy\/airflow)([^"]+)"',
            r'\1="/proxy/airflow/\2"',
            text
        )

        # 4.2) Réécrit les form action="/..." → "/proxy/airflow/..."
        text = re.sub(
            r'action="\/(?!proxy\/airflow)([^"]+)"',
            r'action="/proxy/airflow/\1"',
            text
        )

        # 4.3) Réécrit les meta refresh url=/...
        text = re.sub(
            r'url=\s*\/(?!proxy\/airflow)([^;"]+)',
            r'url=/proxy/airflow/\1',
            text
        )

        content = text.encode("utf-8")

    # 5) Filtrer et réécrire les en-têtes de réponse
    filtered = {}
    for k, v in resp.headers.items():
        kl = k.lower().encode()
        if kl in EXCLUDED_HEADERS:
            continue

        # 5.1) Rewrite Location: /foo → /proxy/airflow/foo
        if k.lower() == "location" and v.startswith("/"):
            v = f"/proxy/airflow{v}"

        # 5.2) Rewrite cookie Path=/ → Path=/proxy/airflow/
        if k.lower() == "set-cookie":
            v = re.sub(r"Path=/", "Path=/proxy/airflow/", v, flags=re.IGNORECASE)

        filtered[k] = v

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=filtered
    )
'''

  