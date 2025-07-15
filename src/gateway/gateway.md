from fastapi import FastAPI, Request
from fastapi.responses import Response, RedirectResponse
import httpx
from urllib.parse import urlsplit, urlunsplit

app = FastAPI()
AUTO_TOKEN = None  # Peut être défini via une variable d'environnement ou un fichier de configuration
EXCLUDED = {
    b"content-encoding", b"transfer-encoding", b"connection",
    b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailers", b"upgrade", b"date", b"server", b"content-length"
}

def strip_and_inject(raw_headers):
    # 1) On filtre les en-têtes hop-by-hop
    headers = [(k, v) for (k, v) in raw_headers if k.lower() not in EXCLUDED]
    # 2) On ajoute le Bearer token si besoin
    global AUTO_TOKEN
    if AUTO_TOKEN:
        headers = [(k, v) for (k, v) in headers if k.lower() != b"authorization"]
        headers.append((b"authorization", f"Bearer {AUTO_TOKEN}".encode()))
    return headers

@app.get("/proxy/mlflow")
async def mlflow_root():
    return RedirectResponse("/proxy/mlflow/")

@app.api_route("/proxy/mlflow/{path:path}", methods=["GET","POST","PUT","DELETE"])
async def proxy_mlflow(path: str, request: Request):
    # Compose toujours sous /proxy/mlflow
    upstream_path = f"/proxy/mlflow/{path}"
    split = urlsplit(str(request.url))
    upstream_url = urlunsplit((
        "http", "mlflow:5005",         # host de votre conteneur MLflow
        upstream_path,
        split.query,                   # transmet la query string
        ""
    ))

    print(f"🔀 Proxy vers {upstream_url}")

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            request.method,
            upstream_url,
            headers=strip_and_inject(request.headers.raw),
            content=await request.body(),
            follow_redirects=True
        )

    # Filtrer les en-têtes indésirables de la réponse
    filtered = {
        k: v for k, v in resp.headers.items()
        if k.lower().encode() not in EXCLUDED
    }
    return Response(content=resp.content,
                    status_code=resp.status_code,
                    headers=filtered)


def strip_and_inject(raw: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    # 1) Filtrer les headers proscrits
    headers = [(k, v) for k, v in raw if k.lower() not in EXCLUDED]
    # 2) Injecter le token si dispo
    if AUTO_TOKEN:
        headers = [(k, v) for k, v in headers if k.lower() != b"authorization"]
        headers.append((b"authorization", f"Bearer {AUTO_TOKEN}".encode()))
    return headers
# 🔁 Proxy vers l'API interne sécurisée
@app.api_route("/proxy/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_api(path: str, request: Request):
    async with httpx.AsyncClient() as client:
        proxy_url = f"http://api:8000/{path}"
        proxy_req = await client.request(
            method=request.method,
            url=proxy_url,
            headers=strip_and_inject(request.headers.raw),
            content=await request.body()
        )
        filtered_headers = {
            k: v for k, v in proxy_req.headers.items()
            if k.lower() not in EXCLUDED
        }
        return Response(content=proxy_req.content, status_code=proxy_req.status_code, headers=filtered_headers)
    