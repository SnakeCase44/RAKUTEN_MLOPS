from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="src/fastapi/endpoints/")

# Simuler une base de données d'utilisateurs
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "fakehashedsecretadmin",
        "disabled": False,
        "role": "admin"
    },
    "dev": {
        "username": "dev",
        "full_name": "Dev User",
        "email": "dev@example.com",
        "hashed_password": "fakehashedsecretdev",
        "disabled": False,
        "role": "dev"
    },
    "client": {
        "username": "client",
        "full_name": "Client User",
        "email": "client@example.com",
        "hashed_password": "fakehashedsecretclient",
        "disabled": False,
        "role": "client"
    }
}

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str

class UserInDB(User):
    hashed_password: str

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

@router.get("/test-login", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/test-login")
async def test_login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(fake_users_db, username)
    if not user or password != "secret":  # Remplacez par une vérification de mot de passe haché
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    return {"message": f"Hello, {user}! You are logged in."}


@router.get("/test")
def hello():
    return {"message": "Hello, Rakuten World!"}

app.include_router(router)

