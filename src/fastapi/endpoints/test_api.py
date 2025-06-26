from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import DictCursor
from passlib.context import CryptContext
import src.fastapi.endpoints.dbcrypt as dbcrypt
app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="src/fastapi/endpoints")

# Configurer le contexte de cryptage
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

# Configuration de la connexion à la base de données
def get_db_connection():
    conn = psycopg2.connect(
        dbname="rakuten_auth",
        user="admin",
        password="admin123",  # Utilisez une variable d'environnement pour le mot de passe
        host="postgres",
        port="5432"
    )
    return conn

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str

class UserInDB(User):
    hashed_password: str

def get_user(username: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return UserInDB(
            username=user['username'],
            full_name=user['full_name'],
            email=user['email'],
            hashed_password=user['hashed_password'],
            disabled=user['disabled'],
            role=user['role']
        )
    return None

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = get_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}

@router.get("/test-login", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/form-login")
async def form_login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    return {"message": f"Hello, {user.username}! You are logged in."}

@router.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.get("/admin-only")
async def admin_only(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"message": "Admin access granted"}

@router.get("/dev-only")
async def dev_only(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "dev":
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"message": "Dev access granted"}

@router.get("/client-only")
async def client_only(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "client":
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"message": "Client access granted"}

@router.get("/test")
def hello():
    return {"message": "Hello, Rakuten World!"}

app.include_router(router)
