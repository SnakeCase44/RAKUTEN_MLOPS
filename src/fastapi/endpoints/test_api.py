from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Request, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="src/fastapi/endpoints")

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
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

def fake_decode_token(token):
    # Simuler le décodage d'un token JWT
    user = get_user(fake_users_db, token)
    return user

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    user = UserInDB(**user_dict)
    if not form_data.password == "secret":  # Dans un vrai cas, vérifiez le mot de passe haché
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}

@router.get("/test-login", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/form-login")
async def form_login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(fake_users_db, username)
    if not user or password != "secret":  # Remplacez par une vérification de mot de passe haché
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
