from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
def hello():
    return {"message": "Hello, Rakuten World!"}
