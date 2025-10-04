from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from api.security.jwt_handler import authenticate_user, create_access_token, get_current_active_user, User
from config.settings import JWT_CONFIG

router = APIRouter()

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class UserResponse(BaseModel):
    username: str
    email: str
    full_name: str
    roles: list[str]

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username, "roles": user.roles}, timedelta(minutes=JWT_CONFIG["access_token_expire_minutes"]))
    return {"access_token": token, "token_type": "bearer", "expires_in": JWT_CONFIG["access_token_expire_minutes"] * 60}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return UserResponse(username=current_user.username, email=current_user.email, full_name=current_user.full_name, roles=current_user.roles)

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    return {"message": f"User {current_user.username} logged out"}
