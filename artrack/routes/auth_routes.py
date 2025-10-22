from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import UserCreate, UserLogin, AuthResponse, UserResponse
from ..auth import register_user, login_user, get_current_user
from pydantic import BaseModel
from typing import Optional
from ..auth import login_with_google

router = APIRouter()

@router.post("/register", response_model=AuthResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        return register_user(db, user_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=AuthResponse)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    try:
        return login_user(db, user_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Login failed: {str(e)}"
        )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user = Depends(get_current_user)):
    """Get current user info"""
    return current_user

class GoogleAuthRequest(BaseModel):
    id_token: str
    device_id: Optional[str] = None
    display_name: Optional[str] = None

@router.post("/google", response_model=AuthResponse)
async def google_login(body: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Login/Register via Firebase Google ID token and return api_key."""
    return login_with_google(db, body.id_token, body.device_id, body.display_name)

@router.post("/refresh")
async def refresh_token(current_user = Depends(get_current_user)):
    """Refresh access token (placeholder - in real app would use refresh token)"""
    from ..auth import create_access_token
    from datetime import timedelta
    
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data={"sub": current_user.email, "user_id": current_user.id},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "expires_in": int(access_token_expires.total_seconds())
    }