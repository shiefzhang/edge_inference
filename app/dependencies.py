from typing import Iterable

from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings
from app.schemas import Role, UserOut
from app.security import verify_session_token
from app.store import JsonStore


def get_store(request: Request) -> JsonStore:
    return request.app.state.store


def current_user(request: Request, store: JsonStore = Depends(get_store)) -> UserOut:
    settings = get_settings()
    token = request.cookies.get("session")
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
    username = verify_session_token(token or "", settings.session_secret)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    user = store.get_user(username)
    if not user or not user.enabled:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user disabled")
    return user


def require_roles(*roles: Role):
    allowed: Iterable[str] = {role.value for role in roles}

    def dependency(user: UserOut = Depends(current_user)) -> UserOut:
        if user.role.value not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="permission denied")
        return user

    return dependency
