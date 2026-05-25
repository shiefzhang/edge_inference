import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.dependencies import get_store
from app.security import create_session_token, verify_session_token
from app.store import JsonStore


templates = Jinja2Templates(directory=str(get_settings().base_dir / "templates"))
router = APIRouter()
ops_logger = logging.getLogger("app.ops")


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "-"


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    ops_logger.info("页面访问: 登录页 client=%s", _client_host(request))
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), store: JsonStore = Depends(get_store)):
    settings = get_settings()
    user = store.authenticate(username, password)
    if not user:
        ops_logger.info("login failed: user=%s client=%s", username, _client_host(request))
        store.add_history_log(username, "login", "user", username, result="failed", message="用户名或密码错误")
        return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误"}, status_code=401)
    store.add_history_log(user.username, "login", "user", user.username)
    ops_logger.info("login succeeded: user=%s role=%s client=%s", user.username, user.role.value, _client_host(request))
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("session", create_session_token(user.username, settings.session_secret), httponly=True, samesite="lax")
    return response


@router.post("/logout")
def logout(request: Request):
    settings = get_settings()
    username = verify_session_token(request.cookies.get("session", ""), settings.session_secret)
    ops_logger.info("logout: user=%s client=%s", username or "-", _client_host(request))
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


@router.get("/", response_class=HTMLResponse)
def index(request: Request, store: JsonStore = Depends(get_store)):
    settings = get_settings()
    username = verify_session_token(request.cookies.get("session", ""), settings.session_secret)
    user = store.get_user(username) if username else None
    if not user or not user.enabled:
        ops_logger.info("page opened: unauthorized client=%s", _client_host(request))
        return RedirectResponse("/login", status_code=303)
    ops_logger.info("page opened: monitor user=%s role=%s client=%s", user.username, user.role.value, _client_host(request))
    return templates.TemplateResponse("index.html", {"request": request, "user": user})
