from __future__ import annotations

import time
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies import current_user, get_store, require_roles
from app.config import get_settings
from app.schemas import (
    AppSnapshot,
    ConnectionIn,
    ConnectionOut,
    HistoryLogOut,
    ModelInfo,
    ModelFunctionIn,
    ModelFunctionOut,
    Role,
    StreamModelIn,
    StreamStartIn,
    StreamStatus,
    UserCreate,
    UserOut,
    UserUpdate,
)
from app.store import JsonStore
from app.streams.probe import probe_video_source


router = APIRouter(prefix="/api")


def _safe_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "uploaded"


def _save_upload(file: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(current_user)):
    return user


@router.get("/snapshot", response_model=AppSnapshot)
def snapshot(request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return AppSnapshot(
        users=store.list_users() if user.role == Role.admin else [],
        connections=store.list_connections(),
        models=request.app.state.registry.list_models(),
        model_functions=store.list_model_functions(),
        history_logs=store.list_history_logs(limit=50),
        streams=request.app.state.streams.statuses(),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request, user: UserOut = Depends(current_user)):
    return request.app.state.registry.list_models()


@router.get("/history-logs", response_model=list[HistoryLogOut])
def list_history_logs(limit: int = 200, store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return store.list_history_logs(limit=max(1, min(limit, 500)))


@router.get("/model-functions", response_model=list[ModelFunctionOut])
def list_model_functions(store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return store.list_model_functions()


@router.post("/model-functions", response_model=ModelFunctionOut)
def create_model_function(data: ModelFunctionIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        item = store.create_model_function(data)
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "create", "model_function", item.id, message=item.name)
        return item
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/model-functions/{function_id}", response_model=ModelFunctionOut)
def update_model_function(function_id: str, data: ModelFunctionIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        item = store.update_model_function(function_id, data)
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "update", "model_function", item.id, message=item.name)
        return item
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model function not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/model-functions/{function_id}", status_code=204)
def delete_model_function(function_id: str, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    active = [stream for stream in request.app.state.streams.statuses() if stream.model_id == function_id and stream.running]
    if active:
        raise HTTPException(status_code=400, detail="model function is used by a running stream")
    try:
        store.delete_model_function(function_id)
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "delete", "model_function", function_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model function not found") from exc
    return Response(status_code=204)


@router.post("/model-functions/{function_id}/upload-pt", response_model=ModelFunctionOut)
def upload_model_file(function_id: str, request: Request, file: UploadFile = File(...), store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    if not file.filename or not file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="only .pt files are supported")
    definition = store.get_model_function(function_id)
    if not definition:
        raise HTTPException(status_code=404, detail="model function not found")
    settings = get_settings()
    safe_name = f"{_safe_stem(function_id)}_{_safe_stem(Path(file.filename).stem)}.pt"
    target = settings.models_dir / safe_name
    _save_upload(file, target)
    payload = definition.model_copy(deep=True)
    payload.config["model_path"] = safe_name
    item = store.update_model_function(function_id, ModelFunctionIn(**payload.model_dump()))
    request.app.state.registry.reload(store.list_model_functions())
    store.add_history_log(user.username, "upload_pt", "model_function", function_id, message=safe_name)
    return item


@router.post("/model-functions/{function_id}/upload-code", response_model=ModelFunctionOut)
def upload_model_code(function_id: str, request: Request, file: UploadFile = File(...), store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    if not file.filename or not file.filename.lower().endswith(".py"):
        raise HTTPException(status_code=400, detail="only .py files are supported")
    definition = store.get_model_function(function_id)
    if not definition:
        raise HTTPException(status_code=404, detail="model function not found")
    settings = get_settings()
    module_stem = _safe_stem(function_id)
    target = settings.user_functions_dir / f"{module_stem}.py"
    _save_upload(file, target)
    payload = definition.model_copy(deep=True)
    payload.entrypoint = f"app.user_functions.{module_stem}:build_model"
    try:
        item = store.update_model_function(function_id, ModelFunctionIn(**payload.model_dump()))
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "upload_code", "model_function", function_id, message=target.name)
    except Exception as exc:
        store.update_model_function(function_id, ModelFunctionIn(**definition.model_dump()))
        request.app.state.registry.reload(store.list_model_functions())
        raise HTTPException(status_code=400, detail=f"uploaded python file must expose build_model(definition, models_dir, modules): {exc}") from exc
    return item


@router.get("/users", response_model=list[UserOut])
def list_users(store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    return store.list_users()


@router.post("/users", response_model=UserOut)
def create_user(data: UserCreate, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        created = store.create_user(data)
        store.add_history_log(user.username, "create", "user", created.username, message=created.role.value)
        return created
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.patch("/users/{username}", response_model=UserOut)
def update_user(username: str, data: UserUpdate, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        updated = store.update_user(username, data)
        store.add_history_log(user.username, "update", "user", username, message=updated.role.value)
        return updated
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="user not found") from exc


@router.delete("/users/{username}", status_code=204)
def delete_user(username: str, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        store.delete_user(username)
        store.add_history_log(user.username, "delete", "user", username)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="user not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/connections", response_model=list[ConnectionOut])
def list_connections(store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return store.list_connections()


@router.post("/connections", response_model=ConnectionOut)
def create_connection(data: ConnectionIn, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    created = store.create_connection(data)
    store.add_history_log(user.username, "create", "connection", created.id, message=created.name)
    return created


@router.put("/connections/{connection_id}", response_model=ConnectionOut)
def update_connection(connection_id: str, data: ConnectionIn, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        updated = store.update_connection(connection_id, data)
        store.add_history_log(user.username, "update", "connection", connection_id, message=updated.name)
        return updated
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="connection not found") from exc


@router.delete("/connections/{connection_id}", status_code=204)
def delete_connection(connection_id: str, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        store.delete_connection(connection_id)
        store.add_history_log(user.username, "delete", "connection", connection_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="connection not found") from exc
    return Response(status_code=204)


@router.post("/connections/{connection_id}/test", response_model=ConnectionOut)
def test_connection(connection_id: str, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    connection = store.get_connection(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="connection not found")
    ok, message = probe_video_source(connection.source, timeout_ms=4000)
    status_value = "online" if ok else "offline"
    updated = store.set_connection_status(connection_id, status_value)
    if not ok:
        store.add_history_log(user.username, "test", "connection", connection_id, result="failed", message=message)
        raise HTTPException(status_code=400, detail=message)
    store.add_history_log(user.username, "test", "connection", connection_id, message=message)
    return updated


@router.get("/streams", response_model=list[StreamStatus])
def list_streams(request: Request, user: UserOut = Depends(current_user)):
    return request.app.state.streams.statuses()


@router.post("/streams", response_model=StreamStatus)
def add_stream(request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    stream_id = request.app.state.streams.add_stream()
    store.set_stream_count(stream_id)
    store.add_history_log(user.username, "add_stream", "stream", str(stream_id))
    return request.app.state.streams.status(stream_id)


@router.post("/streams/add", response_model=StreamStatus)
def add_stream_compat(request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    return add_stream(request, store, user)


@router.delete("/streams/{stream_id}", status_code=204)
def delete_stream(stream_id: int, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        request.app.state.streams.remove_stream(stream_id)
        store.set_stream_count(request.app.state.streams.stream_count())
        store.add_history_log(user.username, "delete_stream", "stream", str(stream_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="请先停止通道再删除") from exc
    return Response(status_code=204)


@router.post("/streams/{stream_id}/start", response_model=StreamStatus)
def start_stream(stream_id: int, data: StreamStartIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    connection = store.get_connection(data.connection_id) if data.connection_id else None
    source = data.source or (connection.source if connection else None)
    model_id = data.model_id or (connection.default_model_id if connection else "person_detector")
    if not source:
        raise HTTPException(status_code=400, detail="source or connection_id is required")
    if model_id not in request.app.state.registry.modules:
        raise HTTPException(status_code=400, detail="unknown model")
    try:
        request.app.state.streams.start(stream_id, source, model_id, data.connection_id, data.rtsp_enabled)
        store.add_history_log(user.username, "start", "stream", str(stream_id), message=f"{source} / {model_id}")
        return request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc


@router.post("/streams/{stream_id}/stop", response_model=StreamStatus)
def stop_stream(stream_id: int, request: Request, user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        request.app.state.streams.stop(stream_id)
        request.app.state.store.add_history_log(user.username, "stop", "stream", str(stream_id))
        return request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc


@router.post("/streams/{stream_id}/model", response_model=StreamStatus)
def switch_model(stream_id: int, data: StreamModelIn, request: Request, user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        request.app.state.streams.switch_model(stream_id, data.model_id)
        request.app.state.store.add_history_log(user.username, "switch_model", "stream", str(stream_id), message=data.model_id)
        return request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream or model not found") from exc


@router.get("/video/{stream_id}")
def video_stream(stream_id: int, request: Request, user: UserOut = Depends(current_user)):
    def frames():
        while True:
            jpeg = request.app.state.streams.latest_jpeg(stream_id)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.08)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
