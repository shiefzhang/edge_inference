from __future__ import annotations

import time
import re
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies import current_user, get_store, require_roles
from app.config import get_settings
from app.schemas import (
    AppSnapshot,
    ConnectionIn,
    ConnectionOut,
    HistoryLogOut,
    ModelFileOut,
    ModelFileDetailOut,
    ModelInfo,
    ModelTypeIn,
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
from app.system_resources import get_memory_status
from app.model_paths import config_for_display, config_for_runtime, model_dir, model_extension, with_model_extension
from app.models.registry import ModelRegistry
from app.pt_model_cache import clear_model_caches_except, get_pt_model_cache, to_detail


router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
ops_logger = logging.getLogger("app.ops")


def _safe_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "uploaded"


def _save_upload(file: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)


def _active_model_context(request: Request | None = None) -> tuple[str, Path, str]:
    settings = get_settings()
    model_type = getattr(request.app.state, "model_type", None) if request else None
    model_type = str(model_type or "pt")
    extension = model_extension(model_type)
    return model_type, model_dir(settings, model_type), extension


def _list_model_files(model_type: str) -> list[ModelFileOut]:
    settings = get_settings()
    active_dir = model_dir(settings, model_type)
    extension = model_extension(model_type)
    cache = get_pt_model_cache(active_dir, extension)
    files = []
    for path in sorted(active_dir.glob(f"*{extension}"), key=lambda item: item.name.lower()):
        stat = path.stat()
        cached = cache.peek(path)
        files.append(
            ModelFileOut(
                name=path.name,
                size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                loaded=cached.loaded,
                memory_allocated_mb=cached.memory_allocated_mb,
                memory_reserved_mb=cached.memory_reserved_mb,
            )
        )
    return files


def _display_model_functions(items: list[ModelFunctionOut]) -> list[ModelFunctionOut]:
    output = []
    for item in items:
        payload = item.model_dump()
        payload["config"] = config_for_display(payload.get("config") or {})
        output.append(ModelFunctionOut(**payload))
    return output


def _missing_files_for_function(item: ModelFunctionOut, active_dir: Path, extension: str) -> list[str]:
    missing: set[str] = set()
    config = config_for_runtime(item.config, extension)
    candidates = [config.get("model_path"), config.get("human_model_path"), config.get("default_model_path")]
    bindings = config.get("model_bindings")
    if isinstance(bindings, dict):
        candidates.extend(bindings.values())
    for candidate in candidates:
        if not candidate:
            continue
        file_name = with_model_extension(Path(str(candidate)).name, extension)
        if not (active_dir / file_name).exists():
            missing.add(file_name)
    return sorted(missing)


def _missing_model_files(items: list[ModelFunctionOut], active_dir: Path, extension: str) -> list[str]:
    missing: set[str] = set()
    for item in items:
        if not item.enabled:
            continue
        missing.update(_missing_files_for_function(item, active_dir, extension))
    return sorted(missing)


def _model_function_with_enabled(item: ModelFunctionOut, enabled: bool) -> ModelFunctionIn:
    payload = item.model_dump()
    payload["enabled"] = enabled
    return ModelFunctionIn(**payload)


def _detect_logic_function_name(path: Path) -> str | None:
    module_stem = path.stem
    candidates = [module_stem.removeprefix("model_"), module_stem]
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="gbk")
    names = set(re.findall(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source, flags=re.MULTILINE))
    for candidate in candidates:
        if candidate in names:
            return candidate
    return next((name for name in names if not name.startswith("_")), None)


def _bind_primary_model_file(config: dict, file_name: str) -> None:
    config["model_path"] = Path(file_name).stem
    bindings = config.get("model_bindings")
    if not isinstance(bindings, dict):
        return
    classifier_keys = [key for key in bindings if key not in {"human_model", "default_model"}]
    if classifier_keys:
        bindings[classifier_keys[-1]] = Path(file_name).stem


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(current_user)):
    return user


@router.get("/snapshot", response_model=AppSnapshot)
def snapshot(request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    model_type = getattr(request.app.state, "model_type", store.get_model_type())
    return AppSnapshot(
        users=store.list_users() if user.role == Role.admin else [],
        connections=store.list_connections(),
        models=request.app.state.registry.list_models(),
        model_files=_list_model_files(model_type),
        model_functions=_display_model_functions(store.list_model_functions()),
        model_type=model_type,
        history_logs=store.list_history_logs(limit=50),
        streams=request.app.state.streams.statuses(),
        memory=get_memory_status(),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(request: Request, user: UserOut = Depends(current_user)):
    return request.app.state.registry.list_models()


@router.post("/model-type")
def switch_model_type(data: ModelTypeIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    started = time.monotonic()
    current_type = getattr(request.app.state, "model_type", store.get_model_type())
    next_type = data.model_type.lower()
    if next_type == current_type:
        return {"model_type": current_type, "changed": False}
    settings = get_settings()
    active_dir = model_dir(settings, next_type)
    active_extension = model_extension(next_type)
    model_functions = store.list_model_functions()
    disabled_functions: list[str] = []
    for item in model_functions:
        if not item.enabled:
            continue
        missing_files = _missing_files_for_function(item, active_dir, active_extension)
        if not missing_files:
            continue
        store.update_model_function(item.id, _model_function_with_enabled(item, False))
        disabled_functions.append(f"{item.id}({', '.join(missing_files)})")
        logger.warning(
            "model function disabled during type switch: id=%s from=%s to=%s missing=%s",
            item.id,
            current_type,
            next_type,
            missing_files,
        )
    if disabled_functions:
        model_functions = store.list_model_functions()
    request.app.state.streams.close()
    cache = get_pt_model_cache(active_dir, active_extension)
    registry = ModelRegistry(model_functions, active_dir)
    try:
        registry.load_all()
    except Exception as exc:
        registry.close()
        logger.exception("model type switch failed from=%s to=%s dir=%s", current_type, next_type, active_dir)
        raise HTTPException(status_code=400, detail=f"{next_type.upper()} 模型加载失败：{exc}") from exc
    request.app.state.registry.close()
    clear_model_caches_except(active_dir, active_extension)
    store.set_model_type(next_type)
    request.app.state.model_type = next_type
    request.app.state.models_dir = active_dir
    request.app.state.model_extension = active_extension
    request.app.state.pt_models = cache
    request.app.state.registry = registry
    request.app.state.streams.registry = registry
    for worker in request.app.state.streams.workers.values():
        worker.registry = registry
    history_message = f"{current_type}->{next_type}"
    if disabled_functions:
        history_message += f"; disabled: {'; '.join(disabled_functions)}"
    store.add_history_log(user.username, "switch_model_type", "model", next_type, message=history_message)
    ops_logger.info(
        "model type switched: by=%s from=%s to=%s dir=%s disabled=%s elapsed_ms=%s",
        user.username,
        current_type,
        next_type,
        active_dir,
        disabled_functions,
        round((time.monotonic() - started) * 1000, 2),
    )
    return {"model_type": next_type, "changed": True}


@router.get("/model-files", response_model=list[ModelFileOut])
def list_model_files(request: Request, user: UserOut = Depends(current_user)):
    return _list_model_files(getattr(request.app.state, "model_type", "pt"))


@router.post("/model-files", response_model=ModelFileOut)
def upload_model_pt_file(request: Request, file: UploadFile = File(...), store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    model_type, active_dir, extension = _active_model_context(request)
    if not file.filename or not file.filename.lower().endswith(extension):
        raise HTTPException(status_code=400, detail=f"only {extension} files are supported")
    safe_name = f"{_safe_stem(Path(file.filename).stem)}{extension}"
    target = active_dir / safe_name
    _save_upload(file, target)
    store.add_history_log(user.username, "upload_pt", "model_file", safe_name, message=safe_name)
    stat = target.stat()
    cached = get_pt_model_cache(active_dir, extension).peek(target)
    request.app.state.registry.reload(store.list_model_functions())
    ops_logger.info("model file uploaded: type=%s file=%s by=%s", model_type, safe_name, user.username)
    return ModelFileOut(
        name=target.name,
        size=stat.st_size,
        modified_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        loaded=cached.loaded,
        memory_allocated_mb=cached.memory_allocated_mb,
        memory_reserved_mb=cached.memory_reserved_mb,
    )


@router.get("/model-files/{file_name}/detail", response_model=ModelFileDetailOut)
def model_file_detail(file_name: str, request: Request, user: UserOut = Depends(current_user)):
    started = time.monotonic()
    _, active_dir, extension = _active_model_context(request)
    target = active_dir / with_model_extension(Path(file_name).name, extension)
    logger.info("model file detail requested by=%s file=%s", user.username, target.name)
    if not target.exists() or target.suffix.lower() != extension:
        logger.info("model file detail missing by=%s file=%s", user.username, target.name)
        raise HTTPException(status_code=404, detail="model file not found")
    item = get_pt_model_cache(active_dir, extension).detail(target, warmup=False)
    detail = to_detail(item)
    logger.info(
        "model file detail returned by=%s file=%s loaded=%s warmup=%s labels=%s error=%s elapsed_ms=%s",
        user.username,
        target.name,
        detail.loaded,
        detail.warmup_done,
        len(detail.labels),
        detail.error or "",
        round((time.monotonic() - started) * 1000, 2),
    )
    return detail


@router.get("/history-logs", response_model=list[HistoryLogOut])
def list_history_logs(limit: int = 200, store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return store.list_history_logs(limit=max(1, min(limit, 500)))


@router.get("/model-functions", response_model=list[ModelFunctionOut])
def list_model_functions(store: JsonStore = Depends(get_store), user: UserOut = Depends(current_user)):
    return _display_model_functions(store.list_model_functions())


@router.post("/model-functions", response_model=ModelFunctionOut)
def create_model_function(data: ModelFunctionIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    if data.enabled:
        model_type, active_dir, extension = _active_model_context(request)
        missing_files = _missing_files_for_function(ModelFunctionOut(**data.model_dump()), active_dir, extension)
        if missing_files:
            raise HTTPException(status_code=400, detail=f"{model_type.upper()} 目录缺少模型文件：{', '.join(missing_files)}。请先上传模型后再启用。")
    try:
        item = store.create_model_function(data)
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "create", "model_function", item.id, message=item.name)
        return _display_model_functions([item])[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/model-functions/{function_id}", response_model=ModelFunctionOut)
def update_model_function(function_id: str, data: ModelFunctionIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    try:
        existing = store.get_model_function(function_id)
        if not existing:
            raise KeyError(function_id)
        if data.enabled:
            model_type, active_dir, extension = _active_model_context(request)
            missing_files = _missing_files_for_function(ModelFunctionOut(**data.model_dump()), active_dir, extension)
            if missing_files:
                raise HTTPException(status_code=400, detail=f"{model_type.upper()} 目录缺少模型文件：{', '.join(missing_files)}。请先上传模型后再启用。")
        item = store.update_model_function(function_id, data)
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "update", "model_function", item.id, message=item.name)
        return _display_model_functions([item])[0]
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
    _, active_dir, extension = _active_model_context(request)
    if not file.filename or not file.filename.lower().endswith(extension):
        raise HTTPException(status_code=400, detail=f"only {extension} files are supported")
    definition = store.get_model_function(function_id)
    if not definition:
        raise HTTPException(status_code=404, detail="model function not found")
    safe_name = f"{_safe_stem(function_id)}_{_safe_stem(Path(file.filename).stem)}{extension}"
    target = active_dir / safe_name
    _save_upload(file, target)
    payload = definition.model_copy(deep=True)
    _bind_primary_model_file(payload.config, safe_name)
    item = store.update_model_function(function_id, ModelFunctionIn(**payload.model_dump()))
    request.app.state.registry.reload(store.list_model_functions())
    store.add_history_log(user.username, "upload_pt", "model_function", function_id, message=safe_name)
    payload = item.model_dump()
    payload["config"] = config_for_display(payload.get("config") or {})
    return ModelFunctionOut(**payload)


@router.post("/model-functions/{function_id}/upload-code", response_model=ModelFunctionOut)
def upload_model_code(function_id: str, request: Request, file: UploadFile = File(...), store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    if not file.filename or not file.filename.lower().endswith(".py"):
        raise HTTPException(status_code=400, detail="only .py files are supported")
    definition = store.get_model_function(function_id)
    if not definition:
        raise HTTPException(status_code=404, detail="model function not found")
    settings = get_settings()
    module_stem = _safe_stem(Path(file.filename).stem)
    if not module_stem.startswith("model_"):
        module_stem = f"model_{module_stem}"
    target = settings.model_logic_dir / f"{module_stem}.py"
    _save_upload(file, target)
    payload = definition.model_copy(deep=True)
    payload.entrypoint = "app.model_functions:build_func_model"
    payload.config["logic_module"] = f"app.func.{module_stem}"
    detected = _detect_logic_function_name(target)
    if detected:
        payload.config["logic_function"] = detected
    try:
        item = store.update_model_function(function_id, ModelFunctionIn(**payload.model_dump()))
        request.app.state.registry.reload(store.list_model_functions())
        store.add_history_log(user.username, "upload_code", "model_function", function_id, message=target.name)
    except Exception as exc:
        store.update_model_function(function_id, ModelFunctionIn(**definition.model_dump()))
        request.app.state.registry.reload(store.list_model_functions())
        raise HTTPException(status_code=400, detail=f"uploaded python file must expose a callable model logic function: {exc}") from exc
    return _display_model_functions([item])[0]


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
    started = time.monotonic()
    logger.info("connection test requested by=%s id=%s", user.username, connection_id)
    step_started = time.monotonic()
    connection = store.get_connection(connection_id)
    logger.info("connection test load config id=%s found=%s elapsed_ms=%s", connection_id, bool(connection), round((time.monotonic() - step_started) * 1000, 2))
    if not connection:
        raise HTTPException(status_code=404, detail="connection not found")
    step_started = time.monotonic()
    ok, message = probe_video_source(connection.source, timeout_ms=get_settings().connection_test_timeout_ms)
    logger.info("connection test probe id=%s ok=%s elapsed_ms=%s", connection_id, ok, round((time.monotonic() - step_started) * 1000, 2))
    status_value = "online" if ok else "offline"
    step_started = time.monotonic()
    updated = store.set_connection_status(connection_id, status_value)
    logger.info("connection test update status id=%s status=%s elapsed_ms=%s", connection_id, status_value, round((time.monotonic() - step_started) * 1000, 2))
    step_started = time.monotonic()
    if not ok:
        store.add_history_log(user.username, "test", "connection", connection_id, result="failed", message=message)
        logger.info(
            "connection test returned id=%s ok=%s total_elapsed_ms=%s history_elapsed_ms=%s message=%s",
            connection_id,
            ok,
            round((time.monotonic() - started) * 1000, 2),
            round((time.monotonic() - step_started) * 1000, 2),
            message,
        )
        raise HTTPException(status_code=400, detail=message)
    store.add_history_log(user.username, "test", "connection", connection_id, message=message)
    logger.info(
        "connection test returned id=%s ok=%s total_elapsed_ms=%s history_elapsed_ms=%s message=%s",
        connection_id,
        ok,
        round((time.monotonic() - started) * 1000, 2),
        round((time.monotonic() - step_started) * 1000, 2),
        message,
    )
    return updated


@router.get("/streams", response_model=list[StreamStatus])
def list_streams(request: Request, user: UserOut = Depends(current_user)):
    return request.app.state.streams.statuses()


@router.post("/streams", response_model=StreamStatus)
def add_stream(request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin))):
    stream_id = request.app.state.streams.add_stream()
    store.set_stream_count(stream_id)
    store.add_history_log(user.username, "add_stream", "stream", str(stream_id))
    ops_logger.info("stream added: stream=%s by=%s", stream_id, user.username)
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
        ops_logger.info("stream deleted: stream=%s by=%s", stream_id, user.username)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="请先停止通道再删除") from exc
    return Response(status_code=204)


@router.post("/streams/{stream_id}/start", response_model=StreamStatus)
def start_stream(stream_id: int, data: StreamStartIn, request: Request, store: JsonStore = Depends(get_store), user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    connection = store.get_connection(data.connection_id) if data.connection_id else None
    source = data.source or (connection.source if connection else None)
    model_id = data.model_id
    if model_id is None:
        model_id = connection.default_model_id if connection else "person_detector"
    if model_id == "":
        model_id = None
    if not source:
        raise HTTPException(status_code=400, detail="source or connection_id is required")
    if model_id and model_id not in request.app.state.registry.modules:
        raise HTTPException(status_code=400, detail="unknown model")
    try:
        ops_logger.info("stream start requested: stream=%s by=%s source=%s model=%s", stream_id, user.username, source, model_id or "none")
        request.app.state.streams.start(stream_id, source, model_id, data.connection_id, data.rtsp_enabled)
        store.add_history_log(user.username, "start", "stream", str(stream_id), message=f"{source} / {model_id or 'none'}")
        ops_logger.info("stream started: stream=%s by=%s model=%s rtsp=%s", stream_id, user.username, model_id or "none", data.rtsp_enabled)
        return request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc


@router.post("/streams/{stream_id}/stop", response_model=StreamStatus)
def stop_stream(stream_id: int, request: Request, user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        started = time.monotonic()
        before = request.app.state.streams.status(stream_id)
        logger.info("api stop stream %s requested by=%s running=%s frames=%s", stream_id, user.username, before.running, before.frames)
        request.app.state.streams.stop(stream_id)
        request.app.state.store.add_history_log(user.username, "stop", "stream", str(stream_id))
        after = request.app.state.streams.status(stream_id)
        ops_logger.info("stream stopped: stream=%s by=%s was_running=%s frames=%s elapsed_ms=%s", stream_id, user.username, before.running, before.frames, round((time.monotonic() - started) * 1000, 2))
        logger.info(
            "api stop stream %s returned running=%s frames=%s elapsed_ms=%s",
            stream_id,
            after.running,
            after.frames,
            round((time.monotonic() - started) * 1000, 2),
        )
        return after
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc


@router.post("/streams/{stream_id}/model", response_model=StreamStatus)
def switch_model(stream_id: int, data: StreamModelIn, request: Request, user: UserOut = Depends(require_roles(Role.admin, Role.operator))):
    try:
        request.app.state.streams.switch_model(stream_id, data.model_id)
        request.app.state.store.add_history_log(user.username, "switch_model", "stream", str(stream_id), message=data.model_id)
        ops_logger.info("stream model switched: stream=%s by=%s model=%s", stream_id, user.username, data.model_id)
        return request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream or model not found") from exc


@router.get("/video/{stream_id}")
def video_stream(stream_id: int, request: Request, user: UserOut = Depends(current_user)):
    def frames():
        settings = get_settings()
        frame_interval_seconds = max(0, settings.mjpeg_frame_interval_ms) / 1000
        logger.info("video stream %s opened by=%s mjpeg_frame_interval_ms=%s", stream_id, user.username, settings.mjpeg_frame_interval_ms)
        yielded = 0
        misses = 0
        started = time.monotonic()
        last_log = started
        fetch_total_ms = 0.0
        yield_total_ms = 0.0
        sleep_total_ms = 0.0
        max_fetch_ms = 0.0
        max_yield_ms = 0.0
        max_sleep_ms = 0.0
        perf_samples = 0
        yielded_samples = 0
        try:
            while request.app.state.streams.is_running(stream_id):
                step_started = time.perf_counter()
                jpeg = request.app.state.streams.latest_jpeg(stream_id)
                fetch_ms = round((time.perf_counter() - step_started) * 1000, 2)
                fetch_total_ms += fetch_ms
                max_fetch_ms = max(max_fetch_ms, fetch_ms)
                if jpeg:
                    yielded += 1
                    step_started = time.perf_counter()
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    yield_ms = round((time.perf_counter() - step_started) * 1000, 2)
                    yield_total_ms += yield_ms
                    max_yield_ms = max(max_yield_ms, yield_ms)
                    yielded_samples += 1
                else:
                    misses += 1
                perf_samples += 1
                now = time.monotonic()
                if now - last_log >= 5:
                    status = request.app.state.streams.status(stream_id)
                    logger.info(
                        "video stream %s perf by=%s yielded=%s misses=%s samples=%s worker_running=%s frames=%s fps=%s avg_fetch_ms=%.2f avg_yield_ms=%.2f avg_sleep_ms=%.2f max_fetch_ms=%.2f max_yield_ms=%.2f max_sleep_ms=%.2f configured_sleep_ms=%s last_error=%s",
                        stream_id,
                        user.username,
                        yielded,
                        misses,
                        perf_samples,
                        status.running,
                        status.frames,
                        status.fps,
                        fetch_total_ms / perf_samples if perf_samples else 0.0,
                        yield_total_ms / yielded_samples if yielded_samples else 0.0,
                        sleep_total_ms / perf_samples if perf_samples else 0.0,
                        max_fetch_ms,
                        max_yield_ms,
                        max_sleep_ms,
                        settings.mjpeg_frame_interval_ms,
                        status.last_error or "",
                    )
                    fetch_total_ms = 0.0
                    yield_total_ms = 0.0
                    sleep_total_ms = 0.0
                    max_fetch_ms = 0.0
                    max_yield_ms = 0.0
                    max_sleep_ms = 0.0
                    perf_samples = 0
                    yielded_samples = 0
                    last_log = now
                step_started = time.perf_counter()
                time.sleep(frame_interval_seconds)
                sleep_ms = round((time.perf_counter() - step_started) * 1000, 2)
                sleep_total_ms += sleep_ms
                max_sleep_ms = max(max_sleep_ms, sleep_ms)
        finally:
            elapsed = round((time.monotonic() - started) * 1000, 2)
            try:
                status = request.app.state.streams.status(stream_id)
                logger.info(
                    "video stream %s closed by=%s yielded=%s misses=%s elapsed_ms=%s worker_running=%s frames=%s fps=%s last_error=%s",
                    stream_id,
                    user.username,
                    yielded,
                    misses,
                    elapsed,
                    status.running,
                    status.frames,
                    status.fps,
                    status.last_error or "",
                )
            except KeyError:
                logger.info("video stream %s closed by=%s yielded=%s misses=%s elapsed_ms=%s status=missing", stream_id, user.username, yielded, misses, elapsed)

    try:
        request.app.state.streams.status(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="stream not found") from exc

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/client-log")
def client_log(payload: dict = Body(...), user: UserOut = Depends(current_user)):
    event = str(payload.get("event") or "client")
    logger.info("client log by=%s event=%s payload=%s", user.username, event, payload)
    return {"ok": True}
