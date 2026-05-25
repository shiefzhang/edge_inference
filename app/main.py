from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.config import get_settings
from app.models.registry import ModelRegistry
from app.model_paths import model_dir, model_extension
from app.pt_model_cache import get_pt_model_cache
from app.store import JsonStore
from app.streams.manager import StreamManager
from app.web import router as web_router


settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)


class ConsoleOpsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "app.ops"


console_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
log_handlers: list[logging.Handler] = [
    RotatingFileHandler(
        settings.data_dir / "server.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
]
if settings.log_to_console:
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ConsoleOpsFilter())
    console_handler.setFormatter(console_formatter)
    log_handlers.append(console_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=log_handlers,
)
logging.getLogger("uvicorn.access").disabled = not settings.access_log_enabled


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = JsonStore()
    model_type = app.state.store.get_model_type()
    active_models_dir = model_dir(settings, model_type)
    active_extension = model_extension(model_type)
    app.state.model_type = model_type
    app.state.models_dir = active_models_dir
    app.state.model_extension = active_extension
    app.state.pt_models = get_pt_model_cache(active_models_dir, active_extension)
    app.state.registry = ModelRegistry(app.state.store.list_model_functions(), active_models_dir)
    app.state.registry.preload_all()
    app.state.streams = StreamManager(app.state.registry, app.state.store.get_stream_count())
    yield
    app.state.streams.close()
    app.state.registry.close()


app = FastAPI(title="Edge Inference Control Platform", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")
app.include_router(api_router)
app.include_router(web_router)
