from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.config import get_settings
from app.models.registry import ModelRegistry
from app.store import JsonStore
from app.streams.manager import StreamManager
from app.web import router as web_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = JsonStore()
    app.state.registry = ModelRegistry(app.state.store.list_model_functions(), settings.models_dir)
    app.state.streams = StreamManager(app.state.registry, app.state.store.get_stream_count())
    yield
    app.state.streams.close()
    app.state.registry.close()


app = FastAPI(title="Edge Inference Control Platform", lifespan=lifespan)
settings = get_settings()
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")
app.include_router(api_router)
app.include_router(web_router)
