from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os


class Settings(BaseModel):
    app_name: str = "Edge Inference"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    user_functions_dir: Path = Path(__file__).resolve().parent / "user_functions"
    session_secret: str = os.getenv("SESSION_SECRET", "change-me-on-device")
    enable_rtsp_push: bool = os.getenv("ENABLE_RTSP_PUSH", "0") == "1"
    mediamtx_host: str = os.getenv("MEDIAMTX_HOST", "127.0.0.1")
    rtsp_public_host: str = os.getenv("RTSP_PUBLIC_HOST", os.getenv("MEDIAMTX_HOST", "127.0.0.1"))
    mediamtx_port: int = int(os.getenv("MEDIAMTX_PORT", "8554"))
    stream_count: int = 4
    frame_width: int = int(os.getenv("FRAME_WIDTH", "1280"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    frame_fps: int = int(os.getenv("FRAME_FPS", "20"))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.user_functions_dir.mkdir(parents=True, exist_ok=True)
    return settings
