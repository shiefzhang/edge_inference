from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
import os
import socket


def detect_lan_ip() -> str:
    configured = os.getenv("RTSP_PUBLIC_HOST")
    if configured:
        return configured
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return os.getenv("MEDIAMTX_HOST", "127.0.0.1")


class Settings(BaseModel):
    app_name: str = "Edge Inference"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    model_logic_dir: Path = Path(__file__).resolve().parent / "func"
    user_functions_dir: Path = Path(__file__).resolve().parent / "user_functions"
    session_secret: str = os.getenv("SESSION_SECRET", "change-me-on-device")
    enable_rtsp_push: bool = os.getenv("ENABLE_RTSP_PUSH", "0") == "1"
    mediamtx_host: str = os.getenv("MEDIAMTX_HOST", "127.0.0.1")
    rtsp_public_host: str = detect_lan_ip()
    mediamtx_port: int = int(os.getenv("MEDIAMTX_PORT", "8554"))
    stream_count: int = int(os.getenv("STREAM_COUNT", "4"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "1280"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    frame_fps: int = int(os.getenv("FRAME_FPS", "20"))
    inference_timeout_seconds: int = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "15"))
    capture_open_timeout_ms: int = int(os.getenv("CAPTURE_OPEN_TIMEOUT_MS", "5000"))
    capture_read_timeout_ms: int = int(os.getenv("CAPTURE_READ_TIMEOUT_MS", "5000"))
    connection_test_timeout_ms: int = int(os.getenv("CONNECTION_TEST_TIMEOUT_MS", "15000"))
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    log_to_console: bool = os.getenv("LOG_TO_CONSOLE", "0") == "1"
    access_log_enabled: bool = os.getenv("ACCESS_LOG_ENABLED", "0") == "1"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.model_logic_dir.mkdir(parents=True, exist_ok=True)
    settings.user_functions_dir.mkdir(parents=True, exist_ok=True)
    return settings
