from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    admin = "admin"
    operator = "operator"
    viewer = "viewer"


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=4, max_length=128)
    role: Role = Role.viewer
    enabled: bool = True


class UserUpdate(BaseModel):
    role: Optional[Role] = None
    enabled: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=4, max_length=128)


class UserOut(BaseModel):
    username: str
    role: Role
    enabled: bool
    last_login: Optional[str] = None


class ConnectionType(str, Enum):
    rtsp = "rtsp"
    file = "file"
    usb = "usb"


class ConnectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    type: ConnectionType
    source: str = Field(min_length=1, max_length=512)
    default_model_id: str = "person_detector"
    default_stream_id: int = Field(default=1, ge=1)


class ConnectionOut(ConnectionIn):
    id: str
    status: str = "idle"


class StreamStartIn(BaseModel):
    connection_id: Optional[str] = None
    source: Optional[str] = None
    model_id: Optional[str] = None
    rtsp_enabled: bool = True


class StreamModelIn(BaseModel):
    model_id: str


class StreamStatus(BaseModel):
    id: int
    running: bool
    source: Optional[str] = None
    connection_id: Optional[str] = None
    model_id: Optional[str] = None
    fps: float = 0.0
    frames: int = 0
    last_error: Optional[str] = None
    browser_url: str
    rtsp_url: str


class ModelInfo(BaseModel):
    id: str
    name: str
    task: str
    path: str
    labels: Dict[int, str]
    requires_person_detector: bool = False
    function_entrypoint: Optional[str] = None
    description: Optional[str] = None


class ModelFileOut(BaseModel):
    name: str
    size: int
    modified_time: str


class ModelLabelOut(BaseModel):
    id: int
    name: str


class ModelFileDetailOut(ModelFileOut):
    loaded: bool = False
    warmup_done: bool = False
    device: str = ""
    memory_allocated_mb: int = 0
    memory_reserved_mb: int = 0
    labels: List[ModelLabelOut] = Field(default_factory=list)
    error: str = ""


class ModelFunctionIn(BaseModel):
    id: str = Field(min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_\\-]+$")
    name: str = Field(min_length=1, max_length=64)
    task: str = Field(min_length=1, max_length=32)
    entrypoint: str = Field(min_length=3, max_length=256)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    description: str = ""


class ModelFunctionOut(ModelFunctionIn):
    pass


class HistoryLogOut(BaseModel):
    id: str
    time: str
    user: str
    action: str
    target_type: str
    target_id: str = ""
    result: str = "success"
    message: str = ""


class MemoryStatus(BaseModel):
    kind: str = "unknown"
    label: str = "显存"
    total_mb: int = 0
    used_mb: int = 0


class AppSnapshot(BaseModel):
    users: List[UserOut]
    connections: List[ConnectionOut]
    models: List[ModelInfo]
    model_files: List[ModelFileOut]
    model_functions: List[ModelFunctionOut]
    history_logs: List[HistoryLogOut]
    streams: List[StreamStatus]
    memory: MemoryStatus
