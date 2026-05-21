from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.schemas import ConnectionIn, ConnectionOut, HistoryLogOut, ModelFunctionIn, ModelFunctionOut, UserCreate, UserOut, UserUpdate
from app.security import hash_password, verify_password


class JsonStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.path = path or settings.data_dir / "state.json"
        self._lock = threading.RLock()
        self._state = {"users": {}, "connections": {}, "model_functions": {}, "history_logs": [], "stream_count": get_settings().stream_count}
        self._load()
        self._ensure_defaults()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            self._state = json.load(handle)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def _ensure_defaults(self) -> None:
        with self._lock:
            if "admin" not in self._state["users"]:
                self._state["users"]["admin"] = {
                    "username": "admin",
                    "password_hash": hash_password("admin123"),
                    "role": "admin",
                    "enabled": True,
                    "last_login": None,
                }
            self._state.setdefault("connections", {})
            self._state.setdefault("model_functions", {})
            self._state.setdefault("history_logs", [])
            self._state.setdefault("stream_count", get_settings().stream_count)
            self._ensure_default_model_functions()
            if not self._state["connections"]:
                self.create_connection(
                    ConnectionIn(
                        name="Local USB Camera 0",
                        type="usb",
                        source="0",
                        default_model_id="person_detector",
                        default_stream_id=1,
                    )
                )
            self._save()

    def _ensure_default_model_functions(self) -> None:
        defaults = [
            ModelFunctionIn(
                id="general_detector",
                name="通用检测",
                task="detect",
                entrypoint="app.model_functions:build_yolo_detector",
                config={"model_path": "yolo11n.pt", "conf": 0.35},
                description="单模型通用目标检测函数",
            ),
            ModelFunctionIn(
                id="person_detector",
                name="人员/设备检测",
                task="detect",
                entrypoint="app.model_functions:build_yolo_detector",
                config={
                    "model_path": "05person_best11m.pt",
                    "conf": 0.35,
                    "allowed_labels": ["person", "cranebody", "excavator", "hook", "othervehicle"],
                },
                description="人员和工地设备检测函数",
            ),
            ModelFunctionIn(
                id="helmet_classifier",
                name="安全帽分类",
                task="classify",
                entrypoint="app.model_functions:build_person_crop_classifier",
                config={
                    "model_path": "human_hat_cls_v4_bestm.pt",
                    "person_detector_id": "person_detector",
                    "pass_labels": ["Bluehelmet", "Otherhelmet", "Redhelmet"],
                    "fail_labels": ["Nohelmet"],
                    "neutral_labels": ["Unclear"],
                },
                description="先进行人员检测，再裁剪人框做安全帽分类",
            ),
            ModelFunctionIn(
                id="vest_classifier",
                name="反光衣分类",
                task="classify",
                entrypoint="app.model_functions:build_person_crop_classifier",
                config={
                    "model_path": "human_vest_cls_best11m.pt",
                    "person_detector_id": "person_detector",
                    "pass_labels": ["Vest"],
                    "fail_labels": ["Novest"],
                    "neutral_labels": ["Other"],
                },
                description="先进行人员检测，再裁剪人框做反光衣分类",
            ),
        ]
        for item in defaults:
            self._state["model_functions"].setdefault(item.id, item.model_dump())

    def authenticate(self, username: str, password: str) -> Optional[UserOut]:
        with self._lock:
            user = self._state["users"].get(username)
            if not user or not user.get("enabled"):
                return None
            if not verify_password(password, user.get("password_hash", "")):
                return None
            user["last_login"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return self._user_out(user)

    def get_user(self, username: str) -> Optional[UserOut]:
        with self._lock:
            user = self._state["users"].get(username)
            return self._user_out(user) if user else None

    def list_users(self) -> List[UserOut]:
        with self._lock:
            return [self._user_out(user) for user in self._state["users"].values()]

    def create_user(self, data: UserCreate) -> UserOut:
        with self._lock:
            if data.username in self._state["users"]:
                raise ValueError("user already exists")
            user = {
                "username": data.username,
                "password_hash": hash_password(data.password),
                "role": data.role.value,
                "enabled": data.enabled,
                "last_login": None,
            }
            self._state["users"][data.username] = user
            self._save()
            return self._user_out(user)

    def update_user(self, username: str, data: UserUpdate) -> UserOut:
        with self._lock:
            user = self._state["users"].get(username)
            if not user:
                raise KeyError(username)
            if data.role is not None:
                user["role"] = data.role.value
            if data.enabled is not None:
                user["enabled"] = data.enabled
            if data.password is not None:
                user["password_hash"] = hash_password(data.password)
            self._save()
            return self._user_out(user)

    def delete_user(self, username: str) -> None:
        with self._lock:
            if username == "admin":
                raise ValueError("default admin cannot be deleted")
            if username not in self._state["users"]:
                raise KeyError(username)
            del self._state["users"][username]
            self._save()

    def list_connections(self) -> List[ConnectionOut]:
        with self._lock:
            return [ConnectionOut(**conn) for conn in self._state["connections"].values()]

    def get_connection(self, connection_id: str) -> Optional[ConnectionOut]:
        with self._lock:
            item = self._state["connections"].get(connection_id)
            return ConnectionOut(**item) if item else None

    def create_connection(self, data: ConnectionIn) -> ConnectionOut:
        with self._lock:
            connection_id = uuid.uuid4().hex[:12]
            item = {"id": connection_id, "status": "idle", **data.model_dump()}
            self._state["connections"][connection_id] = item
            self._save()
            return ConnectionOut(**item)

    def update_connection(self, connection_id: str, data: ConnectionIn) -> ConnectionOut:
        with self._lock:
            if connection_id not in self._state["connections"]:
                raise KeyError(connection_id)
            item = {"id": connection_id, "status": self._state["connections"][connection_id].get("status", "idle"), **data.model_dump()}
            self._state["connections"][connection_id] = item
            self._save()
            return ConnectionOut(**item)

    def delete_connection(self, connection_id: str) -> None:
        with self._lock:
            if connection_id not in self._state["connections"]:
                raise KeyError(connection_id)
            del self._state["connections"][connection_id]
            self._save()

    def set_connection_status(self, connection_id: str, status: str) -> ConnectionOut:
        with self._lock:
            if connection_id not in self._state["connections"]:
                raise KeyError(connection_id)
            self._state["connections"][connection_id]["status"] = status
            self._save()
            return ConnectionOut(**self._state["connections"][connection_id])

    def list_model_functions(self) -> List[ModelFunctionOut]:
        with self._lock:
            return [ModelFunctionOut(**item) for item in self._state["model_functions"].values()]

    def get_model_function(self, function_id: str) -> Optional[ModelFunctionOut]:
        with self._lock:
            item = self._state["model_functions"].get(function_id)
            return ModelFunctionOut(**item) if item else None

    def create_model_function(self, data: ModelFunctionIn) -> ModelFunctionOut:
        with self._lock:
            if data.id in self._state["model_functions"]:
                raise ValueError("model function already exists")
            self._state["model_functions"][data.id] = data.model_dump()
            self._save()
            return ModelFunctionOut(**self._state["model_functions"][data.id])

    def update_model_function(self, function_id: str, data: ModelFunctionIn) -> ModelFunctionOut:
        with self._lock:
            if function_id not in self._state["model_functions"]:
                raise KeyError(function_id)
            if data.id != function_id and data.id in self._state["model_functions"]:
                raise ValueError("model function id already exists")
            item = data.model_dump()
            if data.id != function_id:
                del self._state["model_functions"][function_id]
            self._state["model_functions"][data.id] = item
            self._save()
            return ModelFunctionOut(**item)

    def delete_model_function(self, function_id: str) -> None:
        with self._lock:
            if function_id not in self._state["model_functions"]:
                raise KeyError(function_id)
            del self._state["model_functions"][function_id]
            self._save()

    def add_history_log(
        self,
        user: str,
        action: str,
        target_type: str,
        target_id: str = "",
        result: str = "success",
        message: str = "",
    ) -> HistoryLogOut:
        with self._lock:
            item = {
                "id": uuid.uuid4().hex[:12],
                "time": datetime.now(timezone.utc).isoformat(),
                "user": user,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "result": result,
                "message": message,
            }
            logs = self._state.setdefault("history_logs", [])
            logs.insert(0, item)
            del logs[500:]
            self._save()
            return HistoryLogOut(**item)

    def list_history_logs(self, limit: int = 200) -> List[HistoryLogOut]:
        with self._lock:
            logs = self._state.setdefault("history_logs", [])
            return [HistoryLogOut(**item) for item in logs[:limit]]

    def get_stream_count(self) -> int:
        with self._lock:
            return int(self._state.get("stream_count", get_settings().stream_count))

    def set_stream_count(self, count: int) -> int:
        with self._lock:
            self._state["stream_count"] = max(1, int(count))
            self._save()
            return self._state["stream_count"]

    @staticmethod
    def _user_out(user: Dict) -> UserOut:
        return UserOut(
            username=user["username"],
            role=user["role"],
            enabled=user["enabled"],
            last_login=user.get("last_login"),
        )
