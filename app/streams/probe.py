from __future__ import annotations

import subprocess

import cv2


def probe_video_source(source: str, timeout_ms: int = 4000) -> tuple[bool, str]:
    if source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return _probe_network_source(source, timeout_ms)
    return _probe_opencv_source(source, timeout_ms)


def _probe_network_source(source: str, timeout_ms: int) -> tuple[bool, str]:
    timeout_us = str(max(1, timeout_ms) * 1000)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        timeout_us,
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_ms / 1000),
        )
    except FileNotFoundError:
        return _probe_opencv_source(source, timeout_ms)
    except subprocess.TimeoutExpired:
        return False, f"连接测试超时（{timeout_ms}ms）"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "无法打开视频源").strip()
        return False, detail[-160:]
    if "video" not in result.stdout.lower():
        return False, "已连接，但未发现视频流"
    return True, "连接正常，已发现视频流"


def _probe_opencv_source(source: str, timeout_ms: int) -> tuple[bool, str]:
    capture_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    opened = cap.open(capture_source)
    if not opened:
        cap.release()
        return False, "无法打开视频源"
    ok, _ = cap.read()
    cap.release()
    if not ok:
        return False, "视频源已打开，但读取首帧失败"
    return True, "连接正常，已读取到视频帧"
