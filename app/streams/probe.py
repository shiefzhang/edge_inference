from __future__ import annotations

import subprocess
import time
import logging

import cv2

logger = logging.getLogger(__name__)


def probe_video_source(source: str, timeout_ms: int = 4000) -> tuple[bool, str]:
    started = time.monotonic()
    logger.info("connection probe start source=%s timeout_ms=%s", source, timeout_ms)
    if source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        ok, message = _probe_network_source(source, timeout_ms)
    else:
        ok, message = _probe_opencv_source(source, timeout_ms)
    logger.info(
        "connection probe done source=%s ok=%s elapsed_ms=%s message=%s",
        source,
        ok,
        round((time.monotonic() - started) * 1000, 2),
        message,
    )
    return ok, message


def _probe_network_source(source: str, timeout_ms: int) -> tuple[bool, str]:
    started = time.monotonic()
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
        logger.info("connection probe ffprobe start source=%s timeout_ms=%s", source, timeout_ms)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_ms / 1000),
        )
    except FileNotFoundError:
        logger.info("connection probe ffprobe missing source=%s elapsed_ms=%s", source, round((time.monotonic() - started) * 1000, 2))
        return _probe_opencv_source(source, timeout_ms)
    except subprocess.TimeoutExpired:
        logger.info("connection probe ffprobe timeout source=%s elapsed_ms=%s", source, round((time.monotonic() - started) * 1000, 2))
        return False, f"连接测试超时（{timeout_ms}ms）"
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info("connection probe ffprobe returned source=%s returncode=%s elapsed_ms=%s", source, result.returncode, elapsed_ms)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "无法打开视频源").strip()
        return False, detail[-160:]
    if "video" not in result.stdout.lower():
        return False, "已连接，但未发现视频流"
    return True, "连接正常，已发现视频流"


def _probe_opencv_source(source: str, timeout_ms: int) -> tuple[bool, str]:
    started = time.monotonic()
    capture_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    logger.info("connection probe opencv open start source=%s timeout_ms=%s", source, timeout_ms)
    opened = cap.open(capture_source)
    open_elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info("connection probe opencv open returned source=%s opened=%s elapsed_ms=%s", source, opened, open_elapsed_ms)
    if not opened:
        cap.release()
        return False, "无法打开视频源"
    read_started = time.monotonic()
    ok, _ = cap.read()
    logger.info("connection probe opencv read returned source=%s ok=%s elapsed_ms=%s", source, ok, round((time.monotonic() - read_started) * 1000, 2))
    cap.release()
    if not ok:
        return False, "视频源已打开，但读取首帧失败"
    return True, "连接正常，已读取到视频帧"
