from __future__ import annotations

import subprocess
import time
import logging

import cv2

logger = logging.getLogger(__name__)
FAST_FFPROBE_TIMEOUT_MS = 3000
FAST_FFPROBE_ANALYZE_US = 500_000
FAST_FFPROBE_PROBE_SIZE = 32_768


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
    fast_timeout_ms = min(timeout_ms, FAST_FFPROBE_TIMEOUT_MS)
    fast = _run_ffprobe(source, fast_timeout_ms, fast=True)
    if fast is None:
        logger.info("connection probe ffprobe unavailable source=%s elapsed_ms=%s", source, round((time.monotonic() - started) * 1000, 2))
        return _probe_opencv_source(source, timeout_ms)
    fast_ok, fast_message, fast_retryable = fast
    if fast_ok:
        return True, fast_message
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    remaining_ms = max(1, timeout_ms - int(elapsed_ms))
    if not fast_retryable or remaining_ms <= 500:
        return False, fast_message
    logger.info("connection probe ffprobe fast failed; retrying full source=%s remaining_ms=%s message=%s", source, remaining_ms, fast_message)
    full = _run_ffprobe(source, remaining_ms, fast=False)
    if full is None:
        return _probe_opencv_source(source, remaining_ms)
    ok, message, _ = full
    return ok, message


def _run_ffprobe(source: str, timeout_ms: int, fast: bool) -> tuple[bool, str, bool] | None:
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
    ]
    if fast:
        command.extend(
            [
                "-analyzeduration",
                str(FAST_FFPROBE_ANALYZE_US),
                "-probesize",
                str(FAST_FFPROBE_PROBE_SIZE),
                "-select_streams",
                "v:0",
            ]
        )
    command.extend(
        [
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
        ]
    )
    try:
        logger.info("connection probe ffprobe start source=%s mode=%s timeout_ms=%s", source, "fast" if fast else "full", timeout_ms)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, timeout_ms / 1000),
        )
    except FileNotFoundError:
        logger.info("connection probe ffprobe missing source=%s mode=%s elapsed_ms=%s", source, "fast" if fast else "full", round((time.monotonic() - started) * 1000, 2))
        return None
    except subprocess.TimeoutExpired:
        logger.info("connection probe ffprobe timeout source=%s mode=%s elapsed_ms=%s", source, "fast" if fast else "full", round((time.monotonic() - started) * 1000, 2))
        return False, f"连接测试超时（{timeout_ms}ms）", fast
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    logger.info("connection probe ffprobe returned source=%s mode=%s returncode=%s elapsed_ms=%s", source, "fast" if fast else "full", result.returncode, elapsed_ms)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "无法打开视频源").strip()
        return False, detail[-160:], fast
    if "video" not in result.stdout.lower():
        return False, "已连接，但未发现视频流", fast
    message = "连接正常，已发现视频流"
    if fast:
        message = f"{message}（快速探测）"
    return True, message, False


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
