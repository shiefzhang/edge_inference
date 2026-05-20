from __future__ import annotations

import cv2


def probe_video_source(source: str, timeout_ms: int = 3000) -> tuple[bool, str]:
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
