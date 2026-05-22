from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np

from app.config import get_settings


class RtspPublisher:
    def __init__(self, stream_id: int, width: int, height: int, fps: int) -> None:
        settings = get_settings()
        self.stream_id = stream_id
        self.url = f"rtsp://{settings.mediamtx_host}:{settings.mediamtx_port}/stream/{stream_id}"
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        settings = get_settings()
        if not settings.enable_rtsp_push or self.process:
            return
        command = [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-g",
            str(max(1, self.fps)),
            "-keyint_min",
            str(max(1, self.fps)),
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-rtsp_transport",
            "tcp",
            "-pkt_size",
            "1200",
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "rtsp",
            self.url,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if not self.process or not self.process.stdin:
            return
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.stop()

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.stdin:
            self.process.stdin.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None
