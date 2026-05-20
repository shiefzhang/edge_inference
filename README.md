# Edge Inference Control Platform

FastAPI video inference control platform for Jetson-style edge boxes.

## Features

- 4 concurrent video channels.
- 4 independently wrapped YOLO modules.
- Per-channel model switching.
- Browser preview with annotated frames.
- Optional H264 RTSP push through MediaMTX.
- Simple user management with fixed roles: admin, operator, viewer.
- Connection list for RTSP, video file, and USB camera sources.

## Author

- Author: Pyrrhus
- Email: zhangxuefeng@batonsoft.com

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

Default account:

- Username: `admin`
- Password: `admin123`

Change the default password after the first login.

## RTSP Output

Install and run MediaMTX, then set:

```bash
MEDIAMTX_HOST=127.0.0.1
MEDIAMTX_PORT=8554
ENABLE_RTSP_PUSH=1
```

Each active channel pushes to:

- `rtsp://<host>:8554/stream/1`
- `rtsp://<host>:8554/stream/2`
- `rtsp://<host>:8554/stream/3`
- `rtsp://<host>:8554/stream/4`
