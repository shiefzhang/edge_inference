# Edge Inference 视频推理管控平台

Edge Inference 是面向 Jetson 和边缘盒子的本地视频推理管控平台。系统通过 FastAPI 提供后台管理界面，支持多路视频输入、PT 权重管理、Python 模型逻辑管理、实时推理预览、报警信息输出和 RTSP 转推。

## 功能总览

- **视频监控**：支持多通道并行运行，可在页面中启动、停止、删除通道，并为每个通道独立选择输入连接和推理模型。
- **无模型预览**：通道模型可选择“无模型”，用于只看原始视频、排查摄像头和推流问题。
- **实时 FPS**：通道画面右上角显示实时 FPS，通道卡片下方显示累计帧数。
- **纵向画面适配**：视频监控页面中的浏览器预览使用等比例包含模式，纵向摄像头画面会缩小完整显示，不影响 RTSP 输出原始分辨率。
- **RTSP、文件、USB 输入**：连接列表支持 RTSP 地址、本地视频文件路径、USB 摄像头编号。
- **快速连接测试**：RTSP 连接测试优先使用快速 ffprobe 参数，并在日志中记录各阶段耗时。
- **快速通道打开**：通道运行时为网络流设置 OpenCV FFmpeg 快速打开参数，降低 RTSP 探测和等待时间。
- **RTSP 输出**：通道启动后可通过 FFmpeg 推送到 MediaMTX，供其他客户端拉流。
- **模型管理拆分**：模型管理分为 PT 文件管理和模型逻辑管理两块。
- **PT 文件管理**：支持上传 PT 文件、查看 PT 详情、显示模型标签、加载状态和显存占用。
- **模型逻辑管理**：支持管理 Python 推理逻辑，上传 `.py` 文件，并将逻辑绑定到一个或多个 PT 文件。
- **启动预加载与 warmup**：程序启动时自动加载 `models/` 目录下的 PT 模型并执行 warmup，减少首次推理延迟。
- **显存/内存指标**：视频监控页面显示显存占用；在 Jetson 这类统一内存设备上会按可用系统接口回退展示共享内存占用。
- **日志诊断**：服务日志写入 `data/server.log`，支持轮转；推理逻辑中的 `print`、标准错误会重定向到日志。
- **权限管理**：内置管理员、操作员、只读用户三类角色，按角色限制用户、连接、模型和通道操作。
- **历史日志**：记录登录、用户管理、连接管理、模型管理、通道操作等关键事件。

## 快速启动

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8888
```

访问后台：

```text
http://localhost:8888
```

默认账号：

```text
用户名：admin
密码：admin123
```

首次部署到设备后建议修改默认密码，并通过环境变量设置 `SESSION_SECRET`。

## 目录结构

- `app/`：FastAPI 后端、配置、数据存储、鉴权、模型注册表和视频流 worker。
- `app/func/`：Python 模型逻辑目录，`model_*.py` 文件会作为可绑定的推理逻辑使用。
- `app/models/`：模型适配层，包括 YOLO 检测、人员裁剪分类、函数逻辑适配和注册表。
- `app/streams/`：通道 worker、连接探测、RTSP 推流。
- `models/`：运行时 PT 权重目录。
- `data/`：运行时数据和日志，例如 `data/state.json`、`data/server.log`。
- `static/`：前端交互脚本和样式。
- `templates/`：后台管理页面模板。

## 模型管理流程

模型管理分为两类资源：PT 文件和模型逻辑。

### PT 文件管理

PT 文件是模型权重文件，保存在 `models/` 目录。页面支持：

- 上传 `.pt` 权重文件。
- 查看文件大小、更新时间、加载状态和显存占用。
- 点击“查看”打开详情弹窗，显示设备、warmup 状态、显存占用和标签 ID/名称列表。
- 程序启动时自动加载并 warmup 已存在的 PT 文件。

PT 文件通常较大，属于部署产物，不提交到 Git。仓库已通过 `.gitignore` 忽略：

```text
models/*.pt
```

### 模型逻辑管理

模型逻辑是 Python 程序，默认位于 `app/func/`。其中以 `model_` 开头的文件用于封装业务推理流程，例如：

- `app/func/model_unhat.py`：未戴安全帽检测逻辑。
- `app/func/model_unvest.py`：未穿反光衣检测逻辑。
- `app/func/model_smoke.py`：吸烟检测逻辑。
- `app/func/model_phone.py`：玩手机检测逻辑。

模型逻辑可以绑定一个或多个 PT 文件。通道运行时输入视频帧会调用这些 Python 逻辑，逻辑返回标注后的图像和检测/报警信息，页面和 RTSP 输出使用标注后的结果。

函数逻辑模型使用入口：

```text
app.model_functions:build_func_model
```

配置示例：

```json
{
  "model_path": "human_hat_cls_v4_bestm.pt",
  "logic_module": "app.func.model_unhat",
  "logic_function": "unhat",
  "conf": 0.25,
  "model_bindings": {
    "human_model": "05person_best11m.pt",
    "unhat_cls_model": "human_hat_cls_v4_bestm.pt"
  },
  "logic_kwargs": {}
}
```

说明：

- `logic_module` 是 Python 模块路径。
- `logic_function` 是要调用的函数名；为空时系统会自动从模块中探测可调用函数。
- `model_bindings` 的 key 必须和函数参数名一致，value 是 `models/` 目录下的 PT 文件名。
- `logic_kwargs` 会作为额外关键字参数传给模型逻辑。
- `conf` 或 `conf_threshold` 会作为置信度阈值传入。

模型逻辑函数推荐签名：

```python
from PIL import Image
from ultralytics import YOLO

def unhat(
    image: Image.Image,
    conf_threshold: float = 0.25,
    human_model: YOLO | None = None,
    unhat_cls_model: YOLO | None = None,
):
    ...
    return annotated_image, detections
```

返回值支持：

- `(annotated_image, detections)`：推荐形式。
- `annotated_image`：只返回标注图像。
- `detections`：只返回检测结果，系统会保留原图。

检测结果可以是 dict、Pydantic 模型或普通对象。常用字段包括：

- `box`：`[x1, y1, x2, y2]`
- `label`：标签名称。
- `conf`：置信度。
- `violation`：是否违规。
- `viol_content`：报警内容。
- `viol_color`：报警框颜色，RGB 格式。

## 视频通道流程

1. 在“连接列表”中新增 RTSP、文件或 USB 连接。
2. 点击“测试”验证连接可用性；RTSP 会优先走快速 ffprobe。
3. 在“视频监控”中选择连接和模型。
4. 模型可选择“无模型”，此时只预览原始视频。
5. 点击“启动”后，通道 worker 打开视频源，读取帧，调用模型逻辑，写入浏览器 MJPEG 预览，并按配置推送 RTSP。
6. 点击“停止”后，系统会通知推理线程中断、释放摄像头/RTSP 连接、停止 RTSP 推流，并清空 FPS 和帧数。

浏览器预览地址：

```text
/api/video/{stream_id}
```

通道状态只在视频监控页面轮询，其他页面不会持续请求通道快照。

## 连接地址填写规则

- **RTSP**：填写完整 RTSP 地址，例如 `rtsp://user:pass@192.168.1.20:8554/camera`。
- **文件**：填写服务器本机可访问的视频文件绝对路径或相对路径，例如 `/data/test.mp4`。
- **USB**：填写摄像头编号，例如 `0`、`1`。

注意：浏览器运行在本机或其他电脑上时，“文件”类型填写的是服务端设备上的路径，不是浏览器电脑上的路径。

## RTSP 输出

如需启用 RTSP 输出，请先安装并运行 MediaMTX，然后设置环境变量：

```bash
ENABLE_RTSP_PUSH=1
MEDIAMTX_HOST=127.0.0.1
MEDIAMTX_PORT=8554
RTSP_PUBLIC_HOST=192.168.1.100
```

说明：

- `ENABLE_RTSP_PUSH=1` 后通道启动时才会推 RTSP。
- `MEDIAMTX_HOST` 是 FFmpeg 推送到 MediaMTX 的地址。MediaMTX 和本程序在同一台设备上时建议使用 `127.0.0.1`。
- `RTSP_PUBLIC_HOST` 是其他机器访问 RTSP 时看到的设备 IP。不设置时程序会自动探测局域网 IP。
- `MEDIAMTX_PORT` 默认是 `8554`。

通道输出地址：

```text
rtsp://<RTSP_PUBLIC_HOST>:8554/stream/1
rtsp://<RTSP_PUBLIC_HOST>:8554/stream/2
rtsp://<RTSP_PUBLIC_HOST>:8554/stream/3
rtsp://<RTSP_PUBLIC_HOST>:8554/stream/4
```

排查顺序：

1. MediaMTX 是否启动并监听对应端口。
2. 服务启动前是否设置 `ENABLE_RTSP_PUSH=1`。
3. 通道是否已启动并持续有帧。
4. 防火墙是否放行 TCP 8554。
5. 在设备本机执行 `ffprobe rtsp://127.0.0.1:8554/stream/1` 是否能拉到流。

## 日志

服务日志默认写入：

```text
data/server.log
```

日志使用轮转文件，默认单文件 10 MB，保留 5 个备份。推理逻辑中的 `print` 会被重定向到日志，避免在终端和画面上混杂输出。

常见日志内容：

- 程序启动时 PT 模型预加载和 warmup 结果。
- 连接测试开始、ffprobe 耗时、状态写入耗时、总耗时。
- 通道启动、停止、打开视频源、读取帧、推理、发布 RTSP 等阶段。
- MJPEG 浏览器预览打开、关闭、空帧等待。
- 前端上报的视频布局和错误事件。

可用环境变量：

```bash
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_TO_CONSOLE=0
ACCESS_LOG_ENABLED=0
```

`ACCESS_LOG_ENABLED=0` 时会关闭 uvicorn access log，避免 `/api/snapshot` 等轮询请求刷屏。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SESSION_SECRET` | `change-me-on-device` | 登录会话签名密钥，部署时建议修改 |
| `ENABLE_RTSP_PUSH` | `0` | 是否启用 RTSP 输出 |
| `MEDIAMTX_HOST` | `127.0.0.1` | FFmpeg 推送 MediaMTX 的地址 |
| `MEDIAMTX_PORT` | `8554` | MediaMTX 端口 |
| `RTSP_PUBLIC_HOST` | 自动探测 | 对外展示的 RTSP 访问 IP |
| `STREAM_COUNT` | `4` | 初始通道数量 |
| `FRAME_WIDTH` | `1280` | RTSP 输出宽度 |
| `FRAME_HEIGHT` | `720` | RTSP 输出高度 |
| `FRAME_FPS` | `20` | 默认输出帧率 |
| `INFERENCE_TIMEOUT_SECONDS` | `15` | 单帧推理超时时间 |
| `CAPTURE_OPEN_TIMEOUT_MS` | `5000` | OpenCV 打开视频源超时 |
| `CAPTURE_READ_TIMEOUT_MS` | `5000` | OpenCV 读取帧超时 |
| `CONNECTION_TEST_TIMEOUT_MS` | `15000` | 连接测试总超时 |
| `LOG_MAX_BYTES` | `10485760` | 单个日志文件最大字节数 |
| `LOG_BACKUP_COUNT` | `5` | 日志轮转备份数量 |
| `LOG_TO_CONSOLE` | `0` | 是否同时输出到控制台 |
| `ACCESS_LOG_ENABLED` | `0` | 是否启用 uvicorn access log |

## 角色权限

- **管理员**：可管理用户、连接、模型、通道和系统设置。
- **操作员**：可管理连接和视频通道。
- **只读用户**：只能查看监控、连接、模型和日志。

## 更新日志

完整版本记录见 [CHANGELOG.md](CHANGELOG.md)。

## 作者信息

- 作者：Pyrrhus
- 邮箱：zhangxuefeng@batonsoft.com
