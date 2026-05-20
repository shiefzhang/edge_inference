# Edge Inference 视频推理管控平台

这是一个面向 Jetson 边缘盒子的 FastAPI 视频推理管控平台，用于管理多路视频输入、YOLO 推理函数、浏览器预览和 RTSP 输出。

## 功能特性

- 支持 4 路视频通道并行运行。
- 支持每路视频独立选择和切换推理函数。
- 支持 RTSP、视频文件、USB 摄像头连接配置。
- 支持浏览器预览带红框/蓝框标注的推理结果。
- 支持通过 FFmpeg 推送 H264 RTSP 流到 MediaMTX。
- 支持固定三角色用户管理：管理员、操作员、只读用户。
- 支持模型函数增删改查。
- 支持上传替换 `.pt` 权重文件。
- 支持上传替换 Python 推理逻辑代码。

## 模型函数

平台管理的不是单纯的 `.pt` 文件，而是 Python 推理函数。每个模型函数由以下内容组成：

- 函数 ID
- 名称
- 类型
- Python 入口
- 配置 JSON
- 启用状态

例如安全帽检测可以封装为一个完整推理函数：先调用人员检测模型，再裁剪人框，最后调用安全帽分类模型。后续升级推理流程时，可以上传新的 Python 逻辑代码替换。

上传的 Python 文件需要提供：

```python
def build_model(definition, models_dir, modules):
    ...
```

## 作者信息

- 作者：Pyrrhus
- 邮箱：zhangxuefeng@batonsoft.com

## 快速启动

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问：

```text
http://localhost:8000
```

默认账号：

```text
用户名：admin
密码：admin123
```

首次登录后建议修改默认密码。

## RTSP 输出

如需启用 RTSP 输出，请先安装并运行 MediaMTX，然后设置环境变量：

```bash
MEDIAMTX_HOST=127.0.0.1
MEDIAMTX_PORT=8554
ENABLE_RTSP_PUSH=1
```

每个运行中的通道会推送到：

- `rtsp://<host>:8554/stream/1`
- `rtsp://<host>:8554/stream/2`
- `rtsp://<host>:8554/stream/3`
- `rtsp://<host>:8554/stream/4`

## 目录说明

- `app/`：FastAPI 后端、推理模块、视频流 worker。
- `models/`：默认 YOLO 权重文件。
- `templates/`：后台管理页面模板。
- `static/`：前端样式和交互脚本。
- `data/`：运行时状态数据，默认不提交到 Git。

## 备注

上传的模型函数代码会保存到 `app/user_functions/`，上传的权重文件会保存到 `models/`。如果正在运行的视频通道使用了某个模型函数，删除或替换前建议先停止相关通道。
