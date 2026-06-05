# Avatar System Full

`/scratch/e1554543/avatar_system_full` 是当前正在使用的数字人系统主工程目录。

这份 README 只描述**当前实际可运行的结构、服务和工作流**。

> GitHub 源码仓库不包含模型权重、数据集、训练资产、虚拟环境、容器、缓存和运行输出。
> 发布版的包含/排除范围见 `SOURCE_RELEASE.md`；本地完整部署仍按本文目录结构运行。

## 1. 项目目标

这套工程完成一条从语音输入到数字人视频输出的完整链路：

1. 感知层：ASR + 情感识别 + Task1 输入构建
2. 对话层：AvaMERG 生成回复文本
3. TTS：EmotiVoice 生成回复音频
4. 动作层：DEEPTalk / wav_to_flame 生成 FLAME motion
5. 渲染层：GaussianAvatars 生成最终数字人视频
6. Web：提供网页录音、上传、运行、查看结果、3D 预览与导出

## 2. 当前目录结构

下面这些目录是现在主工程中真正使用的部分：

```text
/scratch/e1554543/avatar_system_full
├── README.md
├── scripts/                         # 启动脚本、服务管理脚本、调试脚本
├── web_app/                         # FastAPI + 前端页面
├── perception_layer/                # 感知层
├── AvaMERG_runs/                    # AvaMERG
├── EmotiVoice_runs/                 # TTS
├── wav_to_flame/                    # DEEPTalk / wav to FLAME
├── GSavatar_runs/                   # GaussianAvatars
├── VHAP_runs/                       # VHAP 仓库副本 + 环境入口
├── data/                            # 新 subject 原始数据与中间资产
├── tools/                           # avatar_agent、ffmpeg 等工具
├── containers/                      # Apptainer writable sandbox
├── cache/                           # HF / XDG / ModelScope / pipeline cache
└── outputs/                         # 运行结果、网页结果、服务日志
```

### 关键子目录

```text
web_app/                             # Web server + static frontend
web_app/static/                      # HTML / CSS / JS / vendor
web_app/.web_venv/                   # Web 服务虚拟环境

scripts/avatar_service.sh            # 一键管理 web + worker
scripts/run_web.sh                   # 只启动 web
scripts/run_tts_worker.sh            # 只启动 TTS worker
scripts/run_avamerg_worker.sh        # 只启动 AvaMERG worker
scripts/run_deeptalk_worker.sh       # 只启动 DEEPTalk worker
scripts/run_perception_worker.sh     # 只启动 perception worker
scripts/run_gaussian_render_worker.sh# 只启动 Gaussian render worker
scripts/run_agent.sh                 # 命令行整条链路入口
scripts/vhap_env.sh                  # VHAP 环境统一入口
scripts/init_subject.sh              # 初始化新 subject 目录
scripts/run_vhap_subject.sh          # 跑 VHAP preprocess/track/export
scripts/export_vhap_to_gaussian.sh   # 单独导出 Gaussian source
scripts/train_gaussian_subject.sh    # 训练新的 Gaussian avatar
scripts/register_avatar_asset.sh     # 注册 point_cloud/template 到 media/<id>

tools/avatar_agent/                  # 命令行 orchestrator / export 工具
tools/ffmpeg-git-20240629-amd64-static/

GSavatar_runs/GaussianAvatars/media/306/
                                    # avatar 306 相关 point_cloud / template 等

VHAP_runs/                           # 项目内 VHAP 仓库 + Python 3.10 + venv
data/subjects/<subject_id>/          # 新 subject 数据工作区

outputs/service_logs/                # 服务日志和 pid 文件
outputs/web_uploads/                 # 网页上传 wav
outputs/web_<...>/                   # 网页每次运行结果
```

## 3. 当前运行方式

现在有两种主用方式：

1. 命令行整条链路运行
2. Web UI 运行

---

## 4. 命令行运行

### 4.1 环境变量

如果要跑完整回复链路，先在 shell 里设置：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openai/gpt-oss-120b:free"
```

### 4.2 直接运行整条链路

```bash
cd /scratch/e1554543/avatar_system_full

bash scripts/run_agent.sh \
  /scratch/e1554543/avatar_system_full/perception_layer/data/demo_wavs/sample_dialog_02.wav \
  306
```

### 4.3 轻量测试

如果只是检查流程和路径，不想跑完整视频导出：

```bash
cd /scratch/e1554543/avatar_system_full

bash scripts/run_agent.sh \
  /scratch/e1554543/avatar_system_full/perception_layer/data/demo_wavs/sample_dialog_02.wav \
  306 \
  --prepare_only --no_video_export --no_llm
```

## 5. Web UI 运行

### 5.1 一键启动推荐方式

当前推荐统一使用：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh start
```

这会启动：

- Web server
- TTS worker
- AvaMERG worker
- DEEPTalk worker
- perception worker
- Gaussian render worker

默认端口：

```text
7861  web
8788  TTS worker
8789  AvaMERG worker
8790  DEEPTalk worker
8791  perception worker
8792  Gaussian render worker
```

### 5.2 重启

改了后端、worker、服务脚本后，使用：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh restart
```

### 5.3 查看状态和日志

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh status
bash scripts/avatar_service.sh logs
```

### 5.4 单独控制 Gaussian render worker

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh stop-gaussian-render
bash scripts/avatar_service.sh start-gaussian-render
```

### 5.5 注意

`avatar_service.sh start` 的逻辑是：

- 如果某个 worker 的 `/health` 已经能通
- 就认为它“已经在运行”
- **不会强制重启它**

所以如果你更新了 worker 代码，但旧进程还占着端口，单纯 `start` 可能不会生效。  
这种情况下请用：

```bash
bash scripts/avatar_service.sh restart
```

或者单独 stop / start 对应 worker。

如需继续使用 `7860`，可以显式覆盖端口：

```bash
PORT=7860 bash scripts/avatar_service.sh start
```

## 6. Web 页面功能

当前网页入口默认是：

```text
http://localhost:7861
```

页面支持：

- 录音
- 上传 wav
- 设置 API Key / Base URL / Model
- 选择 avatar id
- Generate 运行整条网页链路
- 显示日志和结果路径
- 下载输出文件

### 6.1 输出视图

当前页面主要有三个视图：

- `Video`
- `3D Render`
- `3D Debug`

#### Video

显示最终 `final_video.mp4`。

#### 3D Render

当前主线方案。  
它不是浏览器自己做高质量 Gaussian 渲染，而是：

- 前端负责播放、拖拽、进度条、音频同步
- 后端调用常驻 Gaussian render worker
- worker 用 CUDA 渲染当前 frame + 当前 camera

它现在支持：

- 播放 / 暂停
- 拖拽视角
- 音频同步
- 拖动进度条
- Reset camera
- Fit Subject
- Export View

#### 3D Debug

这是实验 / 调试视图，不是主方案。

里面目前有两个子模式：

- `Point`
- `WebGPU`

说明：

- `Point`：点云调试
- `WebGPU`：浏览器端实验性 Gaussian / splat 路线

当前 `WebGPU` 仍然是实验线，**不要把它当最终效果标准**。

## 7. 当前渲染方案状态

### 7.1 主方案

主方案是：

**`3D Render = CUDA interactive viewer`**

当前已经做过的优化包括：

- render worker 常驻
- 直接返回图片 bytes，而不是先写 PNG 再让前端二次拉取
- 播放时连续拉帧
- 拖拽时优先当前视角
- 进度条 / reset / fit subject 时立即刷新

### 7.2 调试方案

`3D Debug / WebGPU` 是继续研究纯前端渲染路线的实验入口。  
目前还达不到 `final_video.mp4` 的质量。

## 8. 运行输出

每次网页运行会生成一个目录：

```text
/scratch/e1554543/avatar_system_full/outputs/web_<run_id>
```

常见结构：

```text
outputs/web_<run_id>/
├── logs/
├── outputs/
├── artifacts/
├── state.json
└── manifest.json
```

常见产物在：

```text
outputs/web_<run_id>/artifacts/
```

通常包括：

- `final_video.mp4`
- `white_model.mp4`
- `reply.wav`
- `reply_enhanced.wav`
- `flame_motion.npz`
- `manifest.json`

### 8.1 自动清理

当前 Web 端已经做了自动清理策略：

- `outputs/` 下网页运行目录默认只保留最近 5 个
- `service_logs` 和 `web_uploads` 不会被误删

## 9. 缓存目录

当前运行统一使用项目内缓存：

```text
/scratch/e1554543/avatar_system_full/cache
```

其中常见子目录包括：

- `cache/hf`
- `cache/xdg`
- `cache/modelscope`
- `cache/nltk_data`
- `cache/cache`（Apptainer / OCI / oras 相关缓存）

如果需要整体迁移、备份或清理缓存，优先围绕这个目录处理。

## 10. 常见问题

### 10.1 为什么我改了 worker 代码，页面还是老行为

`avatar_service.sh` 现在不仅检查 `/health`，还会检查：

- 端口上的监听进程 PID
- PID 文件是否匹配这个监听进程

如果端口上是“旧进程 / 孤儿进程 / 不是脚本当前管理的进程”，脚本会先替换它再启动新服务。  
常用命令：

```bash
bash scripts/avatar_service.sh restart
bash scripts/avatar_service.sh status
```

### 10.2 为什么页面报 `Gaussian render worker failed: HTTP Error 404: Not Found`

这通常说明 web server 和 Gaussian render worker 版本不一致。  
现在推荐直接执行：

```bash
bash scripts/avatar_service.sh restart
```

如果还需要进一步确认端口占用：

```bash
ss -ltnp | grep 8792
```

### 10.3 为什么 3D Render 不是满帧实时

因为当前主方案仍然是：

- 前端发送 camera + frame
- 后端 CUDA 渲染单帧
- 返回压缩图像

这已经比“落盘 PNG 再拉文件”快很多，但它仍然是“高质量交互预览”，不是浏览器本地实时原生渲染。

## 11. 建议工作流

### 改前端

只改：

- `web_app/static/index.html`
- `web_app/static/style_commercial.css`
- `web_app/static/app.js`

通常：

- 不需要重启整套服务
- 强刷浏览器即可

### 改后端 / worker / 脚本

改这些文件后建议重启：

- `web_app/server.py`
- `scripts/*.sh`
- `GSavatar_runs/GaussianAvatars/gaussian_render_worker.py`
- `tools/avatar_agent/*.py`

执行：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh restart
```
