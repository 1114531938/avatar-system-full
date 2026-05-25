# Avatar Head Build Guide

这份文档专门说明两条“从数据到可驱动人头资产”的路线：

1. **自己的视频 -> 新人头**
2. **NeRSemble 官方预处理数据 -> 新人头**

文档目标不是讲论文原理，而是讲现在这个工程里**怎么真的跑**。

---

## 1. 结果长什么样才算“做出一个头”

无论是哪条路线，最后都要落到当前系统可识别的资产格式：

```text
GSavatar_runs/GaussianAvatars/media/<avatar_id>/
├── point_cloud.ply
└── flame_param.npz
```

只要这两个文件在，当前 Web / agent 主流程就可以用这个 `avatar_id`：

- 录音或上传 wav
- 跑 perception / LLM / TTS / DEEPTalk
- 生成 `flame_motion.npz`
- 调用 GaussianAvatars 渲染

也就是说，**“做出一个头”** 在工程里的定义就是：

- 训练出 `point_cloud.ply`
- 整理出匹配它的 `flame_param.npz`
- 注册到 `media/<avatar_id>/`

---

## 2. 路线 A：自己的视频 -> 新人头

这条路线适合：

- 你自己录的新人物
- 单目视频
- 或更规范的多目 / 多视角数据

### 2.1 当前目录约定

每个新人物都放到：

```text
data/subjects/<subject_id>/
├── raw/              # 原始视频 / 图像
├── vhap/             # VHAP tracking 输出
├── gaussian_source/  # 导出给 GaussianAvatars 的训练源
├── gaussian_train/   # 完整训练输出
├── gaussian_train_30k/  # fast-30k 训练输出
└── final_asset/      # 可选，后处理整理区
```

### 2.2 路线 A 的代码流程

#### Step 1. 初始化 subject 工作区

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/init_subject.sh person_001
```

这一步会建立：

```text
data/subjects/person_001/
```

#### Step 2. 放入自己的原始视频

把视频放进：

```text
data/subjects/person_001/raw/
```

例如：

```text
data/subjects/person_001/raw/person_001.mp4
```

#### Step 3. 跑 VHAP tracking / export

单目视频示例：

```bash
cd /scratch/e1554543/avatar_system_full

bash scripts/run_vhap_subject.sh person_001 \
  --mode monocular \
  --input /scratch/e1554543/avatar_system_full/data/subjects/person_001/raw/person_001.mp4
```

如果要分步跑，也可以：

```bash
bash scripts/run_vhap_subject.sh person_001 --mode monocular --sequence person_001 --track-only
bash scripts/run_vhap_subject.sh person_001 --mode monocular --sequence person_001 --export-only
```

这一步的核心结果在：

```text
data/subjects/person_001/vhap/
data/subjects/person_001/gaussian_source/
```

其中 `gaussian_source/` 里最关键的是：

- `canonical_flame_param.npz`
- `transforms_train.json`
- `transforms_val.json`
- `transforms_test.json`

这说明：  
**VHAP 已经把原始视频整理成 GaussianAvatars 能训练的数据集格式。**

#### Step 4. 训练 GaussianAvatars

先跑快速 smoke 版本：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_gaussian_subject.sh person_001 --fast-30k
```

或者完整训练：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_gaussian_subject.sh person_001
```

训练源默认读取：

```text
data/subjects/person_001/gaussian_source/
```

训练输出通常落在：

```text
data/subjects/person_001/gaussian_train_30k/
```

或

```text
data/subjects/person_001/gaussian_train/
```

#### Step 5. 注册成系统可用头像

例如注册成 `avatar_id = 1001`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/register_avatar_asset.sh person_001 1001 \
  --model /scratch/e1554543/avatar_system_full/data/subjects/person_001/gaussian_train_30k \
  --canonical /scratch/e1554543/avatar_system_full/data/subjects/person_001/gaussian_source/canonical_flame_param.npz
```

注册完成后，会生成：

```text
GSavatar_runs/GaussianAvatars/media/1001/
├── point_cloud.ply
└── flame_param.npz
```

#### Step 6. 在 Web 里使用

网页里选择：

- `Avatar = 1001`

然后上传或录音，直接点 `Generate`。

---

## 3. 路线 B：NeRSemble 官方预处理数据 -> 新人头

这条路线适合：

- 先用官方数据快速造几个稳定的人头
- 验证当前系统的训练 / 注册 / Web 使用链路

### 3.1 当前工程里已经落好的内容

官方预处理包下载位置：

```text
GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/release.zip
```

当前真正解出来并确认可用的人物 subject 是：

- `074`
- `165`
- `218`

> 注意：当前你下载到并整理好的这批数据，在工程里最终落地的是这 3 个 subject。

### 3.2 路线 B 的代码流程

#### Step 1. 下载并解压官方数据

相关脚本：

- `scripts/unpack_nersemble_release.sh`
- `scripts/inventory_nersemble_subjects.sh`

当前我们已经进一步把训练入口整理成更稳定的结构：

```text
GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/release_expanded/
GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/
```

其中这三个入口最重要：

- `union10_sources/074`
- `union10_sources/165`
- `union10_sources/218`

它们分别指向每个 subject 的 `UNION10` 训练源。

#### Step 2. 查看 subject 清单和 avatar 映射

当前已经写好的两个文件：

- `GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/nersemble_subjects.tsv`
- `GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/nersemble_avatar_map.tsv`

当前建议映射是：

| subject | subject_id | source_dir | avatar_id |
|---|---|---|---:|
| 074 | `nersemble_074_union10` | `.../union10_sources/074` | 2001 |
| 165 | `nersemble_165_union10` | `.../union10_sources/165` | 2002 |
| 218 | `nersemble_218_union10` | `.../union10_sources/218` | 2003 |

#### Step 3. 单独训练一个官方头

例如训练 `074`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_gaussian_subject.sh nersemble_074_union10 \
  --source /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/074 \
  --fast-30k
```

`165`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_gaussian_subject.sh nersemble_165_union10 \
  --source /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/165 \
  --fast-30k
```

`218`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_gaussian_subject.sh nersemble_218_union10 \
  --source /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/218 \
  --fast-30k
```

#### Step 4. 批量训练三个官方头

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_nersemble_batch.sh --fast-30k
```

#### Step 5. 注册官方头

单独注册示例。

注册 `074 -> 2001`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/register_avatar_asset.sh nersemble_074_union10 2001 \
  --model /scratch/e1554543/avatar_system_full/data/subjects/nersemble_074_union10/gaussian_train_30k \
  --canonical /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/074/canonical_flame_param.npz
```

注册 `165 -> 2002`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/register_avatar_asset.sh nersemble_165_union10 2002 \
  --model /scratch/e1554543/avatar_system_full/data/subjects/nersemble_165_union10/gaussian_train_30k \
  --canonical /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/165/canonical_flame_param.npz
```

注册 `218 -> 2003`：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/register_avatar_asset.sh nersemble_218_union10 2003 \
  --model /scratch/e1554543/avatar_system_full/data/subjects/nersemble_218_union10/gaussian_train_30k \
  --canonical /scratch/e1554543/avatar_system_full/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/union10_sources/218/canonical_flame_param.npz
```

或者批量注册：

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/register_nersemble_batch.sh --model-suffix gaussian_train_30k --base-avatar-id 2001
```

#### Step 6. 在 Web 里使用

当前网页下拉已经会列出：

- `2001 · NeRSemble 074`
- `2002 · NeRSemble 165`
- `2003 · NeRSemble 218`

选中后，直接上传音频 / 录音即可。

---

## 4. 两条路线的关系

### 路线 A：自己的视频

优点：

- 目标是做你自己的新人物
- 能真正扩展头像库

缺点：

- 采集质量更敏感
- VHAP / tracking / 训练更容易踩坑

### 路线 B：NeRSemble 官方数据

优点：

- 数据更干净
- 很适合验证训练、注册和 Web 闭环
- 适合快速得到几个稳定官方头

缺点：

- 不是你自己的新人物

实际工程建议是：

1. 先用路线 B 稳定系统
2. 再用路线 A 做你自己的新人物

---

## 5. 当前系统里已经验证过的关键点

### 已验证

- 自己的视频 `demo1`：
  - `VHAP -> export -> Gaussian train -> register`
  - 已注册为 `avatar_id = 1001`

- NeRSemble 官方头：
  - `074 -> 2001`
  - `165 -> 2002`
  - `218 -> 2003`

- Web / agent 主流程：
  - 已可使用这些已注册头像进行语音驱动

### 还要注意

- 最终 `final_video.mp4` 导出依赖 GPU 可见
- 如果当前服务跑在看不到 CUDA 的环境里，最后渲染会失败
- 训练属于重任务，建议在 GPU 节点 + `tmux` 中执行

---

## 6. 常用命令速查

### Web 服务

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar_service.sh start
bash scripts/avatar_service.sh restart
bash scripts/avatar_service.sh status
```

### 自己的视频 -> 新头

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/init_subject.sh person_001
bash scripts/run_vhap_subject.sh person_001 --mode monocular --input /path/to/person_001.mp4
bash scripts/train_gaussian_subject.sh person_001 --fast-30k
bash scripts/register_avatar_asset.sh person_001 1001 \
  --model /scratch/e1554543/avatar_system_full/data/subjects/person_001/gaussian_train_30k \
  --canonical /scratch/e1554543/avatar_system_full/data/subjects/person_001/gaussian_source/canonical_flame_param.npz
```

### 官方 NeRSemble -> 新头

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/train_nersemble_batch.sh --fast-30k
bash scripts/register_nersemble_batch.sh --model-suffix gaussian_train_30k --base-avatar-id 2001
```

---

## 7. 一句话总结

- **自己的视频路线**：`raw video -> VHAP -> gaussian_source -> Gaussian train -> register -> Web 使用`
- **NeRSemble 路线**：`official preprocessed source -> Gaussian train -> register -> Web 使用`

这两条路线最后都会汇合到同一个结果：

```text
GSavatar_runs/GaussianAvatars/media/<avatar_id>/
├── point_cloud.ply
└── flame_param.npz
```

只要走到这里，这个头就已经进入当前系统的可用头像库了。
