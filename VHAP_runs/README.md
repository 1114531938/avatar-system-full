# VHAP Runs

这个目录把 VHAP 的仓库、解释器和项目侧脚本都收口到 `avatar_system_full` 内，方便把“新 subject 资产构建线”纳入同一套工程。

当前布局：

```text
VHAP_runs/
├── repo/               # 项目内 VHAP 仓库副本
├── python310/          # 项目内便携 CPython 3.10
├── .vhap121/           # 项目内 VHAP 主环境
└── .vhap_pre121/       # 项目内 VHAP 预处理环境
```

说明：

- `repo/` 已经放进本工程，后续不再依赖外部 `vhap_runs/repos/VHAP`。
- `python310/` 是项目内自带的 3.10 解释器基座，用来托管 `.vhap121` 和 `.vhap_pre121`。
- `.vhap121` / `.vhap_pre121` 都已经迁到项目内，`bin/python`、`activate`、`pyvenv.cfg` 和脚本 shebang 都指向当前工程路径。
- `scripts/vhap_env.sh` 默认使用 `.vhap121`；如果要切到预处理环境，可以手动设置 `VHAP_ENV_NAME=.vhap_pre121`。
- 运行 tracking/export 之前，节点上还需要有可用的 CUDA toolkit。`scripts/vhap_env.sh` 现在会优先使用外部已设置的 `CUDA_HOME`，否则自动探测常见的 CUDA 安装目录；如果找不到，会直接报错而不是误把 `CUDA_HOME` 指到 venv。

统一入口请优先使用：

- `scripts/vhap_env.sh`
- `scripts/init_subject.sh`
- `scripts/run_vhap_subject.sh`
- `scripts/export_vhap_to_gaussian.sh`
- `scripts/train_gaussian_subject.sh`
- `scripts/register_avatar_asset.sh`
