# Aliyun Deployment Notes

These notes are a checklist for a future Aliyun server deployment. The current
repository is a source-code release; model weights, datasets, local Python
environments, containers, and generated assets are intentionally not pushed.

## 1. Server Requirements

- Linux server with NVIDIA GPU if you need full rendering and model inference.
- Matching NVIDIA driver/CUDA runtime for the local project environments.
- Enough disk for model weights, checkpoints, avatar media, caches, and outputs.
- Open ports only as needed. For public access, prefer Nginx reverse proxy over
  exposing uvicorn directly.

## 2. Restore Source

```bash
git clone git@github.com:1114531938/avatar-system-full.git /scratch/e1554543/avatar_system_full
cd /scratch/e1554543/avatar_system_full
```

If the deployment path changes, update scripts that currently hard-code:

```text
/scratch/e1554543/avatar_system_full
```

Most first-party scripts and the avatar agent also support:

```bash
export AVATAR_SYSTEM_ROOT=/your/deploy/path/avatar_system_full
```

`src/avatar_system/pipeline_config.yaml` uses `${PROJECT_ROOT}` placeholders
that are expanded from `AVATAR_SYSTEM_ROOT` by the pipeline config loader.

## 3. Restore Local Assets

Restore these from backup or rebuild them:

```text
runtime/cache/
runtime/containers/
runtime/data/
runtime/outputs/                 # optional historical outputs/logs
integrations/gaussian_avatar/media/
integrations/gaussian_avatar/datasets/
integrations/avamerg/ckpt/
integrations/emotivoice/models/
integrations/deeptalk/_downloads/
integrations/vhap/asset/
apps/booth/                      # if the 7862 3DEPB service is needed
```

## 4. Recreate Python Environments

The local working server currently uses component-specific environments, for
example:

```text
runtime/cache/venvs/web/
runtime/cache/venvs/perception/
integrations/avamerg/.avamerg38/
integrations/emotivoice/.EmotiVoice/
integrations/gaussian_avatar/.GSavatar/
runtime/cache/venvs/deeptalk/
runtime/cache/venvs/vhap*/
```

Use each component's requirements file or existing setup notes to recreate them.
Do not commit these environments to GitHub.

## 5. Configure Runtime Secrets

Use `config/runtime.env.example` as a reference and store the real environment
file outside Git, for example:

```bash
cp config/runtime.env.example config/runtime.env
chmod 600 config/runtime.env
```

Load it before starting services:

```bash
set -a
source config/runtime.env
set +a
bash scripts/avatar.sh web
```

## 6. Start Services

Main studio:

```bash
bash scripts/avatar.sh web
```

Booth / 3DEPB:

```bash
bash scripts/avatar.sh booth
```

Check:

```bash
bash scripts/avatar.sh --help
```

## 7. Reverse Proxy Reminder

For public deployment, put Nginx in front of the web service and proxy to
`127.0.0.1:7861` or `127.0.0.1:7862`. Keep worker ports bound to localhost
unless there is a specific reason to expose them.
