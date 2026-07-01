# Reproduction Setup on a New Server

This document is the migration checklist for running `avatar_system_full` on a
new experiment server after cloning from GitHub.

Important: GitHub contains the source code, UI assets, scripts, and docs. It
does not contain local virtual environments, Apptainer containers, downloaded
model checkpoints, avatar point clouds, runtime cache, secrets, uploads, or
generated outputs. Those files are intentionally ignored because they are large,
machine-specific, or sensitive.

## Runtime Asset Release

The full Booth and worker chain needs a separate runtime asset bundle. On a new
server, restore it after cloning:

```bash
cd "$AVATAR_SYSTEM_ROOT"
bash scripts/download_runtime_assets.sh
```

By default the script downloads GitHub Release assets from:

```text
repo: 1114531938/avatar-system-full
tag:  runtime-assets-2026-07-01
```

Override these if a newer Release is published:

```bash
export AVATAR_RUNTIME_REPO=1114531938/avatar-system-full
export AVATAR_RUNTIME_RELEASE_TAG=runtime-assets-2026-07-01
export AVATAR_RUNTIME_ASSET_PREFIX=avatar-system-full-runtime-assets
bash scripts/download_runtime_assets.sh
```

The restore script verifies `sha256sums.txt`, extracts the split
`tar.zst` archive into the repository root, and checks the required paths for
the 7862 Booth stack and workers 8788-8792.

To publish the runtime bundle from the original server:

```bash
cd /scratch/e1554543/avatar_system_full
export GH_TOKEN=...   # GitHub token with repo contents/release permission
bash scripts/publish_runtime_assets.sh
```

Without `GH_TOKEN` or `GITHUB_TOKEN`, the publish script still creates local
split artifacts under `runtime/release_assets/runtime-assets-2026-07-01/`.

## 1. Verify the GitHub Source Snapshot

On the original server, the repository remote is:

```bash
git remote -v
# origin ssh://git@ssh.github.com:443/1114531938/avatar-system-full.git
```

Before moving to another server, the source tree should be clean after commit
and push:

```bash
cd /scratch/e1554543/avatar_system_full
git fetch origin --prune
git status --short --branch
git log --oneline --decorate --left-right --cherry-pick main...origin/main
```

Expected after a successful push:

```text
## main...origin/main
```

No `M`, `??`, ahead, or behind lines should remain. The command below should
print nothing:

```bash
git ls-files -o --exclude-standard
```

Ignored files are not uploaded. Check them explicitly when reproducing the full
runtime:

```bash
git status --porcelain=v1 --ignored=matching
```

## 2. Clone on the New Server

Choose the new installation directory first. The examples below use the same
path as the current machine:

```bash
export AVATAR_SYSTEM_ROOT=/scratch/e1554543/avatar_system_full
git clone ssh://git@ssh.github.com:443/1114531938/avatar-system-full.git "$AVATAR_SYSTEM_ROOT"
cd "$AVATAR_SYSTEM_ROOT"
```

If the new server uses a different path, set these variables in every shell or
service that starts the system:

```bash
export AVATAR_SYSTEM_ROOT=/your/path/avatar_system_full
export PROJECT_ROOT="$AVATAR_SYSTEM_ROOT"
```

Then create a local runtime env file:

```bash
cp config/runtime.env.example config/runtime.env
chmod 600 config/runtime.env
```

Edit at least these values if the path or API provider differs:

```text
AVATAR_SYSTEM_ROOT=/your/path/avatar_system_full
PROJECT_ROOT=/your/path/avatar_system_full
DEPB_ROOT=/your/path/avatar_system_full/apps/booth
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=liquid/lfm-2.5-1.2b-instruct:free
```

Load the file before starting services:

```bash
set -a
source config/runtime.env
set +a
```

## 3. System Packages

Install these outside Python:

```bash
# Ubuntu example
sudo apt-get update
sudo apt-get install -y git curl build-essential ffmpeg python3-venv python3-dev
```

For the full GPU pipeline, the server also needs:

```text
NVIDIA driver compatible with the host GPU
CUDA runtime compatible with the Python wheels and container
Apptainer/Singularity with GPU support
```

The startup script defaults to:

```text
AVATAR_CONTAINER=$AVATAR_SYSTEM_ROOT/runtime/containers/gaussianav_jammy
AVATAR_FFMPEG=$AVATAR_SYSTEM_ROOT/runtime/cache/bin/ffmpeg
AVATAR_FFPROBE=$AVATAR_SYSTEM_ROOT/runtime/cache/bin/ffprobe
```

If you use system FFmpeg instead of the cached binaries:

```bash
export AVATAR_FFMPEG=/usr/bin/ffmpeg
export AVATAR_FFPROBE=/usr/bin/ffprobe
export DEPB_FFMPEG=/usr/bin/ffmpeg
```

## 4. Python Environments

The current scripts expect component-specific environments at these paths:

```text
runtime/cache/venvs/web
runtime/cache/venvs/perception
runtime/cache/venvs/deeptalk
integrations/avamerg/.avamerg38
integrations/emotivoice/.EmotiVoice
integrations/gaussian_avatar/.GSavatar_glibc
```

Create the web environment:

```bash
python3 -m venv "$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/web"
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/web/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/web/bin/python" -m pip install -r apps/web/requirements.txt
```

Create the perception environment:

```bash
python3 -m venv "$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/perception"
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/perception/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/perception/bin/python" -m pip install -r integrations/perception/env/requirements_whisper_stage1.txt
```

Create the DEEPTalk environment:

```bash
python3 -m venv "$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/deeptalk"
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/deeptalk/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/deeptalk/bin/python" -m pip install -r integrations/deeptalk/requirements.txt
```

Create the AvaMERG environment. It uses Python 3.8 in the current layout:

```bash
python3.8 -m venv "$AVATAR_SYSTEM_ROOT/integrations/avamerg/.avamerg38"
"$AVATAR_SYSTEM_ROOT/integrations/avamerg/.avamerg38/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/integrations/avamerg/.avamerg38/bin/python" -m pip install -r integrations/avamerg/requirements.txt
```

Create the EmotiVoice environment:

```bash
python3 -m venv "$AVATAR_SYSTEM_ROOT/integrations/emotivoice/.EmotiVoice"
"$AVATAR_SYSTEM_ROOT/integrations/emotivoice/.EmotiVoice/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/integrations/emotivoice/.EmotiVoice/bin/python" -m pip install -r integrations/emotivoice/requirements.txt
```

Create the GaussianAvatar environment:

```bash
python3 -m venv "$AVATAR_SYSTEM_ROOT/integrations/gaussian_avatar/.GSavatar_glibc"
"$AVATAR_SYSTEM_ROOT/integrations/gaussian_avatar/.GSavatar_glibc/bin/python" -m pip install -U pip
"$AVATAR_SYSTEM_ROOT/integrations/gaussian_avatar/.GSavatar_glibc/bin/python" -m pip install -r integrations/gaussian_avatar/requirements.txt
```

If binary CUDA packages fail, recreate each environment with the exact PyTorch
and CUDA wheel versions supported by the new server.

## 5. Runtime Directories

Create the local runtime directories:

```bash
mkdir -p \
  runtime/cache/hf \
  runtime/cache/modelscope \
  runtime/cache/xdg \
  runtime/cache/nltk_data \
  runtime/cache/bin \
  runtime/containers \
  runtime/data \
  runtime/outputs \
  runtime/tmp
```

Recommended cache variables:

```bash
export HF_HOME="$AVATAR_SYSTEM_ROOT/runtime/cache/hf"
export MODELSCOPE_CACHE="$AVATAR_SYSTEM_ROOT/runtime/cache/modelscope"
export XDG_CACHE_HOME="$AVATAR_SYSTEM_ROOT/runtime/cache/xdg"
export NLTK_DATA="$AVATAR_SYSTEM_ROOT/runtime/cache/nltk_data"
```

## 6. Required External Assets

Restore these ignored assets from backup or download them again. The full
pipeline will not run from a plain GitHub clone without them.

### 6.1 Apptainer Container

Current expected path:

```text
runtime/containers/gaussianav_jammy
```

Current local size observed: about `10G`.

The script runs containerized workers with:

```bash
apptainer exec --nv \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$AVATAR_SYSTEM_ROOT/runtime/containers/gaussianav_jammy" \
  bash -lc "..."
```

If the new server does not have `/scratch` or `/home/svu`, set a compatible
bind string:

```bash
export APPTAINER_FLAGS="--nv -B /your/path:/your/path"
```

### 6.2 FFmpeg

Current expected paths:

```text
runtime/cache/bin/ffmpeg
runtime/cache/bin/ffprobe
```

You can instead set `AVATAR_FFMPEG`, `AVATAR_FFPROBE`, and `DEPB_FFMPEG` to
system binaries.

### 6.3 Perception Models

Whisper small:

```text
runtime/cache/xdg/whisper/small.pt
```

Current local size observed: about `462M`.

ModelScope emotion2vec plus seed:

```text
runtime/cache/modelscope/models/iic/emotion2vec_plus_seed/
```

Current local size observed: about `1.1G`.

DEEPTalk also uses a finetuned emotion2vec checkpoint:

```text
integrations/deeptalk/DEE/models/emo2vec/checkpoint/emotion2vec_base.pt
```

Current local size observed: about `1.1G`.

### 6.4 AvaMERG Checkpoints

Current expected directory:

```text
integrations/avamerg/ckpt/
```

Current local size observed: about `18G`.

Important files include:

```text
integrations/avamerg/ckpt/pretrained_ckpt/imagebind_ckpt/huge/imagebind_huge.pth
integrations/avamerg/ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0/
```

### 6.5 EmotiVoice Assets

Current expected checkpoint directory:

```text
integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/
```

Current local size observed: about `1.4G`.

Important files:

```text
integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/g_00140000
integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/do_00140000
```

The worker is started with:

```text
--logdir prompt_tts_open_source_joint --config_folder config/joint --checkpoint g_00140000
```

SimBERT assets:

```text
integrations/emotivoice/WangZeJun/simbert-base-chinese/
```

### 6.6 DEEPTalk Checkpoints

Current expected paths:

```text
integrations/deeptalk/DEE/checkpoint/DEE.pt
integrations/deeptalk/DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth
integrations/deeptalk/DEEPTalk/checkpoint/TH-VQVAE/TH-VQVAE.pth
integrations/deeptalk/FER/checkpoint/FER.pth
```

Current local sizes observed:

```text
DEE.pt        about 642M
DEEPTalk.pth  about 1.1G
TH-VQVAE.pth about 12M
FER.pth      about 5.4M
```

On the current server these are symlinks to:

```text
/scratch/e1554543/wav_to_flame/DEEPTalk_runs/repos/DEEPTalk/_downloads/DEEPTalk/
```

For a new server, either copy the real files into the expected paths or recreate
valid symlinks to a local checkpoint directory.

### 6.7 Gaussian Avatar Media

Current expected directory:

```text
integrations/gaussian_avatar/media/
```

Current local size observed: about `69M`.

Each runnable avatar needs at least:

```text
integrations/gaussian_avatar/media/<avatar_id>/point_cloud.ply
integrations/gaussian_avatar/media/<avatar_id>/flame_param.npz
```

Observed local avatar IDs include:

```text
306
1001
2001
2001_2
2002
2003
```

The default pipeline avatar is `306`.

Gaussian FLAME assets:

```text
integrations/gaussian_avatar/flame_model/assets/
```

Current local size observed: about `53M`.

### 6.8 VHAP Assets

Current expected directory:

```text
integrations/vhap/asset/
```

Current local size observed: about `102M`.

Important FLAME files:

```text
integrations/vhap/asset/flame/flame2023.pkl
integrations/vhap/asset/flame/FLAME_masks.pkl
integrations/vhap/asset/flame/landmark_embedding_with_eyes.npy
integrations/vhap/asset/flame/tex_mean_painted.png
integrations/vhap/asset/flame/uv_masks.npz
```

## 7. Path Configuration

Primary config file:

```text
src/avatar_system/pipeline_config.yaml
```

Most paths are written as `${PROJECT_ROOT}/...`. The config loader resolves
`PROJECT_ROOT` from the runtime environment. Make sure this points to the new
clone directory.

Key defaults:

```text
paths.perception_root       ${PROJECT_ROOT}/integrations/perception
paths.avamerg_root          ${PROJECT_ROOT}/integrations/avamerg
paths.emotivoice_root       ${PROJECT_ROOT}/integrations/emotivoice
paths.deeptalk_root         ${PROJECT_ROOT}/integrations/deeptalk/DEEPTalk
paths.gaussian_root         ${PROJECT_ROOT}/integrations/gaussian_avatar
paths.gaussian_container_image ${PROJECT_ROOT}/runtime/containers/gaussianav_jammy
runtime.run_root            ${PROJECT_ROOT}/runtime/outputs
```

If you move the repository, do not edit many files by hand first. Prefer:

```bash
export AVATAR_SYSTEM_ROOT=/new/path/avatar_system_full
export PROJECT_ROOT="$AVATAR_SYSTEM_ROOT"
```

Then only adjust `config/runtime.env` for persistent service use.

## 8. Service Ports

Default ports:

```text
7861  Main FastAPI studio UI
7862  Booth / 3DEPB UI
8788  EmotiVoice TTS worker
8789  AvaMERG worker
8790  DEEPTalk worker
8791  Perception worker
8792  Gaussian render worker
```

Keep worker ports bound to `127.0.0.1` unless you intentionally expose them
behind a firewall or reverse proxy.

## 9. Start and Verify

Check the script entrypoints:

```bash
bash scripts/avatar.sh --help
```

Start the full Booth stack. By default it auto-starts workers:

```bash
bash scripts/avatar.sh booth
```

Or start workers manually:

```bash
export DEPB_AUTO_START_WORKERS=0
bash scripts/avatar.sh worker tts
bash scripts/avatar.sh worker avamerg
bash scripts/avatar.sh worker deeptalk
bash scripts/avatar.sh worker perception
bash scripts/avatar.sh worker gaussian
bash scripts/avatar.sh booth
```

Health checks:

```bash
curl -fsS http://127.0.0.1:7862/
curl -fsS http://127.0.0.1:8788/health
curl -fsS http://127.0.0.1:8789/health
curl -fsS http://127.0.0.1:8790/health
curl -fsS http://127.0.0.1:8791/health
curl -fsS http://127.0.0.1:8792/health
```

Run a CLI smoke test:

```bash
PYTHONPATH=src "$AVATAR_SYSTEM_ROOT/runtime/cache/venvs/deeptalk/bin/python" -m avatar_system.pipeline.cli \
  --input_wav "$AVATAR_SYSTEM_ROOT/integrations/perception/data/demo_wavs/sample_dialog_02.wav" \
  --avatar_id 306 \
  --tts_speaker_id 6224 \
  --background study \
  --config "$AVATAR_SYSTEM_ROOT/src/avatar_system/pipeline_config.yaml"
```

The sample WAV above is an ignored local demo asset. If it is not restored on
the new server, replace `--input_wav` with any valid local WAV file.

Successful runs write:

```text
runtime/outputs/<run_id>/state.json
runtime/outputs/<run_id>/manifest.json
runtime/outputs/<run_id>/artifacts/manifest.json
runtime/outputs/<run_id>/artifacts/final_video.mp4
```

## 10. Troubleshooting

If a worker cannot find Python:

```bash
ls -l runtime/cache/venvs/web/bin/python
ls -l runtime/cache/venvs/perception/bin/python
ls -l runtime/cache/venvs/deeptalk/bin/python
ls -l integrations/avamerg/.avamerg38/bin/python
ls -l integrations/emotivoice/.EmotiVoice/bin/python
ls -l integrations/gaussian_avatar/.GSavatar_glibc/bin/python
```

If DEEPTalk fails to load checkpoints:

```bash
ls -lhL \
  integrations/deeptalk/DEE/checkpoint/DEE.pt \
  integrations/deeptalk/DEEPTalk/checkpoint/DEEPTalk/DEEPTalk.pth \
  integrations/deeptalk/DEEPTalk/checkpoint/TH-VQVAE/TH-VQVAE.pth \
  integrations/deeptalk/FER/checkpoint/FER.pth
```

If Gaussian rendering fails for the default avatar:

```bash
ls -lh \
  integrations/gaussian_avatar/media/306/point_cloud.ply \
  integrations/gaussian_avatar/media/306/flame_param.npz
```

If TTS fails to find the checkpoint:

```bash
ls -lh integrations/emotivoice/outputs/prompt_tts_open_source_joint/ckpt/g_00140000
```

If perception downloads models again, confirm cache paths:

```bash
echo "$XDG_CACHE_HOME"
echo "$MODELSCOPE_CACHE"
ls -lh runtime/cache/xdg/whisper/small.pt
ls -lh runtime/cache/modelscope/models/iic/emotion2vec_plus_seed/model.pt
```

If the container cannot see the project path, adjust Apptainer binds:

```bash
export APPTAINER_FLAGS="--nv -B $AVATAR_SYSTEM_ROOT:$AVATAR_SYSTEM_ROOT"
```

If the new server uses a different root outside `/scratch`, this is usually the
first thing to fix.
