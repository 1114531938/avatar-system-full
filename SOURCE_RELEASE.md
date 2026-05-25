# Source Release Notes

This repository is the source-code release of the local avatar system workspace.
It intentionally does not contain downloaded models, licensed identity assets,
training data, generated media, Python environments, caches, or containers.

## Included

- End-to-end pipeline orchestration under `tools/avatar_agent/`
- Service and asset-building scripts under `scripts/`
- FastAPI web application and browser frontend under `web_app/`
- Perception integration code under `perception_layer/scripts/`
- Local integration changes and workers for AvaMERG, EmotiVoice, DEEPTalk,
  GaussianAvatars, and VHAP

## Excluded Assets

The following categories remain local and are ignored by Git:

- Model weights and checkpoints: `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- Trained avatars and motion assets: `*.ply`, `*.npz`, `data/`, `media/`
- NeRSemble and other downloaded datasets
- Virtual environments, Apptainer containers and model caches
- Runtime outputs, uploaded audio and exported videos
- Downloaded `ffmpeg` binaries and large upstream vendor/submodule trees

These files are either too large for a normal GitHub repository, generated at
runtime, or subject to their upstream dataset/model licenses.

## Local Runtime Layout

The runnable local deployment expects the excluded assets to be restored at
the same workspace-relative locations described in `README.md` and
`avatar_head_build_guide.md`. In particular, an available avatar must contain:

```text
GSavatar_runs/GaussianAvatars/media/<avatar_id>/
|-- point_cloud.ply
`-- flame_param.npz
```

For a fresh setup, obtain upstream model repositories and checkpoints under
their respective licenses, create the per-component Python environments, and
provide a CUDA-capable Apptainer runtime before starting the service.
