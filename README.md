# Avatar System Full

`avatar_system_full` is the full-stack workspace for the 3D Emotional Phone
Booth avatar system. It combines a browser booth UI, a FastAPI/studio backend,
an agent-based multimodal pipeline, and local model workers for perception,
dialogue planning, speech, facial motion, and Gaussian avatar rendering.

## System Overview

```text
Browser UI
  |-- apps/web/static/           Main studio UI served on 7861
  `-- apps/booth/                3DEPB booth UI served on 7862

Backend services
  |-- apps/web/server.py         FastAPI server, job API, settings, studio/booth routes
  `-- apps/booth/server.py       Standalone booth adapter, auth, history, assets, jobs

Agent pipeline
  `-- src/avatar_system/
      |-- pipeline/              CLI, config loading, state, manifest, orchestration
      |-- agents/                InputAgent, DialogueAgent, EmbodimentAgent
      `-- tools/                 Perception, AvaMERG, EmotiVoice, DEEPTalk, Gaussian wrappers

Model integrations
  |-- integrations/perception/   ASR, emotion recognition, Task1 input builder
  |-- integrations/avamerg/      Dialogue/reply generation and EmotiVoice input conversion
  |-- integrations/emotivoice/   TTS frontend and worker
  |-- integrations/deeptalk/     Audio-to-FLAME motion
  |-- integrations/gaussian_avatar/
  `-- integrations/vhap/         FLAME/mesh utilities and dataset helpers

Runtime state
  `-- runtime/                   Local cache, outputs, containers, venvs, sqlite data
```

`runtime/` is machine-local and ignored by Git. Source code and committed UI
assets live under `apps/`, `src/avatar_system/`, `integrations/`, `scripts/`,
`config/`, and `docs/`.

## Quick Start

```bash
cd /scratch/e1554543/avatar_system_full

bash scripts/avatar.sh web
bash scripts/avatar.sh booth
bash scripts/avatar.sh 3depb
bash scripts/avatar.sh agent
```

Worker commands:

```bash
bash scripts/avatar.sh worker tts
bash scripts/avatar.sh worker avamerg
bash scripts/avatar.sh worker deeptalk
bash scripts/avatar.sh worker perception
bash scripts/avatar.sh worker gaussian
```

Managed service commands:

```bash
bash scripts/avatar.sh service booth start
bash scripts/avatar.sh service booth status
bash scripts/avatar.sh service booth logs
bash scripts/avatar.sh service booth stop
```

`scripts/avatar.sh booth` auto-starts the local workers by default. Set
`DEPB_AUTO_START_WORKERS=0` to manage workers manually.

## Services and Ports

```text
7861  Main FastAPI studio UI
7862  Booth / 3DEPB UI
8788  EmotiVoice TTS worker
8789  AvaMERG worker
8790  DEEPTalk worker
8791  Perception worker
8792  Gaussian render worker
```

Default URLs:

```text
http://localhost:7861/
http://localhost:7862/
```

See `docs/SERVICES_AND_PORTS.md` for additional service notes.

## Pipeline Architecture

The runnable pipeline is organized as three high-level agents:

```text
InputAgent -> DialogueAgent -> EmbodimentAgent
```

The CLI entrypoint is:

```bash
PYTHONPATH=src python -m avatar_system.pipeline.cli \
  --input_wav /path/to/input.wav \
  --input_video /path/to/optional_user_video.webm \
  --avatar_id 306 \
  --tts_speaker_id 6224 \
  --background study \
  --config src/avatar_system/pipeline_config.yaml
```

### InputAgent

`src/avatar_system/agents/input_agent.py`

- Accepts audio and optional Booth user video.
- Extracts lightweight video frames with ffmpeg when video is provided.
- Runs `PerceptionTool`, which calls the perception worker or local scripts.
- Produces ASR, SER/emotion data, Task1 input, and
  `runtime/outputs/<run_id>/input/perception_result.json`.

### DialogueAgent

`src/avatar_system/agents/dialogue_agent.py`

- Runs `Task1Tool`, which uses AvaMERG or fallback reply generation.
- Builds an explainable reply plan with reply text, style, selected avatar,
  selected TTS speaker, background, and perception references.
- Writes `runtime/outputs/<run_id>/dialogue/reply_plan.json`.
- Keeps the old `PlanAgent` name as a compatibility alias.

### EmbodimentAgent

`src/avatar_system/agents/embodiment_agent.py`

- Converts the reply plan into EmotiVoice text/input JSON.
- Runs EmotiVoice TTS to synthesize the reply WAV.
- Runs DEEPTalk to produce audio-driven facial motion.
- Merges motion into FLAME/Gaussian-compatible parameters.
- Prepares viewer assets and exports final artifacts/video.
- Keeps the old `RenderAgent` name as a compatibility alias.

The orchestrator stage order is:

```text
input_agent
perception
task1
dialogue_agent
plan_agent
emotivoice_prepare
render_agent
emotivoice_tts
deeptalk
flame_merge
viewer
artifact_export
embodiment_agent
```

State is persisted to `runtime/outputs/<run_id>/state.json` after each stage so
jobs can be inspected and resumed safely.

## Runtime Artifacts

Each run writes a stable manifest:

```text
runtime/outputs/<run_id>/manifest.json
runtime/outputs/<run_id>/artifacts/manifest.json
```

Important manifest/state fields include:

```text
agent_pipeline_version
input_wav
input_video
video_frames_dir
perception_json
task1_input_json
perception_result_json
reply_plan_json
reply_text
reply_wav
deeptalk_npy
flame_motion_npz
point_cloud_path
output_video
output_white_model_video
video_export_error
finished_stages
```

Booth uploads, conversation history, recordings, generated video, TTS previews,
and service logs are stored under `runtime/outputs/` and `runtime/data/`.

## Booth Frontend

The Booth UI in `apps/booth/` provides:

- Guest and registered-user sessions.
- Avatar selection with configurable digital humans and voice speakers.
- Background selection and background image management.
- Automatic camera/microphone capture with silence-triggered submission.
- Conversation history, recording review, and history export.
- TTS voice preview caching.
- Video playback, 3D render view, point-cloud loading, frame rendering, and
  timeline controls.

Committed Booth configuration/assets:

```text
apps/booth/digital_humans.json
apps/booth/backgrounds.json
apps/booth/digital_human_images/
apps/booth/digital_human_backgrounds/
```

The generated avatar video is composited over the selected Booth background in
the browser. `apps/booth/app.js` draws the avatar video into a canvas and removes
the connected black backdrop while protecting the detected head/hair area. A
small alpha feather is applied at the matte boundary to reduce jagged edges.

## Backend APIs

`apps/web/server.py` is the main FastAPI service. It serves the studio, the
Booth route, settings, authentication, avatar metadata, TTS preview generation,
pipeline job creation, job status, viewer assets, and output files.

`apps/booth/server.py` is the standalone Booth adapter. It provides:

- `/api/auth/*` for login, registration, nickname, and logout.
- `/api/digital_humans` and `/api/backgrounds` for Booth configuration.
- `/api/tts_preview` for per-speaker preview audio.
- `/api/avatar/respond` and `/api/avatar/jobs/*` for async pipeline runs.
- `/api/avatar/runs/*/viewer_assets` and point-cloud routes for 3D preview.
- `/api/conversations`, `/api/history`, and `/api/recordings` for local history.

The standalone adapter stores its SQLite data and uploads under `runtime/`.

## Configuration

Primary pipeline configuration:

```text
src/avatar_system/pipeline_config.yaml
```

Important sections:

- `paths`: integration roots, venvs, output locations, container image.
- `env`: cache roots and LLM provider/model defaults.
- `perception`: ASR/SER model settings.
- `tts`: default EmotiVoice speaker and prompt mode.
- `*_worker`: worker URLs and timeouts.
- `merge`: FLAME jaw/expression tuning knobs.
- `runtime`: run root, viewer behavior, video export, fps, width, height.

Common environment variables:

```text
AVATAR_SYSTEM_ROOT
AVATAR_PYTHON
AGENT_PYTHON
AVATAR_CONTAINER
AVATAR_FFMPEG
AVATAR_FFPROBE
OPENAI_API_KEY
OPENAI_BASE_URL
LLM_MODEL
DEPB_AUTO_START_WORKERS
START_TTS_WORKER
START_AVAMERG_WORKER
START_DEEPTALK_WORKER
START_PERCEPTION_WORKER
START_GAUSSIAN_RENDER_WORKER
```

## Rendering Details

Final video export is handled by `src/avatar_system/export_gaussian_video.py`.
It loads the Gaussian avatar, applies merged FLAME motion, optionally overlays a
white mesh debug view, encodes a silent MP4, and muxes reply audio. Dimensions,
fps, and export mode are controlled by `runtime.video_width`,
`runtime.video_height`, `runtime.video_fps`, and related CLI flags.

The Gaussian render worker in `integrations/gaussian_avatar/` serves live render
requests for Booth viewer frames. The frontend can switch between video playback
and interactive 3D render assets for the same run.

## Development Notes

- Use `rg` to search the workspace; generated model/runtime data can be large.
- Keep model weights, caches, containers, virtual environments, and generated
  outputs under `runtime/` or integration-specific ignored directories.
- Do not add new source under removed compatibility paths.
- Prefer `scripts/avatar.sh` over ad hoc startup commands so cache paths,
  worker ports, thread limits, and container bindings remain consistent.
- Update this README when changing pipeline stages, ports, worker contracts,
  committed Booth assets, or runtime manifest fields.
