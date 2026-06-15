# Three-Agent Pipeline Architecture

This project now keeps the existing runnable environments in place while making
the orchestration layer explicit:

```text
InputAgent -> DialogueAgent -> EmbodimentAgent
```

The agents live under `tools/avatar_agent/agents/`. The legacy names
`PlanAgent` and `RenderAgent` remain as compatibility aliases.

## Agent Responsibilities

`InputAgent`
: Prepares audio/video input, extracts lightweight video frames when a Booth
video is present, runs perception, and writes
`outputs/<run_id>/input/perception_result.json`.

`DialogueAgent`
: Runs Task1/AvaMERG or fallback reply generation, selects avatar/voice metadata,
and writes `outputs/<run_id>/dialogue/reply_plan.json`.

`EmbodimentAgent`
: Runs TTS, DEEPTalk, FLAME merge, viewer asset preparation, artifact export, and
writes the final manifest through the shared manifest utility.

## Stable Interfaces

The stable run-level manifest remains:

```text
outputs/<run_id>/manifest.json
outputs/<run_id>/artifacts/manifest.json
```

New fields are additive:

```text
agent_pipeline_version
perception_result_json
reply_plan_json
```

Existing fields such as `reply_text`, `reply_wav`, `flame_motion_npz`,
`output_video`, and `finished_stages` are preserved for Web compatibility.

## Migration Rule

The current phase does not move model repositories, virtual environments,
caches, datasets, or generated outputs. Later `src/avatar_system/` migration
should keep old entrypoints as shims until `scripts/run_agent.sh`, Web jobs, and
resume flows are verified.

Path configuration is centralized through `AVATAR_SYSTEM_ROOT` and
`${PROJECT_ROOT}` expansion in `tools/avatar_agent/pipeline/config.py`. The
default remains `/scratch/e1554543/avatar_system_full`.
