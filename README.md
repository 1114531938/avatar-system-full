# Avatar System Full

`avatar_system_full` is the local full-stack avatar pipeline workspace.

## Layout

```text
avatar-system-full/
├── apps/
│   ├── web/                 # FastAPI + 7861 main frontend
│   └── booth/               # 7862 Booth / 3DEPB frontend or adapter
├── src/
│   └── avatar_system/
│       ├── agents/          # core agent architecture
│       ├── tools/           # TTS / AvaMERG / DEEPTalk / Gaussian wrappers
│       ├── pipeline/        # orchestrator, state, config
│       └── api/             # backend API route logic
├── integrations/
│   ├── avamerg/
│   ├── emotivoice/
│   ├── deeptalk/
│   ├── gaussian_avatar/
│   ├── vhap/
│   ├── 3depb/
│   └── perception/          # ASR / SER integration used by InputAgent
├── scripts/                 # unified startup entry
├── config/
├── docs/
├── runtime/                 # local runtime state, ignored by Git
│   ├── cache/
│   ├── data/
│   ├── outputs/
│   └── containers/
└── README.md
```

`runtime/cache/` also holds local virtual environments and model caches that
used to live under compatibility folders.

## Start

```bash
cd /scratch/e1554543/avatar_system_full

bash scripts/avatar.sh web
bash scripts/avatar.sh booth
bash scripts/avatar.sh agent
bash scripts/avatar.sh worker tts
bash scripts/avatar.sh worker avamerg
bash scripts/avatar.sh worker deeptalk
bash scripts/avatar.sh worker perception
bash scripts/avatar.sh worker gaussian
```

Default ports are documented in `docs/SERVICES_AND_PORTS.md`.

## Notes

- Source code lives in `apps/`, `src/avatar_system/`, and `integrations/`.
- Local data, generated outputs, caches, containers, venvs, and model files live
  under `runtime/` and are intentionally ignored.
- Historical compatibility directories have been removed from the project root.
