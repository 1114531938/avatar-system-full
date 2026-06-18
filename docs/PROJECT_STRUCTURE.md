# Project Structure

The repository is organized around first-party app code, first-party pipeline
code, external integrations, and ignored runtime state.

```text
avatar-system-full/
|-- apps/
|   |-- web/                 # FastAPI + 7861 main frontend
|   `-- booth/               # 7862 Booth / 3DEPB frontend or adapter
|-- src/avatar_system/
|   |-- agents/              # agent implementations
|   |-- tools/               # integration tool wrappers
|   |-- pipeline/            # orchestrator, state, config, CLI
|   `-- api/                 # backend route package
|-- integrations/
|   |-- avamerg/
|   |-- emotivoice/
|   |-- deeptalk/
|   |-- gaussian_avatar/
|   |-- vhap/
|   |-- 3depb/
|   `-- perception/
|-- scripts/avatar.sh        # single supported startup entry
|-- config/
|-- docs/
|-- runtime/                 # ignored local cache/data/outputs/containers
`-- README.md
```

Do not add new source under removed compatibility paths. Use `runtime/` for
machine-local files and generated artifacts.
