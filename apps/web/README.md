# Web App

`apps/web/server.py` is the unified FastAPI backend for the avatar system. It
serves static frontend files, accepts uploads, starts pipeline jobs, manages
Booth sessions, and proxies interactive Gaussian render requests.

Run this app through `scripts/avatar.sh web`.

## Frontend Pages

```text
apps/web/static/index.html             # Main research/studio UI
apps/web/static/style_commercial.css
apps/web/static/app.js

apps/web/static/booth.html             # Booth/video-call UI
apps/web/static/booth.css
apps/web/static/booth.js

apps/web/static/vendor/                # Browser-side 3D viewer dependencies
```

Routes:

```text
/          main studio by default, Booth if BOOTH_DEFAULT_ROUTE=1
/studio    always main studio
/booth     always built-in Booth page
/api/*     backend APIs
/outputs/* generated files exposed from runtime/outputs/
```

## Ports

- `7861`: main studio, via `scripts/avatar.sh web`
- `7862`: Booth/3DEPB path, via `scripts/avatar.sh booth` or `scripts/avatar.sh 3depb`

## Backend Responsibilities

- Auth and Booth session database under `runtime/outputs/booth/booth.sqlite3`
- Web uploads under `runtime/outputs/web_uploads/`
- Pipeline jobs under `runtime/outputs/web_<run_id>/`
- TTS preview files under `runtime/outputs/tts_previews/`
- Booth uploads and exports under `runtime/outputs/booth/`
- Interactive render frame API backed by the Gaussian render worker

## Development Notes

Frontend-only changes usually need a browser refresh. Backend, worker, or script
changes should use:

```bash
bash scripts/avatar.sh web
```
