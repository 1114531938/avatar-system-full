# Services and Ports

## Main UI: 7861

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar.sh web
```

Default URL:

```text
http://localhost:7861/
```

This starts the unified FastAPI web server.

## Booth / 3DEPB UI: 7862

```bash
cd /scratch/e1554543/avatar_system_full
bash scripts/avatar.sh booth
```

Default URL:

```text
http://localhost:7862/
```

Current behavior:

- `scripts/avatar.sh booth` runs the FastAPI backend with Booth as the default route.
- `scripts/avatar.sh 3depb` runs the standalone Booth adapter in `apps/booth`.

If you want to serve the built-in Booth page from the FastAPI backend instead,
use:

```bash
PORT=7862 bash scripts/avatar.sh booth
```

The built-in Booth page is always available from the FastAPI backend at:

```text
/booth
```

The research studio page is always available at:

```text
/studio
```

## Worker Ports

```text
7861  FastAPI main studio UI
7862  Booth / 3DEPB UI
8788  EmotiVoice TTS worker
8789  AvaMERG worker
8790  DEEPTalk worker
8791  perception worker
8792  Gaussian render worker
```

## Service Commands

```bash
bash scripts/avatar.sh web
bash scripts/avatar.sh booth
bash scripts/avatar.sh 3depb
bash scripts/avatar.sh agent
```

Single worker commands:

```bash
bash scripts/avatar.sh worker tts
bash scripts/avatar.sh worker avamerg
bash scripts/avatar.sh worker deeptalk
bash scripts/avatar.sh worker perception
bash scripts/avatar.sh worker gaussian
```
