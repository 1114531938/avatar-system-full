#!/usr/bin/env python3
import atexit
import json
import hashlib
import mimetypes
import os
import signal
import secrets
import shlex
import shutil
import sqlite3
import subprocess
import threading
import uuid
import re
import urllib.error
import urllib.request
from base64 import b64decode
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from hashlib import pbkdf2_hmac
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
AVATAR_ROOT = Path(os.environ.get("AVATAR_SYSTEM_ROOT", "/scratch/e1554543/avatar_system_full"))
AVATAR_OUTPUT_ROOT = AVATAR_ROOT / "runtime" / "outputs"
AVATAR_UPLOAD_ROOT = AVATAR_OUTPUT_ROOT / "3depb_uploads"
TTS_PREVIEW_ROOT = AVATAR_OUTPUT_ROOT / "tts_previews"
GUEST_ARTIFACT_MARKER = ".guest_artifact"
AVATAR_SCRIPT = AVATAR_ROOT / "scripts" / "avatar.sh"
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "app.db"
RECORDINGS_DIR = ROOT / "recordings"
EXPORTS_DIR = ROOT / "exports"
DIGITAL_HUMAN_IMAGE_DIR = ROOT / "digital_human_images"
DIGITAL_HUMAN_BACKGROUND_DIR = ROOT / "digital_human_backgrounds"
BOOTH_BACKGROUNDS_PATH = ROOT / "backgrounds.json"
SESSION_COOKIE = "depb_session"
ITERATIONS = 180_000
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("DEPB_PIPELINE_TIMEOUT", "1800"))
PIPELINE_AVATAR_ID = os.environ.get("DEPB_AVATAR_ID", "306")
PIPELINE_TTS_SPEAKER_ID = os.environ.get("DEPB_TTS_SPEAKER_ID", "6224")
PIPELINE_NO_LLM = os.environ.get("DEPB_NO_LLM", "0") == "1" or not bool(os.environ.get("OPENAI_API_KEY"))
FFMPEG_BIN = Path(os.environ.get("DEPB_FFMPEG", str(AVATAR_ROOT / "runtime" / "cache" / "bin" / "ffmpeg")))
PIPELINE_STAGES = [
    ("input_agent", "Preparing video and audio input"),
    ("perception", "Analyzing your video and voice"),
    ("task1", "Generating emotional reply plan"),
    ("dialogue_agent", "Preparing response text"),
    ("plan_agent", "Selecting avatar and voice"),
    ("emotivoice_prepare", "Preparing TTS text"),
    ("render_agent", "Preparing avatar render"),
    ("emotivoice_tts", "Synthesizing companion voice"),
    ("deeptalk", "Driving facial motion"),
    ("flame_merge", "Merging avatar motion"),
    ("viewer", "Preparing viewer assets"),
    ("artifact_export", "Rendering digital human video"),
    ("embodiment_agent", "Finalizing avatar result"),
]
PIPELINE_STAGE_LABELS = dict(PIPELINE_STAGES)
PIPELINE_JOBS = {}
PIPELINE_JOBS_LOCK = threading.Lock()
SUBTITLE_TRANSLATION_CACHE = {}
WORKER_HEALTH_URLS = {
    "tts": os.environ.get("TTS_WORKER_URL", "http://127.0.0.1:8788").rstrip("/"),
    "avamerg": os.environ.get("AVAMERG_WORKER_URL", "http://127.0.0.1:8789").rstrip("/"),
    "deeptalk": os.environ.get("DEEPTALK_WORKER_URL", "http://127.0.0.1:8790").rstrip("/"),
    "perception": os.environ.get("PERCEPTION_WORKER_URL", "http://127.0.0.1:8791").rstrip("/"),
    "gaussian": os.environ.get("GAUSSIAN_RENDER_WORKER_URL", "http://127.0.0.1:8792").rstrip("/"),
}
TTS_PREVIEW_TEXT = "Hello, I am your digital human companion. I am ready to speak with you in English."
AVATARS = {
    "companion": {
        "name": "Emotional Companion",
        "reply": "",
    },
    "mentor": {
        "name": "Growth Mentor",
        "reply": "",
    },
    "friend": {
        "name": "Close Friend",
        "reply": "",
    },
    "coach": {
        "name": "Wellbeing Coach",
        "reply": "",
    },
}
AVATAR_COLORS = ["#32d0a4", "#6fb7ff", "#ffb84d", "#f0798d", "#a78bfa", "#22c55e", "#f97316", "#06b6d4"]
HIDDEN_AVATAR_IDS = {"1001", "2001_2"}
DEFAULT_BOOTH_BACKGROUNDS = [
    {"id": "study", "label": "Study", "image_url": ""},
    {"id": "bedroom", "label": "Bedroom", "image_url": ""},
    {"id": "sofa", "label": "Sofa", "image_url": ""},
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def looks_chinese_text(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def translate_subtitle_to_english(text):
    text = (text or "").strip()
    if not text or not looks_chinese_text(text):
        return text
    if text in SUBTITLE_TRANSLATION_CACHE:
        return SUBTITLE_TRANSLATION_CACHE[text]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate this empathetic subtitle into natural spoken English. "
                    "Preserve the meaning and warmth. Return only the English subtitle."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        translated = data["choices"][0]["message"]["content"].strip().strip("\"'")
    except (OSError, urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return ""
    if not translated or looks_chinese_text(translated):
        return ""
    SUBTITLE_TRANSLATION_CACHE[text] = translated
    return translated


def english_subtitle_text(reply_text="", preferred_text=""):
    preferred_text = (preferred_text or "").strip()
    if preferred_text and not looks_chinese_text(preferred_text):
        return preferred_text
    fixed_fallbacks = {
        "谢谢你愿意和我说这些。你不用着急，想从哪里开始都可以。我会认真听你慢慢说。": (
            "Thank you for sharing this with me. There's no rush; you can start wherever you feel comfortable, and I'll listen carefully."
        )
    }
    if preferred_text in fixed_fallbacks:
        return fixed_fallbacks[preferred_text]
    translated = translate_subtitle_to_english(preferred_text or reply_text)
    if translated:
        return translated
    reply_text = (reply_text or "").strip()
    if reply_text and not looks_chinese_text(reply_text):
        return reply_text
    if reply_text in fixed_fallbacks:
        return fixed_fallbacks[reply_text]
    return ""


def read_task1_tts_text(path):
    if not path:
        return ""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for key in ("subtitle_text", "tts_text", "spoken_text", "english_tts_text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = data.get("raw_result")
    if isinstance(raw, dict):
        for key in ("subtitle_text", "tts_text", "spoken_text", "english_tts_text"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def mark_guest_artifact(path, job_id="", guest_id=""):
    try:
        path.mkdir(parents=True, exist_ok=True)
        (path / GUEST_ARTIFACT_MARKER).write_text(
            json.dumps({"jobId": job_id, "guestId": guest_id, "createdAt": now_iso()}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def cleanup_guest_artifacts():
    roots = [AVATAR_UPLOAD_ROOT]
    if AVATAR_OUTPUT_ROOT.exists():
        roots.extend(path for path in AVATAR_OUTPUT_ROOT.glob("3depb_*") if path.is_dir())
    deleted = 0
    for path in roots:
        marker = path / GUEST_ARTIFACT_MARKER
        if not marker.exists():
            if path == AVATAR_UPLOAD_ROOT and path.exists():
                for child in path.iterdir():
                    if child.is_dir() and (child / GUEST_ARTIFACT_MARKER).exists():
                        try:
                            shutil.rmtree(child)
                            deleted += 1
                        except OSError:
                            pass
            continue
        try:
            shutil.rmtree(path)
            deleted += 1
        except OSError:
            pass
    if deleted:
        print(f"[3depb] cleaned {deleted} guest artifact director{'y' if deleted == 1 else 'ies'}", flush=True)


def hash_password(password, salt=None):
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, ITERATIONS)
    return salt_bytes.hex(), digest.hex()


def public_user(row):
    return {
        "id": row["id"],
        "username": row["username"],
        "nickname": row["nickname"] or "",
        "role": row["role"],
        "status": row["status"],
        "createdAt": row["created_at"],
        "lastLoginAt": row["last_login_at"] or "",
    }


def clean_guest_name(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:32] or "Guest"


def clean_background_label(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:32]


def background_id_from_label(label):
    base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return base[:32] or f"background-{uuid.uuid4().hex[:8]}"


def list_booth_backgrounds():
    items = []
    if BOOTH_BACKGROUNDS_PATH.exists():
        try:
            payload = json.loads(BOOTH_BACKGROUNDS_PATH.read_text(encoding="utf-8"))
            raw_items = payload.get("backgrounds") if isinstance(payload, dict) else payload
            if isinstance(raw_items, list):
                items = [item for item in raw_items if isinstance(item, dict)]
        except json.JSONDecodeError:
            items = []
    if not items:
        items = DEFAULT_BOOTH_BACKGROUNDS

    seen = set()
    normalized = []
    for item in items:
        background_id = str(item.get("id") or background_id_from_label(item.get("label") or "")).strip()
        background_id = re.sub(r"[^A-Za-z0-9_.-]", "-", background_id)[:48] or f"background-{uuid.uuid4().hex[:8]}"
        if background_id in seen:
            continue
        seen.add(background_id)
        normalized.append(
            {
                "id": background_id,
                "label": clean_background_label(item.get("label") or item.get("name") or background_id) or background_id.title(),
                "image_url": str(item.get("image_url") or item.get("imageUrl") or ""),
            }
        )
    return normalized or DEFAULT_BOOTH_BACKGROUNDS


def save_booth_backgrounds(items):
    BOOTH_BACKGROUNDS_PATH.write_text(
        json.dumps({"backgrounds": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def public_background(item):
    return {
        "id": item["id"],
        "label": item["label"],
        "imageUrl": item.get("image_url") or "",
    }


def clean_booth_background(value):
    background = str(value or "").strip()
    valid = {item["id"] for item in list_booth_backgrounds()}
    return background if background in valid else "study"


def load_avatar_labels():
    labels = {
        "306": "Avatar 306",
        "1001": "Avatar 1001",
    }
    map_path = (
        AVATAR_ROOT
        / "integrations"
        / "gaussian_avatar"
        / "datasets"
        / "nersemble_preprocessed"
        / "nersemble_avatar_map.tsv"
    )
    if map_path.exists():
        for line in map_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                subject, _source_dir, avatar_id = parts[:3]
                labels[str(avatar_id)] = f"NeRSemble {subject}"
    return labels


def speaker_wiki_rows():
    wiki_path = AVATAR_ROOT / "integrations" / "emotivoice" / "data" / "youdao" / "text" / "README.md"
    rows = {}
    if not wiki_path.exists():
        return rows
    pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|")
    for line in wiki_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        speaker_id, name, gender, description = [part.strip() for part in match.groups()]
        if speaker_id.isdigit():
            rows[speaker_id] = {"name": name, "gender": gender, "description": description}
    return rows


def list_tts_speakers():
    speaker_path = AVATAR_ROOT / "integrations" / "emotivoice" / "data" / "youdao" / "text" / "speaker2"
    wiki = speaker_wiki_rows()
    if not speaker_path.exists():
        return [{"id": PIPELINE_TTS_SPEAKER_ID, "label": f"{PIPELINE_TTS_SPEAKER_ID} · voice"}]
    speakers = []
    for line in speaker_path.read_text(encoding="utf-8", errors="replace").splitlines():
        speaker_id = line.strip()
        if not speaker_id:
            continue
        meta = wiki.get(speaker_id, {})
        label_parts = [speaker_id]
        for key in ("name", "gender", "description"):
            if meta.get(key):
                label_parts.append(meta[key])
        speakers.append({"id": speaker_id, "label": " · ".join(label_parts), **meta})
    return speakers


def load_digital_human_overrides():
    for path in (ROOT / "digital_humans.json", DB_DIR / "digital_humans.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            humans = payload.get("digital_humans", payload)
            if isinstance(humans, list):
                return {str(item.get("avatar_id") or item.get("id")): item for item in humans if isinstance(item, dict)}
            if isinstance(humans, dict):
                return {str(key): value for key, value in humans.items() if isinstance(value, dict)}
    return {}


def digital_humans_config_path():
    return ROOT / "digital_humans.json"


def save_digital_human_overrides(overrides):
    path = digital_humans_config_path()
    current = {item["id"]: item for item in list_digital_humans()}
    humans = []
    for avatar_id, item in current.items():
        override = dict(overrides.get(avatar_id, {}))
        humans.append(
            {
                "avatar_id": avatar_id,
                "tts_speaker_id": str(override.get("tts_speaker_id") or item["ttsSpeakerId"]),
                "name": str(override.get("name") or item["name"]),
                "role": str(override.get("role") or item["role"]),
                "color": str(override.get("color") or item["color"]),
                "image_url": str(override.get("image_url") or item.get("imageUrl") or ""),
                "background_url": str(override.get("background_url") or item.get("backgroundUrl") or ""),
            }
        )
    path.write_text(json.dumps({"digital_humans": humans}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_digital_humans():
    media_root = AVATAR_ROOT / "integrations" / "gaussian_avatar" / "media"
    labels = load_avatar_labels()
    overrides = load_digital_human_overrides()
    speakers = {item["id"]: item for item in list_tts_speakers()}
    humans = []
    if media_root.exists():
        def avatar_sort_key(item):
            if item.name == PIPELINE_AVATAR_ID:
                return (0, item.name)
            return (1, item.name)

        for index, path in enumerate(sorted(media_root.iterdir(), key=avatar_sort_key)):
            if not path.is_dir() or not (path / "point_cloud.ply").exists() or not (path / "flame_param.npz").exists():
                continue
            avatar_id = path.name
            if avatar_id in HIDDEN_AVATAR_IDS:
                continue
            override = overrides.get(avatar_id, {})
            speaker_id = str(override.get("tts_speaker_id") or override.get("speaker_id") or PIPELINE_TTS_SPEAKER_ID)
            speaker = speakers.get(speaker_id, {"id": speaker_id, "label": f"{speaker_id} · voice"})
            name = str(override.get("name") or override.get("label") or labels.get(avatar_id) or f"Avatar {avatar_id}")
            role = str(override.get("role") or f"Avatar {avatar_id} · Voice {speaker_id}")
            humans.append(
                {
                    "id": avatar_id,
                    "avatarId": avatar_id,
                    "ttsSpeakerId": speaker_id,
                    "name": name,
                    "role": role,
                    "speakerLabel": speaker.get("label", speaker_id),
                    "color": str(override.get("color") or AVATAR_COLORS[index % len(AVATAR_COLORS)]),
                    "imageUrl": str(override.get("image_url") or override.get("imageUrl") or ""),
                    "backgroundUrl": str(override.get("background_url") or override.get("backgroundUrl") or ""),
                    "reply": str(override.get("reply") or ""),
                }
            )
    if not humans:
        humans.append(
            {
                "id": PIPELINE_AVATAR_ID,
                "avatarId": PIPELINE_AVATAR_ID,
                "ttsSpeakerId": PIPELINE_TTS_SPEAKER_ID,
                "name": f"Avatar {PIPELINE_AVATAR_ID}",
                "role": f"Avatar {PIPELINE_AVATAR_ID} · Voice {PIPELINE_TTS_SPEAKER_ID}",
                "speakerLabel": f"{PIPELINE_TTS_SPEAKER_ID} · voice",
                "color": AVATAR_COLORS[0],
                "imageUrl": "",
                "backgroundUrl": "",
                "reply": "",
            }
        )
    return humans


def worker_health(timeout=0.5):
    health = {}
    for name, url in WORKER_HEALTH_URLS.items():
        item = {"ok": False, "url": url, "error": ""}
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            item["ok"] = bool(payload.get("ok"))
            if not item["ok"]:
                item["error"] = str(payload)
        except Exception as exc:
            item["error"] = str(exc)
        health[name] = item
    return health


def worker_health_summary(health):
    offline = [name for name, item in health.items() if not item.get("ok")]
    if not offline:
        return ""
    return "Workers offline: " + ", ".join(offline)


def avatar_meta(avatar_id):
    for human in list_digital_humans():
        if str(human["id"]) == str(avatar_id):
            return {"name": human["name"], "reply": human["reply"]}
    legacy = AVATARS.get(str(avatar_id))
    if legacy:
        return legacy
    return {"name": f"Avatar {avatar_id}", "reply": ""}


def conversation_payload(row):
    avatar = avatar_meta(row["avatar_id"])
    subtitle_text = ""
    if "subtitle_text" in row.keys():
        subtitle_text = row["subtitle_text"] or ""
    if not subtitle_text:
        subtitle_text = english_subtitle_text(row["reply_text"] or "", row["reply_text"] or "")
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "username": row["username"],
        "avatarId": row["avatar_id"],
        "avatarName": avatar["name"],
        "userText": row["user_text"],
        "replyText": subtitle_text or row["reply_text"],
        "subtitleText": subtitle_text,
        "userVideoUrl": row["user_video_url"],
        "videoUrl": row["video_url"],
        "audioUrl": row["audio_url"],
        "combinedVideoUrl": row["combined_video_url"] if "combined_video_url" in row.keys() else "",
        "createdAt": row["created_at"],
    }


def recording_payload(row):
    avatar = avatar_meta(row["avatar_id"])
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "username": row["username"],
        "avatarId": row["avatar_id"],
        "avatarName": avatar["name"],
        "url": row["url"],
        "mimeType": row["mime_type"],
        "sizeBytes": row["size_bytes"],
        "createdAt": row["created_at"],
    }


def init_db():
    DB_DIR.mkdir(exist_ok=True)
    RECORDINGS_DIR.mkdir(exist_ok=True)
    EXPORTS_DIR.mkdir(exist_ok=True)
    DIGITAL_HUMAN_IMAGE_DIR.mkdir(exist_ok=True)
    DIGITAL_HUMAN_BACKGROUND_DIR.mkdir(exist_ok=True)
    if not BOOTH_BACKGROUNDS_PATH.exists():
        save_booth_backgrounds(DEFAULT_BOOTH_BACKGROUNDS)
    AVATAR_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id TEXT PRIMARY KEY,
              username TEXT NOT NULL UNIQUE,
              nickname TEXT NOT NULL DEFAULT '',
              password_salt TEXT NOT NULL,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
              status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
              created_at TEXT NOT NULL,
              last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
              token TEXT PRIMARY KEY,
              user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              avatar_id TEXT NOT NULL,
              user_text TEXT NOT NULL,
              reply_text TEXT NOT NULL,
              subtitle_text TEXT NOT NULL DEFAULT '',
              user_video_url TEXT NOT NULL DEFAULT '',
              video_url TEXT NOT NULL DEFAULT '',
              audio_url TEXT NOT NULL DEFAULT '',
              combined_video_url TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recordings (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              avatar_id TEXT NOT NULL,
              file_name TEXT NOT NULL,
              url TEXT NOT NULL,
              mime_type TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            for username, password, role in (
                ("admin", "admin123", "admin"),
                ("user", "user123", "user"),
            ):
                salt, digest = hash_password(password)
                conn.execute(
                    """
                    INSERT INTO users
                      (id, username, password_salt, password_hash, role, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (str(uuid.uuid4()), username, salt, digest, role, now_iso()),
                )
        ensure_column(conn, "conversations", "user_video_url", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "conversations", "combined_video_url", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "conversations", "subtitle_text", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "users", "nickname", "TEXT NOT NULL DEFAULT ''")
        backfill_conversation_subtitles(conn)
        backfill_user_video_urls(conn)


def ensure_column(conn, table, column, definition):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def backfill_user_video_urls(conn):
    rows = conn.execute(
        """
        SELECT id
        FROM conversations
        WHERE COALESCE(user_video_url, '') = ''
        """
    ).fetchall()
    for row in rows:
        upload_dir = AVATAR_UPLOAD_ROOT / row["id"]
        input_video = next(iter(sorted(upload_dir.glob("input_video.*"))), None)
        if input_video and input_video.exists():
            rel = input_video.resolve().relative_to(AVATAR_OUTPUT_ROOT.resolve())
            conn.execute(
                "UPDATE conversations SET user_video_url = ? WHERE id = ?",
                ("/outputs/" + str(rel).replace(os.sep, "/"), row["id"]),
            )


def backfill_conversation_subtitles(conn):
    rows = conn.execute(
        """
        SELECT id, reply_text
        FROM conversations
        WHERE COALESCE(subtitle_text, '') = ''
        """
    ).fetchall()
    for row in rows:
        subtitle_text = english_subtitle_text(row["reply_text"] or "", row["reply_text"] or "")
        if subtitle_text:
            conn.execute("UPDATE conversations SET subtitle_text = ? WHERE id = ?", (subtitle_text, row["id"]))


class Handler(SimpleHTTPRequestHandler):
    def do_HEAD(self):
        self.serve_static(urlparse(self.path).path, head_only=True)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/auth/me":
            self.handle_me()
            return
        if path.startswith("/api/avatar/jobs/"):
            self.handle_avatar_job(path.rsplit("/", 1)[-1])
            return
        if path.startswith("/api/avatar/runs/") and path.endswith("/viewer_assets"):
            run_id = path.strip("/").split("/")[3]
            self.handle_viewer_assets(run_id)
            return
        if path.startswith("/api/avatar/runs/") and path.endswith("/point_cloud"):
            run_id = path.strip("/").split("/")[3]
            self.handle_point_cloud(run_id)
            return
        if path.startswith("/api/jobs/") and path.endswith("/viewer_assets"):
            run_id = path.strip("/").split("/")[2]
            self.handle_viewer_assets(run_id)
            return
        if path.startswith("/api/jobs/") and path.endswith("/viewer/point_cloud"):
            run_id = path.strip("/").split("/")[2]
            self.handle_point_cloud(run_id)
            return
        if path == "/api/users":
            self.handle_list_users()
            return
        if path == "/api/digital_humans":
            self.handle_list_digital_humans()
            return
        if path == "/api/backgrounds":
            self.handle_list_backgrounds()
            return
        if path == "/api/tts_speakers":
            self.handle_list_tts_speakers()
            return
        if path == "/api/tts_preview":
            self.handle_tts_preview()
            return
        if path == "/api/conversations":
            self.handle_list_conversations()
            return
        if path == "/api/recordings":
            self.handle_list_recordings()
            return
        self.serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/auth/login":
            self.handle_login()
            return
        if path == "/api/auth/register":
            self.handle_register()
            return
        if path == "/api/auth/logout":
            self.handle_logout()
            return
        if path == "/api/users":
            self.handle_create_user()
            return
        if path == "/api/avatar/respond":
            self.handle_avatar_respond()
            return
        if path.startswith("/api/avatar/jobs/") and path.endswith("/cancel"):
            job_id = path.strip("/").split("/")[3]
            self.handle_cancel_avatar_job(job_id)
            return
        if path == "/api/recordings":
            self.handle_create_recording()
            return
        if path == "/api/history/export":
            self.handle_export_history()
            return
        if path == "/api/backgrounds":
            self.handle_create_background()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "backgrounds"] and parts[3] == "image":
            self.handle_upload_background_image(parts[2])
            return
        if len(parts) == 4 and parts[:2] == ["api", "digital_humans"] and parts[3] == "image":
            self.handle_upload_digital_human_image(parts[2])
            return
        if len(parts) == 4 and parts[:2] == ["api", "digital_humans"] and parts[3] == "background":
            self.handle_upload_digital_human_background(parts[2])
            return
        if path.startswith("/api/jobs/") and path.endswith("/viewer/render_frame"):
            run_id = path.strip("/").split("/")[2]
            self.handle_render_frame(run_id)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_PATCH(self):
        path = urlparse(self.path).path
        if path == "/api/auth/nickname":
            self.handle_update_nickname()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "users"]:
            if parts[3] == "password":
                self.handle_reset_password(parts[2])
                return
            if parts[3] == "status":
                self.handle_update_status(parts[2])
                return
        if len(parts) == 4 and parts[:2] == ["api", "digital_humans"] and parts[3] == "speaker":
            self.handle_update_digital_human_speaker(parts[2])
            return
        if len(parts) == 3 and parts[:2] == ["api", "digital_humans"]:
            self.handle_update_digital_human_profile(parts[2])
            return
        if len(parts) == 3 and parts[:2] == ["api", "backgrounds"]:
            self.handle_update_background(parts[2])
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if path == "/api/history":
            self.handle_clear_history()
            return
        if len(parts) == 3 and parts[:2] == ["api", "users"]:
            self.handle_delete_user(parts[2])
            return
        if len(parts) == 3 and parts[:2] == ["api", "backgrounds"]:
            self.handle_delete_background(parts[2])
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return None

    def read_multipart(self):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")
        message = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )

        fields = {}
        files = {}
        if message.is_multipart():
            for part in message.iter_parts():
                disposition = part.get_content_disposition()
                field_name = part.get_param("name", header="content-disposition")
                if disposition != "form-data" or not field_name:
                    continue
                payload = part.get_payload(decode=True) or b""
                filename = part.get_filename()
                if filename:
                    files[field_name] = {
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "data": payload,
                    }
                else:
                    fields[field_name] = payload.decode(part.get_content_charset() or "utf-8")
        return fields, files

    def send_json(self, payload, status=HTTPStatus.OK, headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def current_user(self):
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get(SESSION_COOKIE)
        if not morsel:
            return None
        with connect() as conn:
            return conn.execute(
                """
                SELECT users.*
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = ?
                """,
                (morsel.value,),
            ).fetchone()

    def require_user(self):
        user = self.current_user()
        if not user:
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return None
        if user["status"] != "active":
            self.send_json({"error": "Account disabled"}, HTTPStatus.FORBIDDEN)
            return None
        return user

    def require_admin(self):
        user = self.require_user()
        if not user:
            return None
        if user["role"] != "admin":
            self.send_json({"error": "Admin only"}, HTTPStatus.FORBIDDEN)
            return None
        return user

    def handle_login(self):
        data = self.read_json()
        if data is None:
            return
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        with connect() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not user:
                self.send_json({"error": "Incorrect username or password"}, HTTPStatus.UNAUTHORIZED)
                return
            salt, digest = hash_password(password, user["password_salt"])
            if digest != user["password_hash"]:
                self.send_json({"error": "Incorrect username or password"}, HTTPStatus.UNAUTHORIZED)
                return
            if user["status"] != "active":
                self.send_json({"error": "This account has been disabled"}, HTTPStatus.FORBIDDEN)
                return

            token = secrets.token_urlsafe(32)
            login_at = now_iso()
            conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user["id"], login_at))
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (login_at, user["id"]))
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()

        self.send_json(
            {"user": public_user(user)},
            headers={"Set-Cookie": f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/"},
        )

    def handle_register(self):
        data = self.read_json()
        if data is None:
            return
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", "")).strip()
        if not username or not password:
            self.send_json({"error": "Username and password are required"}, HTTPStatus.BAD_REQUEST)
            return
        if len(username) < 3:
            self.send_json({"error": "Username must be at least 3 characters"}, HTTPStatus.BAD_REQUEST)
            return
        if len(password) < 6:
            self.send_json({"error": "Password must be at least 6 characters"}, HTTPStatus.BAD_REQUEST)
            return

        salt, digest = hash_password(password)
        user_id = str(uuid.uuid4())
        try:
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users
                      (id, username, password_salt, password_hash, role, status, created_at)
                    VALUES (?, ?, ?, ?, 'user', 'active', ?)
                    """,
                    (user_id, username, salt, digest, now_iso()),
                )
                token = secrets.token_urlsafe(32)
                login_at = now_iso()
                conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, login_at))
                conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (login_at, user_id))
                user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        except sqlite3.IntegrityError:
            self.send_json({"error": "This username already exists"}, HTTPStatus.CONFLICT)
            return

        self.send_json(
            {"user": public_user(user)},
            headers={"Set-Cookie": f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/"},
        )

    def handle_logout(self):
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        token = cookie.get(SESSION_COOKIE)
        if token:
            with connect() as conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token.value,))
        self.send_json(
            {"ok": True},
            headers={"Set-Cookie": f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"},
        )

    def handle_update_nickname(self):
        user = self.require_user()
        if not user:
            return
        data = self.read_json()
        if data is None:
            return
        nickname = clean_guest_name(data.get("nickname") or "")
        with connect() as conn:
            conn.execute("UPDATE users SET nickname = ? WHERE id = ?", (nickname, user["id"]))
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        self.send_json({"user": public_user(user)})

    def handle_me(self):
        user = self.require_user()
        if user:
            self.send_json({"user": public_user(user)})

    def handle_list_users(self):
        if not self.require_admin():
            return
        with connect() as conn:
            users = conn.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
        self.send_json({"users": [public_user(user) for user in users]})

    def handle_create_user(self):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", "")).strip()
        role = str(data.get("role", "user"))
        if not username or not password:
            self.send_json({"error": "Username and password are required"}, HTTPStatus.BAD_REQUEST)
            return
        if role not in ("admin", "user"):
            self.send_json({"error": "Invalid role"}, HTTPStatus.BAD_REQUEST)
            return

        salt, digest = hash_password(password)
        try:
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users
                      (id, username, password_salt, password_hash, role, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (str(uuid.uuid4()), username, salt, digest, role, now_iso()),
                )
        except sqlite3.IntegrityError:
            self.send_json({"error": "This username already exists"}, HTTPStatus.CONFLICT)
            return
        self.handle_list_users()

    def handle_reset_password(self, user_id):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        password = str(data.get("password", "")).strip()
        if not password:
            self.send_json({"error": "New password is required"}, HTTPStatus.BAD_REQUEST)
            return
        salt, digest = hash_password(password)
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        current_token = cookie.get(SESSION_COOKIE)
        current_token = current_token.value if current_token else ""
        with connect() as conn:
            conn.execute(
                "UPDATE users SET password_salt = ?, password_hash = ? WHERE id = ?",
                (salt, digest, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ? AND token != ?", (user_id, current_token))
        self.handle_list_users()

    def handle_update_status(self, user_id):
        admin = self.require_admin()
        if not admin:
            return
        data = self.read_json()
        if data is None:
            return
        status = str(data.get("status", ""))
        if status not in ("active", "disabled"):
            self.send_json({"error": "Invalid status"}, HTTPStatus.BAD_REQUEST)
            return
        if user_id == admin["id"] and status == "disabled":
            self.send_json({"error": "You cannot disable the currently signed-in administrator account"}, HTTPStatus.BAD_REQUEST)
            return
        with connect() as conn:
            conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
            if status == "disabled":
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        self.handle_list_users()

    def handle_delete_user(self, user_id):
        admin = self.require_admin()
        if not admin:
            return
        if user_id == admin["id"]:
            self.send_json({"error": "You cannot delete the currently signed-in administrator account"}, HTTPStatus.BAD_REQUEST)
            return
        with connect() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.handle_list_users()

    def handle_list_digital_humans(self):
        self.send_json({"digitalHumans": list_digital_humans()})

    def handle_list_backgrounds(self):
        self.send_json({"backgrounds": [public_background(item) for item in list_booth_backgrounds()]})

    def handle_create_background(self):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        label = clean_background_label(data.get("label") or data.get("name") or "")
        if not label:
            self.send_json({"error": "Background name is required"}, HTTPStatus.BAD_REQUEST)
            return
        items = list_booth_backgrounds()
        existing_ids = {item["id"] for item in items}
        background_id = background_id_from_label(label)
        if background_id in existing_ids:
            background_id = f"{background_id}-{uuid.uuid4().hex[:6]}"
        item = {"id": background_id, "label": label, "image_url": ""}
        items.append(item)
        save_booth_backgrounds(items)
        self.send_json({"background": public_background(item), "backgrounds": [public_background(bg) for bg in items]})

    def handle_update_background(self, background_id):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        items = list_booth_backgrounds()
        for item in items:
            if item["id"] != background_id:
                continue
            label = clean_background_label(data.get("label") or data.get("name") or item["label"])
            if not label:
                self.send_json({"error": "Background name is required"}, HTTPStatus.BAD_REQUEST)
                return
            item["label"] = label
            save_booth_backgrounds(items)
            self.send_json({"background": public_background(item), "backgrounds": [public_background(bg) for bg in items]})
            return
        self.send_json({"error": f"Unknown background id: {background_id}"}, HTTPStatus.NOT_FOUND)

    def handle_delete_background(self, background_id):
        if not self.require_admin():
            return
        if background_id in {item["id"] for item in DEFAULT_BOOTH_BACKGROUNDS}:
            self.send_json({"error": "Default backgrounds cannot be deleted"}, HTTPStatus.BAD_REQUEST)
            return
        items = list_booth_backgrounds()
        next_items = [item for item in items if item["id"] != background_id]
        if len(next_items) == len(items):
            self.send_json({"error": f"Unknown background id: {background_id}"}, HTTPStatus.NOT_FOUND)
            return
        save_booth_backgrounds(next_items)
        self.send_json({"ok": True, "backgrounds": [public_background(bg) for bg in next_items]})

    def handle_upload_background_image(self, background_id):
        if not self.require_admin():
            return
        items = list_booth_backgrounds()
        target_item = next((item for item in items if item["id"] == background_id), None)
        if not target_item:
            self.send_json({"error": f"Unknown background id: {background_id}"}, HTTPStatus.NOT_FOUND)
            return
        _fields, files = self.read_multipart()
        image_file = files.get("background") or files.get("image") or files.get("file")
        if not image_file or not image_file.get("data"):
            self.send_json({"error": "Background image file is required"}, HTTPStatus.BAD_REQUEST)
            return
        content_type = image_file.get("content_type") or ""
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(content_type)
        if not extension:
            suffix = Path(image_file.get("filename") or "").suffix.lower()
            extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ""
        if not extension:
            self.send_json({"error": "Only jpg, png, or webp background images are supported"}, HTTPStatus.BAD_REQUEST)
            return
        if len(image_file["data"]) > 12 * 1024 * 1024:
            self.send_json({"error": "Background image cannot exceed 12MB"}, HTTPStatus.BAD_REQUEST)
            return

        DIGITAL_HUMAN_BACKGROUND_DIR.mkdir(exist_ok=True)
        filename = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', background_id)}_{uuid.uuid4().hex[:8]}{extension}"
        target = DIGITAL_HUMAN_BACKGROUND_DIR / filename
        target.write_bytes(image_file["data"])
        image_url = f"/digital_human_backgrounds/{filename}"
        target_item["image_url"] = image_url
        save_booth_backgrounds(items)
        self.send_json({"imageUrl": image_url, "background": public_background(target_item), "backgrounds": [public_background(bg) for bg in items]})

    def handle_list_tts_speakers(self):
        user = self.require_user()
        if not user:
            return
        self.send_json({"speakers": list_tts_speakers(), "defaultSpeakerId": PIPELINE_TTS_SPEAKER_ID})

    def handle_tts_preview(self):
        query = parse_qs(urlparse(self.path).query)
        speaker_id = str((query.get("speakerId") or query.get("ttsSpeakerId") or [PIPELINE_TTS_SPEAKER_ID])[0]).strip()
        text = str((query.get("text") or [TTS_PREVIEW_TEXT])[0]).strip() or TTS_PREVIEW_TEXT
        text = text[:220]
        valid_speakers = {speaker["id"] for speaker in list_tts_speakers()}
        if not speaker_id or (valid_speakers and speaker_id not in valid_speakers):
            self.send_json({"error": f"Unknown EmotiVoice speaker id: {speaker_id}"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            audio_url = self.ensure_tts_preview(speaker_id, text)
            self.send_json({"audioUrl": audio_url, "text": text, "ttsSpeakerId": speaker_id})
        except Exception as exc:
            self.send_json({"error": f"Could not prepare voice preview: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def ensure_tts_preview(self, speaker_id, text):
        TTS_PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"{speaker_id}\n{text}".encode("utf-8")).hexdigest()[:16]
        base = TTS_PREVIEW_ROOT / f"speaker_{speaker_id}_{key}"
        input_json = base.with_suffix(".json")
        input_txt = base.with_suffix(".txt")
        output_wav = base.with_suffix(".wav")
        if output_wav.exists() and output_wav.stat().st_size > 1024:
            return self.output_url(output_wav)

        payload = {
            "reply_text": text,
            "batch_preview": {
                "response_emotion": "calm",
                "conversations": [
                    {
                        "chain_of_empathy": {
                            "goal_to_response": "Speak in a calm, warm, natural English voice."
                        }
                    }
                ],
            },
        }
        input_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.prepare_tts_preview_input(input_json, input_txt, speaker_id)
        self.synthesize_tts_preview(input_txt, output_wav)
        if not output_wav.exists() or output_wav.stat().st_size <= 1024:
            raise RuntimeError("preview audio was not created")
        return self.output_url(output_wav)

    def prepare_tts_preview_input(self, input_json, input_txt, speaker_id):
        root = AVATAR_ROOT / "integrations" / "emotivoice"
        py = root / ".EmotiVoice" / "bin" / "python"
        converter = AVATAR_ROOT / "integrations" / "avamerg" / "json_to_emotivoice_input.py"
        frontend = root / "frontend.py"
        container = AVATAR_ROOT / "runtime" / "containers" / "gaussianav_jammy"
        q = shlex.quote
        cmd = (
            f"cd {q(str(root))} && "
            f"export NLTK_DATA={q(str(AVATAR_ROOT / 'runtime' / 'cache' / 'nltk_data'))} && "
            f"{q(str(py))} {q(str(converter))} "
            f"--input_json {q(str(input_json))} "
            f"--output_txt {q(str(input_txt))} "
            f"--frontend_py {q(str(frontend))} "
            f"--speaker_id {q(str(speaker_id))} "
            f"--prompt_mode goal --wrap_sos_eos"
        )
        proc = subprocess.run(
            [
                "apptainer",
                "exec",
                "--nv",
                "-B",
                "/scratch:/scratch,/home/svu:/home/svu",
                str(container),
                "bash",
                "-lc",
                cmd,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stdout or "EmotiVoice prepare failed")[-1200:])

    def synthesize_tts_preview(self, input_txt, output_wav):
        url = WORKER_HEALTH_URLS["tts"].rstrip("/")
        with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if not health.get("ok"):
            raise RuntimeError(f"TTS worker unhealthy: {health}")
        body = json.dumps({"test_file": str(input_txt), "output_wav": str(output_wav)}).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or result))

    def handle_update_digital_human_speaker(self, avatar_id):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        speaker_id = str(data.get("ttsSpeakerId") or data.get("tts_speaker_id") or "").strip()
        valid_speakers = {speaker["id"] for speaker in list_tts_speakers()}
        if not speaker_id or (valid_speakers and speaker_id not in valid_speakers):
            self.send_json({"error": f"Unknown EmotiVoice speaker id: {speaker_id}"}, HTTPStatus.BAD_REQUEST)
            return

        humans = {str(item["id"]): item for item in list_digital_humans()}
        if avatar_id not in humans:
            self.send_json({"error": f"Unknown digital human avatar id: {avatar_id}"}, HTTPStatus.NOT_FOUND)
            return

        overrides = load_digital_human_overrides()
        current = humans[avatar_id]
        override = dict(overrides.get(avatar_id, {}))
        override["avatar_id"] = avatar_id
        override["tts_speaker_id"] = speaker_id
        override.setdefault("name", current["name"])
        override["role"] = re.sub(r"Voice\s+\S+", f"Voice {speaker_id}", str(override.get("role") or current["role"]))
        if "Voice " not in override["role"]:
            override["role"] = f"Avatar {avatar_id} · Voice {speaker_id}"
        override.setdefault("color", current["color"])
        overrides[avatar_id] = override
        save_digital_human_overrides(overrides)
        self.send_json({"digitalHumans": list_digital_humans()})

    def handle_update_digital_human_profile(self, avatar_id):
        if not self.require_admin():
            return
        data = self.read_json()
        if data is None:
            return
        humans = {str(item["id"]): item for item in list_digital_humans()}
        if avatar_id not in humans:
            self.send_json({"error": f"Unknown digital human avatar id: {avatar_id}"}, HTTPStatus.NOT_FOUND)
            return

        current = humans[avatar_id]
        speaker_id = str(data.get("ttsSpeakerId") or data.get("tts_speaker_id") or current["ttsSpeakerId"]).strip()
        valid_speakers = {speaker["id"] for speaker in list_tts_speakers()}
        if not speaker_id or (valid_speakers and speaker_id not in valid_speakers):
            self.send_json({"error": f"Unknown EmotiVoice speaker id: {speaker_id}"}, HTTPStatus.BAD_REQUEST)
            return

        name = str(data.get("name") or current["name"]).strip()
        role = str(data.get("role") or current["role"]).strip()
        if not name:
            self.send_json({"error": "Digital human name is required"}, HTTPStatus.BAD_REQUEST)
            return
        if not role:
            role = f"Avatar {avatar_id} · Voice {speaker_id}"

        overrides = load_digital_human_overrides()
        override = dict(overrides.get(avatar_id, {}))
        override["avatar_id"] = avatar_id
        override["tts_speaker_id"] = speaker_id
        override["name"] = name
        override["role"] = role
        override["color"] = str(data.get("color") or override.get("color") or current["color"])
        override["image_url"] = str(data.get("imageUrl") or data.get("image_url") or override.get("image_url") or current.get("imageUrl") or "")
        override["background_url"] = str(
            data.get("backgroundUrl")
            or data.get("background_url")
            or override.get("background_url")
            or current.get("backgroundUrl")
            or ""
        )
        overrides[avatar_id] = override
        save_digital_human_overrides(overrides)
        self.send_json({"digitalHumans": list_digital_humans()})

    def handle_upload_digital_human_image(self, avatar_id):
        if not self.require_admin():
            return
        humans = {str(item["id"]): item for item in list_digital_humans()}
        if avatar_id not in humans:
            self.send_json({"error": f"Unknown digital human avatar id: {avatar_id}"}, HTTPStatus.NOT_FOUND)
            return
        _fields, files = self.read_multipart()
        image_file = files.get("image") or files.get("file")
        if not image_file or not image_file.get("data"):
            self.send_json({"error": "Image file is required"}, HTTPStatus.BAD_REQUEST)
            return
        content_type = image_file.get("content_type") or ""
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }.get(content_type)
        if not extension:
            suffix = Path(image_file.get("filename") or "").suffix.lower()
            extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ""
        if not extension:
            self.send_json({"error": "Only jpg, png, webp, or gif images are supported"}, HTTPStatus.BAD_REQUEST)
            return
        if len(image_file["data"]) > 8 * 1024 * 1024:
            self.send_json({"error": "Image file cannot exceed 8MB"}, HTTPStatus.BAD_REQUEST)
            return

        DIGITAL_HUMAN_IMAGE_DIR.mkdir(exist_ok=True)
        filename = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', avatar_id)}_{uuid.uuid4().hex[:8]}{extension}"
        target = DIGITAL_HUMAN_IMAGE_DIR / filename
        target.write_bytes(image_file["data"])
        image_url = f"/digital_human_images/{filename}"

        overrides = load_digital_human_overrides()
        current = humans[avatar_id]
        override = dict(overrides.get(avatar_id, {}))
        override["avatar_id"] = avatar_id
        override.setdefault("tts_speaker_id", current["ttsSpeakerId"])
        override.setdefault("name", current["name"])
        override.setdefault("role", current["role"])
        override.setdefault("color", current["color"])
        override["image_url"] = image_url
        overrides[avatar_id] = override
        save_digital_human_overrides(overrides)
        self.send_json({"imageUrl": image_url, "digitalHumans": list_digital_humans()})

    def handle_upload_digital_human_background(self, avatar_id):
        if not self.require_admin():
            return
        humans = {str(item["id"]): item for item in list_digital_humans()}
        if avatar_id not in humans:
            self.send_json({"error": f"Unknown digital human avatar id: {avatar_id}"}, HTTPStatus.NOT_FOUND)
            return
        _fields, files = self.read_multipart()
        image_file = files.get("background") or files.get("image") or files.get("file")
        if not image_file or not image_file.get("data"):
            self.send_json({"error": "Background image file is required"}, HTTPStatus.BAD_REQUEST)
            return
        content_type = image_file.get("content_type") or ""
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(content_type)
        if not extension:
            suffix = Path(image_file.get("filename") or "").suffix.lower()
            extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ""
        if not extension:
            self.send_json({"error": "Only jpg, png, or webp background images are supported"}, HTTPStatus.BAD_REQUEST)
            return
        if len(image_file["data"]) > 12 * 1024 * 1024:
            self.send_json({"error": "Background image cannot exceed 12MB"}, HTTPStatus.BAD_REQUEST)
            return

        DIGITAL_HUMAN_BACKGROUND_DIR.mkdir(exist_ok=True)
        filename = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', avatar_id)}_{uuid.uuid4().hex[:8]}{extension}"
        target = DIGITAL_HUMAN_BACKGROUND_DIR / filename
        target.write_bytes(image_file["data"])
        background_url = f"/digital_human_backgrounds/{filename}"

        overrides = load_digital_human_overrides()
        current = humans[avatar_id]
        override = dict(overrides.get(avatar_id, {}))
        override["avatar_id"] = avatar_id
        override.setdefault("tts_speaker_id", current["ttsSpeakerId"])
        override.setdefault("name", current["name"])
        override.setdefault("role", current["role"])
        override.setdefault("color", current["color"])
        override.setdefault("image_url", current.get("imageUrl") or "")
        override["background_url"] = background_url
        overrides[avatar_id] = override
        save_digital_human_overrides(overrides)
        self.send_json({"backgroundUrl": background_url, "digitalHumans": list_digital_humans()})

    def handle_avatar_respond(self):
        user = self.current_user()
        if user and user["status"] != "active":
            self.send_json({"error": "Account disabled"}, HTTPStatus.FORBIDDEN)
            return
        is_guest = user is None
        if is_guest:
            user = {
                "id": str(self.headers.get("X-Guest-Id") or f"guest-{uuid.uuid4()}"),
                "username": clean_guest_name(self.headers.get("X-Guest-Name") or "Guest"),
                "role": "guest",
                "status": "active",
            }
        content_type = self.headers.get("Content-Type", "")
        files = {}
        if content_type.startswith("multipart/form-data"):
            data, files = self.read_multipart()
        else:
            data = self.read_json()
            if data is None:
                return

        digital_humans = list_digital_humans()
        human_by_id = {str(item["id"]): item for item in digital_humans}
        avatar_id = str(data.get("avatarId") or data.get("avatar_id") or PIPELINE_AVATAR_ID)
        if avatar_id not in human_by_id:
            self.send_json({"error": f"Unknown digital human avatar id: {avatar_id}"}, HTTPStatus.BAD_REQUEST)
            return
        tts_speaker_id = str(data.get("ttsSpeakerId") or data.get("tts_speaker_id") or human_by_id[avatar_id]["ttsSpeakerId"])
        text = str(data.get("text", "")).strip()
        if not text:
            self.send_json({"error": "Conversation text is required"}, HTTPStatus.BAD_REQUEST)
            return
        background = clean_booth_background(data.get("background") or data.get("backgroundId") or data.get("background_id"))

        avatar = avatar_meta(avatar_id)
        record_id = str(uuid.uuid4())
        created_at = now_iso()
        reply_text = ""
        video_url = ""
        audio_url = ""

        audio_file = files.get("audio")
        video_file = files.get("video")
        if audio_file or video_file:
            try:
                job = self.start_avatar_job(record_id, user, avatar_id, tts_speaker_id, text, audio_file, video_file, is_guest=is_guest, background=background)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(job, HTTPStatus.ACCEPTED)
            return
        else:
            summary = text[:36]
            reply_text = f"{avatar['reply']} You mentioned \"{summary}\"; record audio/video to generate a real avatar."

        if not is_guest:
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                      (id, user_id, avatar_id, user_text, reply_text, subtitle_text, user_video_url, video_url, audio_url, combined_video_url, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        user["id"],
                        avatar_id,
                        text,
                        reply_text,
                        english_subtitle_text(reply_text, reply_text),
                        "",
                        video_url,
                        audio_url,
                        "",
                        created_at,
                    ),
                )

        health = worker_health()
        self.send_json(
            {
                "id": record_id,
                "avatarId": avatar_id,
                "avatarName": avatar["name"],
                "replyText": reply_text,
                "videoUrl": video_url,
                "audioUrl": audio_url,
                "createdAt": created_at,
                "workerHealth": health,
                "workerHealthWarning": worker_health_summary(health),
            }
        )

    def start_avatar_job(self, record_id, user, avatar_id, tts_speaker_id, text, audio_file, video_file, is_guest=False, background="study"):
        if not AVATAR_SCRIPT.exists():
            raise RuntimeError(f"Avatar pipeline script not found: {AVATAR_SCRIPT}")

        upload_dir = AVATAR_UPLOAD_ROOT / record_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        if is_guest:
            mark_guest_artifact(upload_dir, record_id, user.get("id", ""))

        input_video = None
        input_video_valid = False
        input_video_warning = ""
        if video_file and video_file.get("data"):
            suffix = Path(video_file.get("filename") or "input.webm").suffix.lower() or ".webm"
            if suffix not in {".webm", ".mp4", ".mov", ".m4v"}:
                suffix = ".webm"
            input_video = upload_dir / f"input_video{suffix}"
            input_video.write_bytes(video_file["data"])
            input_video_valid, input_video_warning = self.validate_input_video(input_video)

        input_wav = upload_dir / "input.wav"
        if audio_file and audio_file.get("data"):
            input_wav.write_bytes(audio_file["data"])
        elif input_video and input_video_valid:
            self.extract_audio_from_video(input_video, input_wav)
        else:
            raise RuntimeError("Record video or audio before sending to the avatar system.")
        if input_wav.stat().st_size == 0:
            raise RuntimeError("Captured microphone audio is empty. Turn Mic on and record again.")

        run_id = f"3depb_{record_id.replace('-', '')[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = AVATAR_OUTPUT_ROOT / run_id
        if is_guest:
            mark_guest_artifact(run_dir, record_id, user.get("id", ""))
        web_log = upload_dir / "pipeline.log"
        command = [
            "bash",
            str(AVATAR_SCRIPT),
            "agent",
            str(input_wav),
            str(avatar_id),
            "--run_id",
            run_id,
            "--tts_speaker_id",
            str(tts_speaker_id),
            "--background",
            clean_booth_background(background),
        ]
        if input_video and input_video_valid:
            command.extend(["--input_video", str(input_video)])
        if PIPELINE_NO_LLM:
            command.append("--no_llm")
        health = worker_health()

        env = os.environ.copy()
        env.setdefault("HF_HOME", str(AVATAR_ROOT / "runtime" / "cache" / "hf"))
        env.setdefault("XDG_CACHE_HOME", str(AVATAR_ROOT / "runtime" / "cache" / "xdg"))
        env.setdefault("MODELSCOPE_CACHE", str(AVATAR_ROOT / "runtime" / "cache" / "modelscope"))
        env.setdefault("NLTK_DATA", str(AVATAR_ROOT / "runtime" / "cache" / "nltk_data"))

        job = {
            "jobId": record_id,
            "runId": run_id,
            "status": "queued",
            "stage": "queued",
            "stageLabel": "Queued",
            "progress": 2,
            "avatarId": avatar_id,
            "ttsSpeakerId": tts_speaker_id,
            "background": clean_booth_background(background),
            "userId": user["id"],
            "username": user["username"],
            "isGuest": bool(is_guest),
            "text": text,
            "createdAt": now_iso(),
            "runDir": str(run_dir),
            "logPath": str(web_log),
            "replyText": "",
            "inputVideoUrl": self.output_url(input_video),
            "inputVideoPath": str(input_video) if input_video else "",
            "inputVideoValid": bool(input_video_valid),
            "inputVideoWarning": input_video_warning,
            "workerHealth": health,
            "workerHealthWarning": worker_health_summary(health),
            "videoUrl": "",
            "audioUrl": "",
            "combinedVideoUrl": "",
            "error": "",
        }
        with PIPELINE_JOBS_LOCK:
            PIPELINE_JOBS[record_id] = job

        thread = threading.Thread(
            target=self.run_avatar_job,
            args=(record_id, command, env, run_dir, web_log),
            daemon=True,
        )
        thread.start()
        return self.avatar_job_payload(record_id)

    def run_avatar_job(self, job_id, command, env, run_dir, web_log):
        with PIPELINE_JOBS_LOCK:
            if job_id in PIPELINE_JOBS:
                PIPELINE_JOBS[job_id]["status"] = "running"
                PIPELINE_JOBS[job_id]["stage"] = "starting"
                PIPELINE_JOBS[job_id]["stageLabel"] = "Starting avatar system"
                PIPELINE_JOBS[job_id]["progress"] = 5
        with web_log.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n\n")
            proc = subprocess.Popen(
                command,
                cwd=str(AVATAR_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with PIPELINE_JOBS_LOCK:
                if job_id in PIPELINE_JOBS:
                    PIPELINE_JOBS[job_id]["processPid"] = proc.pid
            try:
                return_code = proc.wait(timeout=PIPELINE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                return_code = -9

        with PIPELINE_JOBS_LOCK:
            if PIPELINE_JOBS.get(job_id, {}).get("status") == "cancelled":
                return
        if return_code != 0:
            tail = web_log.read_text(encoding="utf-8", errors="replace")[-2000:]
            with PIPELINE_JOBS_LOCK:
                if job_id in PIPELINE_JOBS:
                    PIPELINE_JOBS[job_id]["status"] = "failed"
                    PIPELINE_JOBS[job_id]["error"] = f"Avatar pipeline failed. Log tail:\n{tail}"
            return

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            with PIPELINE_JOBS_LOCK:
                if job_id in PIPELINE_JOBS:
                    PIPELINE_JOBS[job_id]["status"] = "failed"
                    PIPELINE_JOBS[job_id]["error"] = f"Avatar pipeline finished but manifest is missing: {manifest_path}"
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("error"):
            with PIPELINE_JOBS_LOCK:
                if job_id in PIPELINE_JOBS:
                    PIPELINE_JOBS[job_id]["status"] = "failed"
                    PIPELINE_JOBS[job_id]["error"] = str(manifest["error"])
            return

        output_video = manifest.get("output_video")
        reply_wav = manifest.get("artifact_enhanced_reply_wav") or manifest.get("artifact_reply_wav") or manifest.get("reply_wav")
        subtitle_text = english_subtitle_text(
            manifest.get("reply_text") or "",
            manifest.get("subtitle_text") or manifest.get("tts_text") or read_task1_tts_text(manifest.get("task1_reply_json")),
        )
        combined_video_url = ""
        with PIPELINE_JOBS_LOCK:
            job = PIPELINE_JOBS.get(job_id)
            if not job:
                return
            if job.get("inputVideoPath") and job.get("inputVideoValid") and output_video:
                combined_video_url = self.combine_turn_videos(
                    Path(job["inputVideoPath"]),
                    Path(output_video),
                    AVATAR_UPLOAD_ROOT / job_id / "combined_turn.mp4",
                )
            job["status"] = "done"
            job["stage"] = "done"
            job["stageLabel"] = "Avatar ready"
            job["progress"] = 100
            job["replyText"] = manifest.get("reply_text") or ""
            job["subtitleText"] = subtitle_text
            job["videoUrl"] = self.output_url(output_video)
            job["audioUrl"] = self.output_url(reply_wav)
            job["combinedVideoUrl"] = combined_video_url
            avatar = avatar_meta(job["avatarId"])
            if not job.get("isGuest"):
                with connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO conversations
                          (id, user_id, avatar_id, user_text, reply_text, subtitle_text, user_video_url, video_url, audio_url, combined_video_url, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            job["userId"],
                            job["avatarId"],
                            job["text"],
                            job["replyText"] or "",
                            job.get("subtitleText") or english_subtitle_text(job.get("replyText") or "", job.get("replyText") or ""),
                            job.get("inputVideoUrl") or "",
                            job["videoUrl"],
                            job["audioUrl"],
                            job.get("combinedVideoUrl") or "",
                            now_iso(),
                        ),
                    )

    def combine_turn_videos(self, user_video, avatar_video, output):
        if not FFMPEG_BIN.exists():
            return ""
        if not user_video.exists() or not avatar_video.exists():
            return ""
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(FFMPEG_BIN),
            "-y",
            "-i",
            str(user_video),
            "-i",
            str(avatar_video),
            "-filter_complex",
            (
                "[0:v]scale=568:430:force_original_aspect_ratio=decrease,"
                "pad=568:430:(ow-iw)/2:(oh-ih)/2:color=0b1117,setsar=1[left];"
                "[1:v]scale=568:430:force_original_aspect_ratio=decrease,"
                "pad=568:430:(ow-iw)/2:(oh-ih)/2:color=0b1117,setsar=1[right];"
                "color=c=0d1117:s=1280x720:r=25[base];"
                "[base]drawbox=x=36:y=92:w=592:h=500:color=0x15202b@0.95:t=fill,"
                "drawbox=x=652:y=92:w=592:h=500:color=0x15202b@0.95:t=fill[bg];"
                "[bg][left]overlay=48:132[tmp1];"
                "[tmp1][right]overlay=664:132[tmp2];"
                "[tmp2]drawbox=x=36:y=92:w=592:h=500:color=0x32d0a4@0.75:t=4,"
                "drawbox=x=652:y=92:w=592:h=500:color=0x6fb7ff@0.75:t=4,"
                "drawbox=x=48:y=132:w=568:h=430:color=0xffffff@0.08:t=2,"
                "drawbox=x=664:y=132:w=568:h=430:color=0xffffff@0.08:t=2,"
                "drawtext=text='You':x=58:y=108:fontsize=26:fontcolor=white:box=1:boxcolor=0x0d1117@0.65:boxborderw=10,"
                "drawtext=text='Avatar':x=674:y=108:fontsize=26:fontcolor=white:box=1:boxcolor=0x0d1117@0.65:boxborderw=10,"
                "format=yuv420p[v]"
            ),
            "-map",
            "[v]",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-r",
            "25",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not output.exists():
            (output.parent / "combined_turn.log").write_text(proc.stdout or "", encoding="utf-8")
            return ""
        return self.output_url(output)

    def validate_input_video(self, input_video):
        if not input_video or not Path(input_video).exists():
            return False, "No captured video file was saved."
        if Path(input_video).stat().st_size < 2048:
            return False, "Captured video is too small to decode; continuing with audio only."
        ffprobe = FFMPEG_BIN.with_name("ffprobe") if isinstance(FFMPEG_BIN, Path) else Path(str(FFMPEG_BIN)).with_name("ffprobe")
        ffprobe_cmd = str(ffprobe) if ffprobe.exists() else "ffprobe"
        command = [
            ffprobe_cmd,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(input_video),
        ]
        try:
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        except Exception as exc:
            return False, f"Could not validate captured video; continuing with audio only: {exc}"
        if proc.returncode != 0 or "video" not in (proc.stdout or ""):
            return False, f"Captured video could not be decoded; continuing with audio only: {(proc.stdout or '').strip()[-500:]}"
        return True, ""

    def handle_avatar_job(self, job_id):
        user = self.current_user()
        payload = self.avatar_job_payload(job_id)
        if not payload:
            self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return
        if user and user["status"] != "active":
            self.send_json({"error": "Account disabled"}, HTTPStatus.FORBIDDEN)
            return
        if user and payload.get("userId") != user["id"] and user["role"] != "admin":
            self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            return
        if not user and not payload.get("isGuest"):
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        self.send_json(payload)

    def handle_cancel_avatar_job(self, job_id):
        user = self.current_user()
        payload = self.avatar_job_payload(job_id)
        if not payload:
            self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return
        if user and user["status"] != "active":
            self.send_json({"error": "Account disabled"}, HTTPStatus.FORBIDDEN)
            return
        if user and payload.get("userId") != user["id"] and user["role"] != "admin":
            self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            return
        if not user and not payload.get("isGuest"):
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        pid = int(payload.get("processPid") or 0)
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        with PIPELINE_JOBS_LOCK:
            if job_id in PIPELINE_JOBS:
                PIPELINE_JOBS[job_id]["status"] = "cancelled"
                PIPELINE_JOBS[job_id]["stage"] = "cancelled"
                PIPELINE_JOBS[job_id]["stageLabel"] = "Stopped"
                PIPELINE_JOBS[job_id]["progress"] = 0
                PIPELINE_JOBS[job_id]["error"] = ""
        self.send_json({"ok": True, "status": "cancelled"})

    def avatar_job_payload(self, job_id):
        with PIPELINE_JOBS_LOCK:
            job = PIPELINE_JOBS.get(job_id)
            if not job:
                return None
            payload = dict(job)
        run_dir = Path(payload.get("runDir") or "")
        state = {}
        manifest = {}
        if run_dir.exists():
            state_path = run_dir / "state.json"
            manifest_path = run_dir / "manifest.json"
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {}
        if payload["status"] == "running":
            current = state.get("current_stage") or payload.get("stage") or "starting"
            finished = state.get("finished_stages") or []
            failed = state.get("failed_stage")
            if failed or state.get("error"):
                payload["status"] = "failed"
                payload["error"] = state.get("error") or f"Failed at {failed}"
            else:
                payload["stage"] = current
                payload["stageLabel"] = PIPELINE_STAGE_LABELS.get(current, current.replace("_", " ").title())
                total = max(1, len(PIPELINE_STAGES))
                index = max(len(finished), 0)
                if current in PIPELINE_STAGE_LABELS:
                    index = max(index, [name for name, _ in PIPELINE_STAGES].index(current))
                next_progress = min(96, max(5, round((index / total) * 100)))
                payload["progress"] = max(int(payload.get("progress") or 0), next_progress)
                with PIPELINE_JOBS_LOCK:
                    if job_id in PIPELINE_JOBS:
                        PIPELINE_JOBS[job_id]["stage"] = payload["stage"]
                        PIPELINE_JOBS[job_id]["stageLabel"] = payload["stageLabel"]
                        PIPELINE_JOBS[job_id]["progress"] = payload["progress"]
        if state and payload["status"] != "failed":
            dialogue_extra = (state.get("extra") or {}).get("dialogue_agent") or {}
            partial_reply = state.get("reply_text") or dialogue_extra.get("reply_text")
            partial_subtitle = english_subtitle_text(
                partial_reply or "",
                state.get("subtitle_text") or state.get("tts_text") or read_task1_tts_text(state.get("task1_reply_json")),
            )
            partial_wav = state.get("artifact_enhanced_reply_wav") or state.get("artifact_reply_wav") or state.get("reply_wav")
            if partial_reply and not payload.get("replyText"):
                payload["replyText"] = partial_reply
            if partial_subtitle and not payload.get("subtitleText"):
                payload["subtitleText"] = partial_subtitle
            if partial_wav and not payload.get("audioUrl"):
                payload["audioUrl"] = self.output_url(partial_wav) or payload.get("audioUrl") or ""
            with PIPELINE_JOBS_LOCK:
                if job_id in PIPELINE_JOBS:
                    if payload.get("replyText"):
                        PIPELINE_JOBS[job_id]["replyText"] = payload["replyText"]
                    if payload.get("subtitleText"):
                        PIPELINE_JOBS[job_id]["subtitleText"] = payload["subtitleText"]
                    if payload.get("audioUrl"):
                        PIPELINE_JOBS[job_id]["audioUrl"] = payload["audioUrl"]
        if manifest and payload["status"] != "failed":
            output_video = manifest.get("output_video")
            reply_wav = manifest.get("artifact_enhanced_reply_wav") or manifest.get("artifact_reply_wav") or manifest.get("reply_wav")
            subtitle_text = english_subtitle_text(
                manifest.get("reply_text") or payload.get("replyText") or "",
                manifest.get("subtitle_text") or manifest.get("tts_text") or read_task1_tts_text(manifest.get("task1_reply_json")),
            )
            payload["replyText"] = manifest.get("reply_text") or payload.get("replyText") or ""
            payload["subtitleText"] = subtitle_text or english_subtitle_text(payload.get("replyText") or "", payload.get("subtitleText") or "")
            payload["videoUrl"] = self.output_url(output_video) or payload.get("videoUrl") or ""
            payload["audioUrl"] = self.output_url(reply_wav) or payload.get("audioUrl") or ""
            if not manifest.get("error") and output_video:
                payload["status"] = "done"
                payload["stage"] = "done"
                payload["stageLabel"] = "Avatar ready"
                payload["progress"] = 100
        if state:
            input_extra = (state.get("extra") or {}).get("input_agent") or {}
            if input_extra.get("video_warning") and not payload.get("inputVideoWarning"):
                payload["inputVideoWarning"] = input_extra.get("video_warning")
            if "input_video_valid" in input_extra:
                payload["inputVideoValid"] = bool(input_extra.get("input_video_valid"))
        payload["workerHealth"] = worker_health()
        payload["workerHealthWarning"] = worker_health_summary(payload["workerHealth"])
        return payload

    def load_run_json(self, run_id, name):
        if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
            return {}
        path = AVATAR_OUTPUT_ROOT / run_id / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def handle_viewer_assets(self, run_id):
        manifest = self.load_run_json(run_id, "manifest.json")
        state = self.load_run_json(run_id, "state.json")
        if not manifest and not state:
            self.send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return
        point_cloud_path = state.get("point_cloud_path") or manifest.get("point_cloud_path")
        motion_path = manifest.get("artifact_flame_motion_npz") or state.get("artifact_flame_motion_npz") or state.get("flame_motion_npz")
        audio_path = manifest.get("artifact_enhanced_reply_wav") or manifest.get("artifact_reply_wav") or manifest.get("reply_wav")
        has_point_cloud = bool(point_cloud_path and Path(point_cloud_path).exists())
        frame_count = 0
        if motion_path and Path(motion_path).exists():
            frame_count = int(manifest.get("frame_count") or state.get("frame_count") or 0)
        self.send_json(
            {
                "runId": run_id,
                "run_id": run_id,
                "fps": 25,
                "frame_count": frame_count,
                "pointCloudUrl": f"/api/avatar/runs/{run_id}/point_cloud" if has_point_cloud else "",
                "point_cloud_url": f"/api/jobs/{run_id}/viewer/point_cloud" if has_point_cloud else None,
                "motionUrl": self.output_url(motion_path),
                "motion_url": self.output_url(motion_path),
                "audioUrl": self.output_url(audio_path),
                "audio_url": self.output_url(audio_path),
                "point_cloud_path": point_cloud_path if has_point_cloud else None,
                "flame_motion_path": motion_path if motion_path and Path(motion_path).exists() else None,
                "replyText": manifest.get("reply_text") or "",
            }
        )

    def handle_point_cloud(self, run_id):
        manifest = self.load_run_json(run_id, "manifest.json")
        state = self.load_run_json(run_id, "state.json")
        point_cloud_path = state.get("point_cloud_path") or manifest.get("point_cloud_path")
        if not point_cloud_path:
            self.send_json({"error": "Point cloud not found"}, HTTPStatus.NOT_FOUND)
            return
        path = Path(point_cloud_path).resolve()
        try:
            path.relative_to(AVATAR_ROOT.resolve())
        except ValueError:
            self.send_json({"error": "Point cloud path is outside project root"}, HTTPStatus.FORBIDDEN)
            return
        if not path.exists() or path.suffix.lower() != ".ply":
            self.send_json({"error": "Point cloud not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_render_frame(self, run_id):
        if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
            self.send_json({"error": "Invalid run id"}, HTTPStatus.BAD_REQUEST)
            return
        manifest = self.load_run_json(run_id, "manifest.json")
        state = self.load_run_json(run_id, "state.json")
        if not manifest and not state:
            self.send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return
        data = self.read_json()
        if data is None:
            return

        point_cloud_path = state.get("point_cloud_path") or manifest.get("point_cloud_path")
        motion_path = manifest.get("artifact_flame_motion_npz") or state.get("artifact_flame_motion_npz") or state.get("flame_motion_npz")
        for label, value in (("point cloud", point_cloud_path), ("motion npz", motion_path)):
            if not value or not Path(value).exists():
                self.send_json({"error": f"Missing {label} for render preview"}, HTTPStatus.BAD_REQUEST)
                return

        worker_url = os.environ.get("GAUSSIAN_RENDER_WORKER_URL", "http://127.0.0.1:8792").rstrip("/")
        try:
            with urllib.request.urlopen(f"{worker_url}/health", timeout=2) as resp:
                health = json.loads(resp.read().decode("utf-8"))
            if not health.get("ok"):
                raise RuntimeError(f"unhealthy response: {health}")
        except Exception as exc:
            self.send_json({"error": f"Gaussian render worker unavailable: {exc}"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return

        reason = str(data.get("reason") or "idle")
        quality = 52 if reason == "playback" else 62 if reason == "drag" else 84
        payload = json.dumps(
            {
                "point_path": point_cloud_path,
                "motion_path": motion_path,
                "camera": data.get("camera") or {},
                "frame": max(0, int(data.get("frame") or 0)),
                "width": max(1, int(data.get("width") or 640)),
                "height": max(1, int(data.get("height") or 640)),
                "image_format": "jpeg",
                "quality": quality,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{worker_url}/render_frame_bytes",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                image_bytes = resp.read()
                content_type = resp.headers.get_content_type() or "image/jpeg"
                frame_header = resp.headers.get("X-Frame", str(data.get("frame") or 0))
                frame_count_header = resp.headers.get("X-Frame-Count", "")
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            self.send_json({"error": f"Gaussian render worker failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame", str(frame_header))
        if frame_count_header:
            self.send_header("X-Frame-Count", str(frame_count_header))
        self.end_headers()
        self.wfile.write(image_bytes)

    def extract_audio_from_video(self, input_video, input_wav):
        if not FFMPEG_BIN.exists():
            raise RuntimeError("No audio wav was captured, and ffmpeg is unavailable for extracting audio from video.")
        command = [
            str(FFMPEG_BIN),
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(input_wav),
        ]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not input_wav.exists():
            raise RuntimeError(f"Could not extract microphone audio from the recorded video:\n{proc.stdout[-1200:]}")

    def output_url(self, path_value):
        if not path_value:
            return ""
        path = Path(path_value)
        try:
            rel = path.resolve().relative_to(AVATAR_OUTPUT_ROOT.resolve())
        except ValueError:
            return ""
        return "/outputs/" + str(rel).replace(os.sep, "/")

    def handle_list_conversations(self):
        user = self.require_user()
        if not user:
            return
        query = parse_qs(urlparse(self.path).query)
        avatar_id = str((query.get("avatarId") or query.get("avatar_id") or [""])[0]).strip()
        with connect() as conn:
            if user["role"] == "admin":
                if avatar_id:
                    rows = conn.execute(
                        """
                        SELECT conversations.*, users.username
                        FROM conversations
                        JOIN users ON users.id = conversations.user_id
                        WHERE conversations.avatar_id = ?
                        ORDER BY conversations.created_at DESC
                        LIMIT 80
                        """,
                        (avatar_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT conversations.*, users.username
                        FROM conversations
                        JOIN users ON users.id = conversations.user_id
                        ORDER BY conversations.created_at DESC
                        LIMIT 80
                        """
                    ).fetchall()
            else:
                if avatar_id:
                    rows = conn.execute(
                        """
                        SELECT conversations.*, users.username
                        FROM conversations
                        JOIN users ON users.id = conversations.user_id
                        WHERE conversations.user_id = ? AND conversations.avatar_id = ?
                        ORDER BY conversations.created_at DESC
                        LIMIT 80
                        """,
                        (user["id"], avatar_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT conversations.*, users.username
                        FROM conversations
                        JOIN users ON users.id = conversations.user_id
                        WHERE conversations.user_id = ?
                        ORDER BY conversations.created_at DESC
                        LIMIT 80
                        """,
                        (user["id"],),
                    ).fetchall()
        self.send_json({"conversations": [conversation_payload(row) for row in rows]})

    def handle_export_history(self):
        user = self.require_user()
        if not user:
            return
        if not FFMPEG_BIN.exists():
            self.send_json({"error": "ffmpeg is unavailable, so history videos cannot be exported."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        data = self.read_json()
        if data is None:
            return
        entries = data.get("entries")
        urls = []
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    url = str(item.get("combinedVideoUrl") or item.get("combined_video_url") or "").strip()
                    if url:
                        urls.append(url)
        if not urls:
            query_avatar_id = str(data.get("avatarId") or data.get("avatar_id") or "").strip()
            with connect() as conn:
                if query_avatar_id:
                    rows = conn.execute(
                        "SELECT combined_video_url FROM conversations WHERE user_id = ? AND avatar_id = ? ORDER BY created_at ASC",
                        (user["id"], query_avatar_id),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT combined_video_url FROM conversations WHERE user_id = ? ORDER BY created_at ASC",
                        (user["id"],),
                    ).fetchall()
            urls = [row["combined_video_url"] for row in rows if row["combined_video_url"]]
        paths = []
        for url in urls:
            path = self.public_media_path(url)
            if path and path.exists() and path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
                paths.append(path)
        if not paths:
            self.send_json({"error": "No combined history videos are ready to export."}, HTTPStatus.BAD_REQUEST)
            return
        export_id = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        export_dir = EXPORTS_DIR / export_id
        export_dir.mkdir(parents=True, exist_ok=True)
        normalized = []
        logs = []
        for index, path in enumerate(paths):
            out = export_dir / f"segment_{index:03d}.mp4"
            command = [
                str(FFMPEG_BIN),
                "-y",
                "-i",
                str(path),
                "-vf",
                "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-r",
                "25",
                "-c:a",
                "aac",
                str(out),
            ]
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logs.append("$ " + " ".join(command) + "\n" + (proc.stdout or ""))
            if proc.returncode == 0 and out.exists():
                normalized.append(out)
        list_path = export_dir / "inputs.txt"
        output = export_dir / "history_export.mp4"
        with list_path.open("w", encoding="utf-8") as f:
            for path in normalized:
                f.write(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")
        command = [str(FFMPEG_BIN), "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs.append("$ " + " ".join(command) + "\n" + (proc.stdout or ""))
        (export_dir / "export.log").write_text("\n\n".join(logs), encoding="utf-8")
        if proc.returncode != 0 or not output.exists():
            self.send_json({"error": "History export failed. Check export.log."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json({"ok": True, "url": f"/exports/{export_id}/history_export.mp4"})

    def handle_create_recording(self):
        user = self.require_user()
        if not user:
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            self.handle_create_recording_multipart(user)
            return

        data = self.read_json()
        if data is None:
            return
        avatar_id = str(data.get("avatarId", "companion"))
        mime_type = str(data.get("mimeType", "video/webm"))
        content = str(data.get("data", ""))
        if not content:
            self.send_json({"error": "Recording content is required"}, HTTPStatus.BAD_REQUEST)
            return
        if not mime_type.startswith("video/"):
            self.send_json({"error": "Only video recording files are supported"}, HTTPStatus.BAD_REQUEST)
            return

        if "," in content:
            content = content.split(",", 1)[1]
        content = re.sub(r"\s+", "", content)
        padding = len(content) % 4
        if padding:
            content += "=" * (4 - padding)

        try:
            video_bytes = b64decode(content, validate=False)
        except Exception:
            self.send_json({"error": "Recording file encoding is invalid"}, HTTPStatus.BAD_REQUEST)
            return
        if len(video_bytes) == 0:
            self.send_json({"error": "Recording file is empty"}, HTTPStatus.BAD_REQUEST)
            return
        if len(video_bytes) > 80 * 1024 * 1024:
            self.send_json({"error": "Recording file cannot exceed 80MB"}, HTTPStatus.BAD_REQUEST)
            return

        record_id = str(uuid.uuid4())
        extension = ".webm" if "webm" in mime_type else ".mp4"
        file_name = f"{record_id}{extension}"
        url = f"/recordings/{file_name}"
        target = RECORDINGS_DIR / file_name
        target.write_bytes(video_bytes)
        created_at = now_iso()

        with connect() as conn:
            conn.execute(
                """
                INSERT INTO recordings
                  (id, user_id, avatar_id, file_name, url, mime_type, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, user["id"], avatar_id, file_name, url, mime_type, len(video_bytes), created_at),
            )
            row = conn.execute(
                """
                SELECT recordings.*, users.username
                FROM recordings
                JOIN users ON users.id = recordings.user_id
                WHERE recordings.id = ?
                """,
                (record_id,),
            ).fetchone()
        self.send_json({"recording": recording_payload(row)})

    def handle_create_recording_multipart(self, user):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")
        message = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )

        fields = {}
        file_bytes = None
        file_content_type = "video/webm"
        if message.is_multipart():
            for part in message.iter_parts():
                disposition = part.get_content_disposition()
                field_name = part.get_param("name", header="content-disposition")
                if disposition != "form-data" or not field_name:
                    continue
                payload = part.get_payload(decode=True) or b""
                if field_name == "file":
                    file_bytes = payload
                    file_content_type = part.get_content_type()
                else:
                    fields[field_name] = payload.decode(part.get_content_charset() or "utf-8")

        if file_bytes is None:
            self.send_json({"error": "Recording file is required"}, HTTPStatus.BAD_REQUEST)
            return

        avatar_id = fields.get("avatarId", "companion")
        mime_type = fields.get("mimeType", file_content_type)
        if not mime_type.startswith("video/"):
            self.send_json({"error": "Only video recording files are supported"}, HTTPStatus.BAD_REQUEST)
            return

        self.save_recording_bytes(user, avatar_id, mime_type, file_bytes)

    def save_recording_bytes(self, user, avatar_id, mime_type, video_bytes):
        if len(video_bytes) == 0:
            self.send_json({"error": "Recording file is empty"}, HTTPStatus.BAD_REQUEST)
            return
        if len(video_bytes) > 80 * 1024 * 1024:
            self.send_json({"error": "Recording file cannot exceed 80MB"}, HTTPStatus.BAD_REQUEST)
            return

        record_id = str(uuid.uuid4())
        extension = ".webm" if "webm" in mime_type else ".mp4"
        file_name = f"{record_id}{extension}"
        url = f"/recordings/{file_name}"
        target = RECORDINGS_DIR / file_name
        target.write_bytes(video_bytes)
        created_at = now_iso()

        with connect() as conn:
            conn.execute(
                """
                INSERT INTO recordings
                  (id, user_id, avatar_id, file_name, url, mime_type, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, user["id"], avatar_id, file_name, url, mime_type, len(video_bytes), created_at),
            )
            row = conn.execute(
                """
                SELECT recordings.*, users.username
                FROM recordings
                JOIN users ON users.id = recordings.user_id
                WHERE recordings.id = ?
                """,
                (record_id,),
            ).fetchone()
        self.send_json({"recording": recording_payload(row)})

    def handle_list_recordings(self):
        user = self.require_user()
        if not user:
            return
        with connect() as conn:
            if user["role"] == "admin":
                rows = conn.execute(
                    """
                    SELECT recordings.*, users.username
                    FROM recordings
                    JOIN users ON users.id = recordings.user_id
                    ORDER BY recordings.created_at DESC
                    LIMIT 1
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT recordings.*, users.username
                    FROM recordings
                    JOIN users ON users.id = recordings.user_id
                    WHERE recordings.user_id = ?
                    ORDER BY recordings.created_at DESC
                    LIMIT 1
                    """,
                    (user["id"],),
                ).fetchall()
        self.send_json({"recordings": [recording_payload(row) for row in rows]})

    def handle_clear_history(self):
        user = self.require_user()
        if not user:
            return
        query = parse_qs(urlparse(self.path).query)
        avatar_id = str((query.get("avatarId") or query.get("avatar_id") or [""])[0]).strip()
        media_urls = set()
        with connect() as conn:
            if avatar_id:
                conversation_rows = conn.execute(
                    """
                    SELECT user_video_url, video_url, audio_url
                    FROM conversations
                    WHERE user_id = ? AND avatar_id = ?
                    """,
                    (user["id"], avatar_id),
                ).fetchall()
                recording_rows = conn.execute(
                    """
                    SELECT url
                    FROM recordings
                    WHERE user_id = ? AND avatar_id = ?
                    """,
                    (user["id"], avatar_id),
                ).fetchall()
            else:
                conversation_rows = conn.execute(
                    """
                    SELECT user_video_url, video_url, audio_url
                    FROM conversations
                    WHERE user_id = ?
                    """,
                    (user["id"],),
                ).fetchall()
                recording_rows = conn.execute(
                    """
                    SELECT url
                    FROM recordings
                    WHERE user_id = ?
                    """,
                    (user["id"],),
                ).fetchall()
            for row in conversation_rows:
                media_urls.update(url for url in (row["user_video_url"], row["video_url"], row["audio_url"]) if url)
            for row in recording_rows:
                if row["url"]:
                    media_urls.add(row["url"])
            if avatar_id:
                conn.execute("DELETE FROM conversations WHERE user_id = ? AND avatar_id = ?", (user["id"], avatar_id))
                conn.execute("DELETE FROM recordings WHERE user_id = ? AND avatar_id = ?", (user["id"], avatar_id))
            else:
                conn.execute("DELETE FROM conversations WHERE user_id = ?", (user["id"],))
                conn.execute("DELETE FROM recordings WHERE user_id = ?", (user["id"],))

        deleted_files = 0
        for url in media_urls:
            if self.delete_public_media(url):
                deleted_files += 1
        self.send_json({"ok": True, "deletedFiles": deleted_files})

    def delete_public_media(self, url):
        target = self.public_media_path(url)
        if not target or not target.exists() or not target.is_file():
            return False
        try:
            target.unlink()
            return True
        except OSError:
            return False

    def public_media_path(self, url):
        if url.startswith("/outputs/"):
            root = AVATAR_OUTPUT_ROOT.resolve()
            target = (AVATAR_OUTPUT_ROOT / url.removeprefix("/outputs/")).resolve()
        elif url.startswith("/recordings/"):
            root = RECORDINGS_DIR.resolve()
            target = (RECORDINGS_DIR / url.removeprefix("/recordings/")).resolve()
        elif url.startswith("/exports/"):
            root = EXPORTS_DIR.resolve()
            target = (EXPORTS_DIR / url.removeprefix("/exports/")).resolve()
        else:
            return None
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return target

    def serve_static(self, path, head_only=False):
        if path.startswith("/api/"):
            self.send_json({"error": "API endpoint not found. Restart the Booth server if this endpoint was just added."}, HTTPStatus.NOT_FOUND)
            return
        if path == "/":
            path = "/index.html"
        if path.startswith("/vendor/"):
            target = (ROOT / "vendor" / path.removeprefix("/vendor/")).resolve()
            static_root = (ROOT / "vendor").resolve()
        elif path.startswith("/outputs/"):
            target = (AVATAR_OUTPUT_ROOT / path.removeprefix("/outputs/")).resolve()
            static_root = AVATAR_OUTPUT_ROOT.resolve()
        elif path.startswith("/exports/"):
            target = (EXPORTS_DIR / path.removeprefix("/exports/")).resolve()
            static_root = EXPORTS_DIR.resolve()
        elif path.startswith("/digital_human_images/"):
            target = (DIGITAL_HUMAN_IMAGE_DIR / path.removeprefix("/digital_human_images/")).resolve()
            static_root = DIGITAL_HUMAN_IMAGE_DIR.resolve()
        elif path.startswith("/digital_human_backgrounds/"):
            target = (DIGITAL_HUMAN_BACKGROUND_DIR / path.removeprefix("/digital_human_backgrounds/")).resolve()
            static_root = DIGITAL_HUMAN_BACKGROUND_DIR.resolve()
        else:
            target = (ROOT / path.lstrip("/")).resolve()
            static_root = ROOT.resolve()
        if not str(target).startswith(str(static_root)) or not target.exists() or target.is_dir():
            target = ROOT / "index.html"

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if path.startswith("/outputs/tts_previews/") or path.startswith("/digital_human_images/") or path.startswith("/digital_human_backgrounds/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def main():
    init_db()
    atexit.register(cleanup_guest_artifacts)

    def shutdown_handler(signum, _frame):
        cleanup_guest_artifacts()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    port = int(os.environ.get("PORT", "4173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"3DEPB server running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
