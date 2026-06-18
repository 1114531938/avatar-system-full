#!/usr/bin/env python3
import json
import mimetypes
import os
import secrets
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
AVATAR_SCRIPT = AVATAR_ROOT / "scripts" / "avatar.sh"
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "app.db"
RECORDINGS_DIR = ROOT / "recordings"
DIGITAL_HUMAN_IMAGE_DIR = ROOT / "digital_human_images"
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
    ("plan_agent", "Selecting avatar and voice"),
    ("emotivoice_prepare", "Preparing TTS text"),
    ("render_agent", "Preparing avatar render"),
    ("emotivoice_tts", "Synthesizing companion voice"),
    ("deeptalk", "Driving facial motion"),
    ("flame_merge", "Merging avatar motion"),
    ("viewer", "Preparing viewer assets"),
    ("artifact_export", "Rendering digital human video"),
]
PIPELINE_STAGE_LABELS = dict(PIPELINE_STAGES)
PIPELINE_JOBS = {}
PIPELINE_JOBS_LOCK = threading.Lock()
DEFAULT_REPLY = "I hear what you are feeling. We can slow down and spend a little more time with the most important part."
AVATARS = {
    "companion": {
        "name": "Emotional Companion",
        "reply": "I hear what you are feeling. We can slow down and spend a little more time with the most important part.",
    },
    "mentor": {
        "name": "Growth Mentor",
        "reply": "This can be broken into one small next step. Start with the part you can influence most.",
    },
    "friend": {
        "name": "Close Friend",
        "reply": "I am here. You do not have to make it perfect; just begin wherever your thoughts are.",
    },
    "coach": {
        "name": "Wellbeing Coach",
        "reply": "Take one slow breath first. For now, we only need to notice the clearest signal in your body.",
    },
}
AVATAR_COLORS = ["#32d0a4", "#6fb7ff", "#ffb84d", "#f0798d", "#a78bfa", "#22c55e", "#f97316", "#06b6d4"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password, salt=None):
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, ITERATIONS)
    return salt_bytes.hex(), digest.hex()


def public_user(row):
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "status": row["status"],
        "createdAt": row["created_at"],
        "lastLoginAt": row["last_login_at"] or "",
    }


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
                    "reply": str(override.get("reply") or DEFAULT_REPLY),
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
                "reply": DEFAULT_REPLY,
            }
        )
    return humans


def avatar_meta(avatar_id):
    for human in list_digital_humans():
        if str(human["id"]) == str(avatar_id):
            return {"name": human["name"], "reply": human["reply"]}
    legacy = AVATARS.get(str(avatar_id))
    if legacy:
        return legacy
    return {"name": f"Avatar {avatar_id}", "reply": DEFAULT_REPLY}


def conversation_payload(row):
    avatar = avatar_meta(row["avatar_id"])
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "username": row["username"],
        "avatarId": row["avatar_id"],
        "avatarName": avatar["name"],
        "userText": row["user_text"],
        "replyText": row["reply_text"],
        "userVideoUrl": row["user_video_url"],
        "videoUrl": row["video_url"],
        "audioUrl": row["audio_url"],
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
    DIGITAL_HUMAN_IMAGE_DIR.mkdir(exist_ok=True)
    AVATAR_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id TEXT PRIMARY KEY,
              username TEXT NOT NULL UNIQUE,
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
              user_video_url TEXT NOT NULL DEFAULT '',
              video_url TEXT NOT NULL DEFAULT '',
              audio_url TEXT NOT NULL DEFAULT '',
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


class Handler(SimpleHTTPRequestHandler):
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
        if path == "/api/tts_speakers":
            self.handle_list_tts_speakers()
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
        if path == "/api/recordings":
            self.handle_create_recording()
            return
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "digital_humans"] and parts[3] == "image":
            self.handle_upload_digital_human_image(parts[2])
            return
        if path.startswith("/api/jobs/") and path.endswith("/viewer/render_frame"):
            run_id = path.strip("/").split("/")[2]
            self.handle_render_frame(run_id)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_PATCH(self):
        path = urlparse(self.path).path
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
        user = self.require_user()
        if not user:
            return
        self.send_json({"digitalHumans": list_digital_humans()})

    def handle_list_tts_speakers(self):
        user = self.require_user()
        if not user:
            return
        self.send_json({"speakers": list_tts_speakers(), "defaultSpeakerId": PIPELINE_TTS_SPEAKER_ID})

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
        old_url = str(override.get("image_url") or current.get("imageUrl") or "")
        override["avatar_id"] = avatar_id
        override.setdefault("tts_speaker_id", current["ttsSpeakerId"])
        override.setdefault("name", current["name"])
        override.setdefault("role", current["role"])
        override.setdefault("color", current["color"])
        override["image_url"] = image_url
        overrides[avatar_id] = override
        save_digital_human_overrides(overrides)
        if old_url.startswith("/digital_human_images/"):
            old_path = ROOT / old_url.lstrip("/")
            if old_path.exists() and old_path.is_file():
                old_path.unlink()
        self.send_json({"imageUrl": image_url, "digitalHumans": list_digital_humans()})

    def handle_avatar_respond(self):
        user = self.require_user()
        if not user:
            return
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
                job = self.start_avatar_job(record_id, user, avatar_id, tts_speaker_id, text, audio_file, video_file)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(job, HTTPStatus.ACCEPTED)
            return
        else:
            summary = text[:36]
            reply_text = f"{avatar['reply']} You mentioned \"{summary}\"; record audio/video to generate a real avatar."

        with connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                  (id, user_id, avatar_id, user_text, reply_text, user_video_url, video_url, audio_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, user["id"], avatar_id, text, reply_text, "", video_url, audio_url, created_at),
            )

        self.send_json(
            {
                "id": record_id,
                "avatarId": avatar_id,
                "avatarName": avatar["name"],
                "replyText": reply_text,
                "videoUrl": video_url,
                "audioUrl": audio_url,
                "createdAt": created_at,
            }
        )

    def start_avatar_job(self, record_id, user, avatar_id, tts_speaker_id, text, audio_file, video_file):
        if not AVATAR_SCRIPT.exists():
            raise RuntimeError(f"Avatar pipeline script not found: {AVATAR_SCRIPT}")

        upload_dir = AVATAR_UPLOAD_ROOT / record_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        input_video = None
        if video_file and video_file.get("data"):
            suffix = Path(video_file.get("filename") or "input.webm").suffix.lower() or ".webm"
            if suffix not in {".webm", ".mp4", ".mov", ".m4v"}:
                suffix = ".webm"
            input_video = upload_dir / f"input_video{suffix}"
            input_video.write_bytes(video_file["data"])

        input_wav = upload_dir / "input.wav"
        if audio_file and audio_file.get("data"):
            input_wav.write_bytes(audio_file["data"])
        elif input_video:
            self.extract_audio_from_video(input_video, input_wav)
        else:
            raise RuntimeError("Record video or audio before sending to the avatar system.")
        if input_wav.stat().st_size == 0:
            raise RuntimeError("Captured microphone audio is empty. Turn Mic on and record again.")

        run_id = f"3depb_{record_id.replace('-', '')[:16]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = AVATAR_OUTPUT_ROOT / run_id
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
        ]
        if input_video:
            command.extend(["--input_video", str(input_video)])
        if PIPELINE_NO_LLM:
            command.append("--no_llm")

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
            "userId": user["id"],
            "username": user["username"],
            "text": text,
            "createdAt": now_iso(),
            "runDir": str(run_dir),
            "logPath": str(web_log),
            "replyText": "",
            "inputVideoUrl": self.output_url(input_video),
            "videoUrl": "",
            "audioUrl": "",
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
            )
            try:
                return_code = proc.wait(timeout=PIPELINE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = -9

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
        with PIPELINE_JOBS_LOCK:
            job = PIPELINE_JOBS.get(job_id)
            if not job:
                return
            job["status"] = "done"
            job["stage"] = "done"
            job["stageLabel"] = "Avatar ready"
            job["progress"] = 100
            job["replyText"] = manifest.get("reply_text") or ""
            job["videoUrl"] = self.output_url(output_video)
            job["audioUrl"] = self.output_url(reply_wav)
            avatar = avatar_meta(job["avatarId"])
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversations
                      (id, user_id, avatar_id, user_text, reply_text, user_video_url, video_url, audio_url, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job["userId"],
                        job["avatarId"],
                        job["text"],
                        job["replyText"] or avatar["reply"],
                        job.get("inputVideoUrl") or "",
                        job["videoUrl"],
                        job["audioUrl"],
                        now_iso(),
                    ),
                )

    def handle_avatar_job(self, job_id):
        user = self.require_user()
        if not user:
            return
        payload = self.avatar_job_payload(job_id)
        if not payload:
            self.send_json({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
            return
        if payload.get("userId") != user["id"] and user["role"] != "admin":
            self.send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            return
        self.send_json(payload)

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
                payload["progress"] = min(96, max(5, round((index / total) * 100)))
        if manifest and payload["status"] != "failed":
            output_video = manifest.get("output_video")
            reply_wav = manifest.get("artifact_enhanced_reply_wav") or manifest.get("artifact_reply_wav") or manifest.get("reply_wav")
            payload["replyText"] = manifest.get("reply_text") or payload.get("replyText") or ""
            payload["videoUrl"] = self.output_url(output_video) or payload.get("videoUrl") or ""
            payload["audioUrl"] = self.output_url(reply_wav) or payload.get("audioUrl") or ""
            if not manifest.get("error") and output_video:
                payload["status"] = "done"
                payload["stage"] = "done"
                payload["stageLabel"] = "Avatar ready"
                payload["progress"] = 100
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
        user = self.require_user()
        if not user:
            return
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
        user = self.require_user()
        if not user:
            return
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
        user = self.require_user()
        if not user:
            return
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

        payload = json.dumps(
            {
                "point_path": point_cloud_path,
                "motion_path": motion_path,
                "camera": data.get("camera") or {},
                "frame": max(0, int(data.get("frame") or 0)),
                "width": max(1, int(data.get("width") or 640)),
                "height": max(1, int(data.get("height") or 640)),
                "image_format": "jpeg",
                "quality": 86,
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
        else:
            return None
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return target

    def serve_static(self, path):
        if path == "/":
            path = "/index.html"
        if path.startswith("/vendor/"):
            target = (ROOT / "vendor" / path.removeprefix("/vendor/")).resolve()
            static_root = (ROOT / "vendor").resolve()
        elif path.startswith("/outputs/"):
            target = (AVATAR_OUTPUT_ROOT / path.removeprefix("/outputs/")).resolve()
            static_root = AVATAR_OUTPUT_ROOT.resolve()
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
        self.end_headers()
        self.wfile.write(body)


def main():
    init_db()
    port = int(os.environ.get("PORT", "4173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"3DEPB server running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
