import * as THREE from "./vendor/three.module.js?v=3depb-render-drag-20260614-03";
import { OrbitControls } from "./vendor/OrbitControls.js?v=3depb-render-drag-20260614-03";
import { PLYLoader } from "./vendor/PLYLoader.js?v=3depb-render-drag-20260614-03";

const SESSION_KEY = "3depb_session";
const GUEST_HISTORY_KEY = "3depb_guest_history";
const GUEST_ID_KEY = "3depb_guest_id";
const SAVED_NICKNAME_KEY = "3depb_saved_nickname";
const USER_NICKNAMES_KEY = "3depb_user_nicknames";
const AUTO_SUBMIT_SILENCE_MS = 5000;
const AUTO_SUBMIT_MIN_RECORDING_MS = 1200;
const AUTO_SUBMIT_REARM_MS = 1200;
const AUTO_SUBMIT_MAX_RECORDING_MS = 10000;
const MIN_RECORDING_AUDIO_SECONDS = 0.8;
const MIN_RECORDING_VIDEO_BYTES = 2048;
const RENDER_FRAME_TIMEOUT_MS = 20000;
const DEFAULT_AVATAR_BACKGROUNDS = [
  { id: "study", label: "Study" },
  { id: "bedroom", label: "Bedroom" },
  { id: "sofa", label: "Sofa" },
];
let avatarBackgrounds = DEFAULT_AVATAR_BACKGROUNDS;
let avatars = [
  {
    id: "306",
    avatarId: "306",
    ttsSpeakerId: "6224",
    name: "Avatar 306",
    role: "Avatar 306 · Voice 6224",
    speakerLabel: "6224 · voice",
    color: "#32d0a4",
    reply: ""
  }
];
let digitalHumansLoaded = false;

let cameraStream = null;
let recorder = null;
let recordedChunks = [];
let recordedMimeType = "video/webm";
let recordedVideoBlob = null;
let recordedAudioFile = null;
let recordingStopPromise = null;
let recordingStopResolve = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let monitorGain = null;
let recordingBuffers = [];
let recordingLength = 0;
let recordingSampleRate = 0;
let currentAvatarId = "306";
let avatarSelectionPrompted = false;
let currentAvatarBackgroundId = "study";
let cameraOn = true;
let micOn = true;
let isRecording = false;
let autoCallActive = false;
let callStopped = false;
let autoCaptureState = "idle";
let autoCaptureStartedAt = 0;
let autoSubmissionActive = false;
let autoNextAllowedAt = 0;
let waitingForAvatarPlayback = false;
let autoVad = {
  context: null,
  source: null,
  analyser: null,
  data: null,
  raf: 0,
  speaking: false,
  speechStartedAt: 0,
  listenStartedAt: 0,
  lastVoiceAt: 0,
  baseline: 0.012,
};
let avatarVideoUrl = "";
let avatarAudioUrl = "";
let avatarRunId = "";
let avatarVideoReadyForListening = true;
let avatarVideoLoadError = false;
let avatarVideoCanvasRaf = 0;
let avatarPreviewAudio = null;
const avatarVoicePreviewCache = new Map();
const avatarVoicePreviewRequests = new Map();
const pendingDigitalHumanImages = new Map();
const pendingBoothBackgroundImages = new Map();
let avatarVoicePreviewPreloadTimer = 0;
let activeAvatarView = "video";
let currentReplyText = "";
let currentSubtitleText = "";
let conversations = [];
let guestConversations = [];
let recordings = [];
let currentRecordingPreviewUrl = "";
let generationActive = false;
let generationProgress = 0;
let generationStatus = "Ready";
let generationServerUpdatedAt = 0;
let generationTimer = null;
let autoSubmitTimer = 0;
let renderPreviewTimer = null;
let renderPreviewLoopTimer = null;
let activeJobId = "";
let currentUploadController = null;
let guestId = "";
let guestSessionStartedThisPage = false;
let avatarViewerState = {
  runId: null,
  assets: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  pointCloud: null,
  centerOffset: [0, 0, 0],
  defaultCameraDistance: 1,
  animationId: null,
  fps: 25,
  frameCount: 0,
  loading: false,
  renderPreviewLoading: false,
  renderPreviewDirty: false,
  renderPreviewDirtyReason: "idle",
  pendingRenderReason: "idle",
  renderPreviewRequestId: 0,
  lastRenderedSignature: "",
  renderPreviewObjectUrl: null,
  renderPreviewInteracting: false,
};
let generationPollTimer = null;
let currentRunId = "";

localStorage.removeItem(GUEST_HISTORY_KEY);
localStorage.removeItem(GUEST_ID_KEY);
localStorage.removeItem(SAVED_NICKNAME_KEY);

function getSession() {
  const saved = sessionStorage.getItem(SESSION_KEY);
  return saved ? JSON.parse(saved) : null;
}

function getUserNicknames() {
  try {
    return JSON.parse(localStorage.getItem(USER_NICKNAMES_KEY) || "{}");
  } catch {
    return {};
  }
}

function displayNameForUser(user) {
  if (!user) return "";
  const names = getUserNicknames();
  return String(user.nickname || names[user.id] || user.username || "").trim();
}

function saveDisplayName(user, nickname) {
  const cleanNickname = String(nickname || "").trim().slice(0, 32);
  if (!user || !cleanNickname) return cleanNickname;
  if (!isGuestSession(user)) {
    const names = getUserNicknames();
    names[user.id] = cleanNickname;
    localStorage.setItem(USER_NICKNAMES_KEY, JSON.stringify(names));
  }
  return cleanNickname;
}

function getGuestId() {
  if (!guestId) {
    guestId = `guest-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
  return guestId;
}

function startGuestSession(nickname = "Guest") {
  const cleanNickname = String(nickname || "").trim().slice(0, 32) || "Guest";
  const user = { id: getGuestId(), username: cleanNickname, role: "guest", guest: true };
  guestSessionStartedThisPage = true;
  callStopped = false;
  setSession(user);
  return user;
}

function isGuestSession(session = getSession()) {
  return Boolean(session && (session.guest || session.role === "guest"));
}

function setSession(user) {
  const displayName = displayNameForUser(user) || user.username;
  sessionStorage.setItem(
    SESSION_KEY,
    JSON.stringify({ id: user.id, username: displayName, accountUsername: user.username, role: user.role, guest: Boolean(user.guest) })
  );
}

function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
  guestId = "";
  guestConversations = [];
  guestSessionStartedThisPage = false;
}

function resetGuestRuntimeState() {
  localStorage.removeItem(GUEST_HISTORY_KEY);
  localStorage.removeItem(GUEST_ID_KEY);
  localStorage.removeItem(SAVED_NICKNAME_KEY);
  if (generationPollTimer) clearInterval(generationPollTimer);
  generationPollTimer = null;
  if (generationTimer) clearInterval(generationTimer);
  generationTimer = null;
  if (renderPreviewTimer) clearTimeout(renderPreviewTimer);
  renderPreviewTimer = null;
  if (renderPreviewLoopTimer) clearTimeout(renderPreviewLoopTimer);
  renderPreviewLoopTimer = null;
  if (currentUploadController) {
    currentUploadController.abort();
    currentUploadController = null;
  }
  autoSubmissionActive = false;
  guestId = "";
  guestConversations = [];
  guestSessionStartedThisPage = false;
  conversations = [];
  avatarVideoUrl = "";
  avatarAudioUrl = "";
  avatarRunId = "";
  currentRunId = "";
  activeJobId = "";
  currentReplyText = "";
  currentSubtitleText = "";
  waitingForAvatarPlayback = false;
  avatarVideoReadyForListening = true;
  avatarVideoLoadError = false;
  activeAvatarView = "video";
  generationActive = false;
  generationProgress = 0;
  generationStatus = "Ready";
  recordedChunks = [];
  recordedVideoBlob = null;
  recordedAudioFile = null;
  clearCurrentRecordingPreview();
}

async function hydrateSession() {
  const saved = getSession();
  if (saved && isGuestSession(saved)) {
    if (!guestSessionStartedThisPage) {
      clearSession();
      return null;
    }
    return saved;
  }
  if (saved) return saved;
  try {
    const { user } = await apiFetch("/api/auth/me");
    setSession(user);
    await ensureSignedInNickname(user);
    return user;
  } catch {
    return null;
  }
}

async function apiFetch(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const session = getSession();
  const apiPath = location.protocol === "file:" && path.startsWith("/") ? `http://127.0.0.1:7862${path}` : path;
  let response;
  try {
    response = await fetch(apiPath, {
      credentials: "same-origin",
      headers: isFormData
        ? { ...(isGuestSession(session) ? { "X-Guest-Id": session.id } : {}), ...(options.headers || {}) }
        : {
            "Content-Type": "application/json",
            ...(isGuestSession(session) ? { "X-Guest-Id": session.id } : {}),
            ...(options.headers || {})
          },
      ...options
    });
  } catch (error) {
    throw new Error("Failed to reach the login service. Start it with: bash scripts/avatar.sh 3depb");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

async function loadDigitalHumans() {
  if (digitalHumansLoaded) return;
  try {
    const [payload, backgroundsPayload] = await Promise.all([
      apiFetch("/api/digital_humans"),
      apiFetch("/api/backgrounds").catch(() => ({ backgrounds: DEFAULT_AVATAR_BACKGROUNDS })),
    ]);
    const nextBackgrounds = Array.isArray(backgroundsPayload.backgrounds) ? backgroundsPayload.backgrounds : [];
    avatarBackgrounds = normalizeBoothBackgrounds(nextBackgrounds);
    if (!avatarBackgrounds.some((item) => item.id === currentAvatarBackgroundId)) {
      currentAvatarBackgroundId = avatarBackgrounds[0].id;
    }
    const items = Array.isArray(payload.digitalHumans) ? payload.digitalHumans : [];
    if (items.length) {
      applyDigitalHumansPayload(payload);
      if (!avatars.some((item) => item.id === currentAvatarId)) {
        currentAvatarId = avatars[0].id;
      }
    }
  } catch (error) {
    console.warn("Failed to load digital humans", error);
  } finally {
    digitalHumansLoaded = true;
  }
}

function navigate(hash) {
  location.hash = hash;
}

function avatarSwatchStyle(item) {
  const color = `--avatar-color: ${escapeHtml(item.color || "#32d0a4")};`;
  if (!item.imageUrl) return color;
  const separator = String(item.imageUrl).includes("?") ? "&" : "?";
  const imageUrl = escapeHtml(`${item.imageUrl}${separator}v=${encodeURIComponent(item.imageVersion || "1")}`);
  return `${color} background: var(--avatar-color) url("${imageUrl}") center / cover no-repeat;`;
}

function versionedImageUrl(url, version = "1") {
  if (!url) return "";
  const separator = String(url).includes("?") ? "&" : "?";
  return `${url}${separator}v=${encodeURIComponent(version || "1")}`;
}

function adminAvatarPreviewMarkup(item) {
  const imageUrl = versionedImageUrl(item.imageUrl || "", item.imageVersion || "1");
  return `
    <span class="admin-avatar-preview" data-human-preview="${escapeHtml(item.id)}" style="--avatar-color: ${escapeHtml(item.color || "#32d0a4")};">
      ${imageUrl ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(item.name || item.id)} avatar" loading="lazy" />` : ""}
    </span>
  `;
}

function avatarSwatchMarkup(item) {
  const imageUrl = versionedImageUrl(item.imageUrl || "", item.imageVersion || "1");
  return `
    <span class="avatar-swatch" style="--avatar-color: ${escapeHtml(item.color || "#32d0a4")};">
      ${imageUrl ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(item.name || item.id)} avatar" loading="lazy" />` : ""}
    </span>
  `;
}

function avatarBackgroundStyle(item) {
  if (!item?.imageUrl) return "";
  const separator = String(item.imageUrl).includes("?") ? "&" : "?";
  const backgroundUrl = escapeHtml(`${item.imageUrl}${separator}v=${encodeURIComponent(item.imageVersion || "1")}`);
  return `--custom-avatar-background: url('${backgroundUrl}');`;
}

function adminBackgroundPreviewMarkup(item) {
  const imageUrl = versionedImageUrl(item.imageUrl || item.image_url || "", item.imageVersion || item.image_version || "1");
  return `
    <span class="admin-background-preview ${imageUrl ? "has-image" : ""}" style="${avatarBackgroundStyle(item)}">
      ${imageUrl ? `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(item.label || item.id || "Background")} background" loading="lazy" />` : ""}
    </span>
  `;
}

function avatarDisplayDescription(item) {
  let text = String(item.role || "");
  text = text.replace(/\s*·\s*Voice\s+\S+/gi, "");
  text = text.replace(/\s*·\s*\d{3,6}\s*·.*$/g, "");
  text = text.replace(/\s*Voice\s+\S+/gi, "");
  return text.trim() || String(item.name || "");
}

function updateAvatarImageInMemory(avatarId, imageUrl) {
  const version = Date.now();
  avatars = avatars.map((item) =>
    String(item.id) === String(avatarId) || String(item.avatarId) === String(avatarId)
      ? { ...item, imageUrl, imageVersion: version }
      : item
  );
}

function applyDigitalHumansPayload(payload) {
  const items = Array.isArray(payload?.digitalHumans) ? payload.digitalHumans : [];
  if (!items.length) return;
  avatars = items.map((item, index) => ({
    id: String(item.id || item.avatarId),
    avatarId: String(item.avatarId || item.id),
    ttsSpeakerId: String(item.ttsSpeakerId || "6224"),
    name: String(item.name || item.label || `Avatar ${item.avatarId || item.id}`),
    role: String(item.role || `Avatar ${item.avatarId || item.id} · Voice ${item.ttsSpeakerId || "6224"}`),
    speakerLabel: String(item.speakerLabel || `Voice ${item.ttsSpeakerId || "6224"}`),
    color: String(item.color || ["#32d0a4", "#6fb7ff", "#ffb84d", "#f0798d"][index % 4]),
    imageUrl: String(item.imageUrl || ""),
    backgroundUrl: String(item.backgroundUrl || ""),
    reply: String(item.reply || ""),
  }));
}

function updateAvatarBackgroundInMemory(avatarId, backgroundUrl) {
  const version = Date.now();
  avatars = avatars.map((item) =>
    String(item.id) === String(avatarId) || String(item.avatarId) === String(avatarId)
      ? { ...item, backgroundUrl, backgroundVersion: version }
      : item
  );
}

function normalizeBoothBackgrounds(items, version = "1") {
  const source = Array.isArray(items) && items.length ? items : DEFAULT_AVATAR_BACKGROUNDS;
  return source.map((item) => ({
    id: String(item.id || "study"),
    label: String(item.label || item.name || item.id || "Study"),
    imageUrl: String(item.imageUrl || item.image_url || ""),
    imageVersion: item.imageVersion || item.image_version || version,
  }));
}

function applyBoothBackgroundsPayload(payload) {
  const items = Array.isArray(payload?.backgrounds) ? payload.backgrounds : [];
  if (!items.length) return;
  avatarBackgrounds = normalizeBoothBackgrounds(items, Date.now());
  if (!avatarBackgrounds.some((item) => item.id === currentAvatarBackgroundId)) {
    currentAvatarBackgroundId = avatarBackgrounds[0].id;
  }
}

function updateAdminImagePreview(avatarId, imageUrl) {
  const rowInput = document.querySelector(`[data-human-image="${cssEscape(avatarId)}"]`);
  const row = rowInput ? rowInput.closest("tr") : null;
  const preview = row ? row.querySelector(".admin-avatar-preview") : null;
  const avatar = avatars.find((item) => String(item.id) === String(avatarId) || String(item.avatarId) === String(avatarId));
  if (preview && avatar) {
    preview.setAttribute("style", `--avatar-color: ${escapeHtml(avatar.color || "#32d0a4")};`);
    preview.innerHTML = imageUrl
      ? `<img src="${escapeHtml(versionedImageUrl(imageUrl, Date.now()))}" alt="${escapeHtml(avatar.name || avatarId)} avatar" />`
      : "";
  }
}

function requireAuth() {
  const session = getSession();
  if (!session) {
    navigate("#/");
    return null;
  }
  return session;
}

function requireSignedIn() {
  const session = getSession();
  return session && !isGuestSession(session) ? session : null;
}

function topbar(title, session, options = {}) {
  const adminButton =
    session.role === "admin"
      ? `
        <button class="secondary-btn" data-action="room">Conversation</button>
        <button class="secondary-btn" data-action="digital-humans-admin">Digital Humans Management</button>
        <button class="secondary-btn" data-action="admin">User Management</button>
      `
      : "";
  const authButton = isGuestSession(session)
    ? '<button class="secondary-btn" data-action="open-auth">Log In / Register</button>'
    : '<button class="danger-btn" data-action="logout">Log Out</button>';
  const displayName = displayNameForUser(session) || session.username;
  return `
    <header class="topbar">
      <div class="topbar-left">
        ${options.back ? `<button class="secondary-btn compact-btn" type="button" data-action="back-home" data-back-target="${escapeHtml(options.backTarget || "#/")}">Back</button>` : ""}
        <div class="topbar-title"><span class="status-dot"></span>${escapeHtml(title)}</div>
      </div>
      <div class="topbar-actions">
        <button class="badge name-badge-btn" type="button" data-action="rename-user">${escapeHtml(displayName)}</button>
        ${adminButton}
        ${options.history === false ? "" : '<button class="secondary-btn" type="button" data-action="history">History</button>'}
        ${authButton}
      </div>
    </header>
  `;
}

function renderLogin() {
  stopCamera();
  document.querySelector("#app").innerHTML = `
    <main class="page login-page">
      <button class="login-corner-btn" type="button" id="cornerLoginBtn">Log In</button>
      <section class="login-shell">
        <div class="brand-stage">
          <div class="booth-visual" aria-hidden="true"></div>
          <div class="brand-copy">
            <h1 class="brand-title">3D Emotional Phone Booth</h1>
            <p class="brand-subtitle">Have immersive emotional conversations with your chosen digital human, then record, review, and manage every session.</p>
            <button class="hero-start-btn" type="button" id="startGuestBtn">
              <span>Start</span>
            </button>
          </div>
        </div>
      </section>
    </main>
  `;

  document.querySelector("#cornerLoginBtn").addEventListener("click", async () => {
    try {
      await openAuthModal();
      callStopped = false;
      avatarSelectionPrompted = false;
      navigate("#/room");
    } catch {
      // User cancelled the modal.
    }
  });
  document.querySelector("#startGuestBtn").addEventListener("click", () => {
    startGuestOnboarding().catch((error) => {
      if (error && error.message !== "Nickname cancelled" && error.message !== "Avatar selection cancelled") {
        alert(error.message || String(error));
      }
    });
  });
}

async function startGuestOnboarding() {
  const nickname = await openNicknameModal({ title: "Enter guest nickname" });
  startGuestSession(nickname);
  avatarSelectionPrompted = false;
  await loadDigitalHumans();
  await openAvatarSelectionModal({ requireSelection: true, showCancel: false });
  navigate("#/room");
}

function openNicknameModal(options = {}) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector("#nicknameModal");
    if (existing) existing.remove();
    const wrap = document.createElement("div");
    wrap.id = "nicknameModal";
    wrap.className = "modal-backdrop";
    const allowCancel = options.allowCancel !== false;
    wrap.innerHTML = `
      <form class="auth-card modal-card nickname-modal" id="nicknameForm">
        <h2>${escapeHtml(options.title || "Enter your name")}</h2>
        <div class="field">
          <input id="guestNickname" name="nickname" maxlength="32" autocomplete="nickname" placeholder="Enter your name" value="${escapeHtml(options.value || "")}" required />
        </div>
        <div class="auth-actions">
          ${allowCancel ? '<button class="secondary-btn" type="button" data-modal-close>Cancel</button>' : ""}
          <button class="primary-btn" type="submit">Next</button>
        </div>
        <p class="error" id="nicknameError"></p>
      </form>
    `;
    document.body.appendChild(wrap);
    const form = wrap.querySelector("#nicknameForm");
    const input = wrap.querySelector("#guestNickname");
    input.focus();
    const closeButton = wrap.querySelector("[data-modal-close]");
    if (closeButton) {
      closeButton.addEventListener("click", () => {
        wrap.remove();
        reject(new Error("Nickname cancelled"));
      });
    }
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const nickname = String(new FormData(form).get("nickname") || "").trim();
      if (!nickname) {
        wrap.querySelector("#nicknameError").textContent = "Please enter a nickname.";
        return;
      }
      wrap.remove();
      resolve(nickname);
    });
  });
}

async function ensureSignedInNickname(user) {
  if (!user || isGuestSession(user) || String(user.nickname || "").trim()) return user;
  const nickname = await openNicknameModal({
    title: "Set your display name",
    value: user.username || "",
    allowCancel: false,
  });
  const cleanNickname = saveDisplayName(user, nickname);
  const payload = await apiFetch("/api/auth/nickname", {
    method: "PATCH",
    body: JSON.stringify({ nickname: cleanNickname }),
  });
  const nextUser = payload.user || { ...user, nickname: cleanNickname };
  setSession(nextUser);
  return nextUser;
}

function openAuthModal() {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector("#authModal");
    if (existing) existing.remove();
    const wrap = document.createElement("div");
    wrap.id = "authModal";
    wrap.className = "modal-backdrop";
    wrap.innerHTML = `
      <form class="auth-card modal-card" id="modalAuthForm">
        <div class="auth-tabs">
          <button class="secondary-btn compact-btn is-active" type="button" data-modal-auth-mode="login">Log In</button>
          <button class="secondary-btn compact-btn" type="button" data-modal-auth-mode="register">Register</button>
        </div>
        <div class="field">
          <label for="modalUsername">Username</label>
          <input id="modalUsername" name="username" autocomplete="username" placeholder="Enter username" required />
        </div>
        <div class="field">
          <label for="modalPassword">Password</label>
          <input id="modalPassword" name="password" type="password" autocomplete="current-password" placeholder="Enter password" required />
        </div>
        <div class="auth-actions">
          <button class="secondary-btn" type="button" data-modal-close>Cancel</button>
          <button class="primary-btn" type="submit">Log In</button>
        </div>
        <p class="error" id="modalAuthError"></p>
      </form>
    `;
    document.body.appendChild(wrap);
    let mode = "login";
    const form = wrap.querySelector("#modalAuthForm");
    const submit = form.querySelector("button[type='submit']");
    const password = form.querySelector("#modalPassword");
    wrap.querySelectorAll("[data-modal-auth-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        mode = button.dataset.modalAuthMode;
        wrap.querySelectorAll("[data-modal-auth-mode]").forEach((item) => item.classList.toggle("is-active", item === button));
        submit.textContent = mode === "register" ? "Register" : "Log In";
        password.autocomplete = mode === "register" ? "new-password" : "current-password";
      });
    });
    wrap.querySelector("[data-modal-close]").addEventListener("click", () => {
      wrap.remove();
      reject(new Error("Login cancelled"));
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const error = wrap.querySelector("#modalAuthError");
      error.textContent = "";
      submit.disabled = true;
      try {
        const previousSession = getSession();
        const { user } = await apiFetch(mode === "register" ? "/api/auth/register" : "/api/auth/login", {
          method: "POST",
          body: JSON.stringify({
            username: String(data.get("username")).trim(),
            password: String(data.get("password")),
          }),
        });
        setSession(user);
        if (isGuestSession(previousSession)) {
          resetGuestRuntimeState();
        }
        wrap.remove();
        let nextUser = user;
        try {
          nextUser = await ensureSignedInNickname(user);
        } catch (nicknameError) {
          alert(nicknameError.message || "Nickname could not be saved. You can set it from the name badge later.");
        }
        resolve(nextUser);
      } catch (err) {
        error.textContent = err.message;
      } finally {
        submit.disabled = false;
      }
    });
  });
}

function renderRoom() {
  const session = requireAuth();
  if (!session) return;
  if (!callStopped && currentReplyText === "Conversation stopped.") {
    currentReplyText = "";
  }

  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  const displayName = displayNameForUser(session) || session.username;
  const background = avatarBackgroundById(currentAvatarBackgroundId);
  const waitingForAvatarVideo = Boolean(avatarVideoUrl && !avatarVideoReadyForListening && !avatarVideoLoadError);
  const listeningTitle = callStopped
    ? "Stopped"
    : waitingForAvatarPlayback
      ? "Playing response"
      : waitingForAvatarVideo
        ? "Loading video"
        : avatarVideoLoadError
          ? "Video unavailable"
          : "Listening";
  const showVoiceWave = !callStopped && !waitingForAvatarPlayback && !waitingForAvatarVideo && !avatarVideoLoadError;
  const captionText = currentSubtitleText || currentReplyText;
  document.querySelector("#app").innerHTML = `
    <main class="page">
      ${topbar("3D Emotional Phone Booth", session, { back: true })}
      <section class="room">
        <article class="video-pane">
          <div class="video-frame">
            <video id="localVideo" autoplay muted playsinline ${cameraStream && cameraOn ? "" : "hidden"}></video>
            <div class="video-name-label">${escapeHtml(displayName)}</div>
            <div class="camera-placeholder" id="cameraPlaceholder" ${cameraStream && cameraOn ? "hidden" : ""}>
              <div>
                <h2>${escapeHtml(session.username)}</h2>
                <p class="hint">${cameraOn ? "Requesting camera and microphone access" : "Camera is off"}</p>
              </div>
            </div>
          </div>
        </article>

        <article class="video-pane">
          <div class="video-frame">
            <div class="video-name-label avatar-name-label">${escapeHtml(avatar.name)}</div>
            <div class="avatar-stage avatar-bg-${escapeHtml(background.id)} ${background.imageUrl ? "has-custom-bg" : ""}" style="--avatar-color: ${avatar.color}; ${avatarBackgroundStyle(background)}">
              ${
                avatarVideoUrl
                  ? `<video id="avatarVideo" class="avatar-video" src="${escapeHtml(avatarVideoUrl)}" playsinline controls ${activeAvatarView === "video" ? "" : "hidden"}></video><canvas id="avatarVideoCanvas" class="avatar-video-canvas" ${activeAvatarView === "video" ? "" : "hidden"}></canvas>`
                  : ""
              }
              <div class="avatar-render-view" id="avatarRenderView" ${activeAvatarView === "rendered" ? "" : "hidden"}>
                <div class="avatar-render-stage" id="avatarRenderStage">
                  <canvas id="avatarRenderCanvas"></canvas>
                  <img id="avatarRenderImage" alt="Rendered Gaussian preview" hidden />
                  <div class="avatar-render-loading" id="avatarRenderLoading" hidden>
                    <span class="avatar-render-spinner"></span>
                    <span>Rendering view...</span>
                  </div>
                  <div class="avatar-render-notice" id="avatarRenderNotice">${avatarRunId ? "Loading 3D render assets..." : "Generate an avatar reply first, then open 3D Render."}</div>
                  <div class="avatar-render-status" id="avatarRenderStatus"></div>
                  <div class="avatar-render-controls">
                    <button class="secondary-btn compact-btn" type="button" data-avatar-view="video" aria-selected="false">Video</button>
                    <button class="icon-btn" type="button" data-action="render-play" title="Play/Pause">Play</button>
                    <input id="avatarRenderSeek" type="range" min="0" max="1" step="0.001" value="0" aria-label="3D render timeline" />
                    <span class="avatar-render-time" id="avatarRenderTime">0:00 / 0:00 · frame 0</span>
                    <button class="secondary-btn compact-btn" type="button" data-action="render-fit">Fit</button>
                  </div>
                  <audio id="avatarRenderAudio" preload="metadata" ${avatarAudioUrl ? `src="${escapeHtml(avatarAudioUrl)}"` : ""}></audio>
                </div>
              </div>
              <div class="avatar-person" aria-hidden="true" ${avatarVideoUrl ? "hidden" : ""}>
                <div class="avatar-head"></div>
                <div class="avatar-body"></div>
              </div>
              <div class="generation-progress" ${generationActive ? "" : "hidden"}>
                <div class="generation-track">
                  <span style="width: ${Math.max(4, generationProgress)}%"></span>
                </div>
              </div>
              <div class="room-control-cluster" ${activeAvatarView === "rendered" ? "hidden" : ""}>
                <div class="avatar-caption" ${captionText ? "" : "hidden"}>
                  <div class="reply-box" id="replyBox">${escapeHtml(captionText)}</div>
                </div>
                <div class="room-control-row">
                  <div class="avatar-view-tabs" role="tablist" aria-label="Avatar view">
                    <button class="${activeAvatarView === "video" ? "active" : ""}" type="button" data-avatar-view="video" aria-selected="${activeAvatarView === "video"}">Video</button>
                    <button class="${activeAvatarView === "rendered" ? "active" : ""}" type="button" data-avatar-view="rendered" aria-selected="${activeAvatarView === "rendered"}">3D Render</button>
                  </div>
                  <label class="background-select">
                    <span>Background</span>
                    <select data-action="avatar-background">
                      ${avatarBackgrounds.map((item) => `<option value="${escapeHtml(item.id)}" ${item.id === background.id ? "selected" : ""}>${escapeHtml(item.label)}</option>`).join("")}
                    </select>
                  </label>
                  <div class="listening-panel" ${generationActive ? "hidden" : ""}>
                    <div class="listening-copy">
                      <strong>${listeningTitle}</strong>
                    </div>
                    ${
                      showVoiceWave
                        ? '<div class="voice-wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>'
                        : ""
                    }
                  </div>
                  <span class="badge avatar-status-badge ${generationActive ? "is-generating" : ""}">${generationActive ? "Generating avatar" : "Digital human online"}</span>
                  <div class="toolbar" id="roomToolbar">
                    <button class="icon-btn ${micOn ? "" : "is-off"}" title="Toggle audio" data-action="toggle-mic">${micOn ? "Mic" : "Off"}</button>
                    <button class="icon-btn ${cameraOn ? "" : "is-off"}" title="Toggle video" data-action="toggle-camera">${cameraOn ? "Cam" : "Off"}</button>
                    <button class="icon-btn ${callStopped ? "action-btn" : "stop-btn"}" title="${callStopped ? "Resume recording" : "Stop recording and generation"}" data-action="toggle-call-state">${callStopped ? "Action" : "Stop"}</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </article>

        <audio id="avatarAudio" ${avatarAudioUrl && !avatarVideoUrl ? `src="${escapeHtml(avatarAudioUrl)}"` : ""} hidden></audio>
      </section>
    </main>
  `;

  bindTopbar();
  bindRoom();
  bindMediaExclusion();
  scheduleLocalPreviewRestore();
}

function renderHistoryPage() {
  const session = requireAuth();
  if (!session) return;

  stopCamera();
  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  document.querySelector("#app").innerHTML = `
    <main class="page">
      ${topbar("History", session, { back: true, backTarget: "#/room", history: false })}
      <section class="history-page-stack">
        <section class="history-page-panel history-panel">
          <div class="history-head">
            <div class="pane-title">
              <strong>${escapeHtml(avatar.name)} History</strong>
              <span>Conversation records and exported videos</span>
            </div>
            <div class="history-actions">
              <button class="secondary-btn compact-btn" data-action="refresh-history">Refresh</button>
              <button class="secondary-btn compact-btn" data-action="export-history">Export All</button>
              <button class="danger-btn compact-btn" data-action="clear-history">Clear</button>
            </div>
          </div>
          <div class="history-list history-page-list" id="historyList">
            ${renderConversationHistory(conversations)}
          </div>
        </section>
      </section>
    </main>
  `;

  bindTopbar();
  bindHistoryPage();
  loadConversations();
}

function bindHistoryPage() {
  document.querySelector("[data-action='refresh-history']").addEventListener("click", loadConversations);
  document.querySelector("[data-action='clear-history']").addEventListener("click", clearHistory);
  document.querySelector("[data-action='export-history']").addEventListener("click", exportHistory);
  bindMediaExclusion();
}

function bindTopbar() {
  document.querySelectorAll("[data-action='back-home']").forEach((button) => {
    button.addEventListener("click", async () => {
      if (button.dataset.backTarget !== "#/room") {
        await stopConversation();
      }
      stopCamera();
      navigate(button.dataset.backTarget || "#/");
    });
  });
  document.querySelectorAll("[data-action='logout']").forEach((button) => {
    button.addEventListener("click", async () => {
      await apiFetch("/api/auth/logout", { method: "POST" }).catch(() => {});
      clearSession();
      stopCamera();
      navigate("#/");
    });
  });
  document.querySelectorAll("[data-action='open-auth']").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await openAuthModal();
        await loadDigitalHumans();
        renderRoom();
      } catch {
        // User cancelled the modal.
      }
    });
  });
  document.querySelectorAll("[data-action='rename-user']").forEach((button) => {
    button.addEventListener("click", async () => {
      const session = getSession();
      if (!session) return;
      try {
        const nickname = await openNicknameModal({ value: displayNameForUser(session) || session.username });
        const cleanNickname = saveDisplayName(session, nickname);
        let nextSession = { ...session, username: cleanNickname, nickname: cleanNickname };
        if (!isGuestSession(session)) {
          const payload = await apiFetch("/api/auth/nickname", {
            method: "PATCH",
            body: JSON.stringify({ nickname: cleanNickname }),
          });
          nextSession = payload.user || nextSession;
        }
        setSession(nextSession);
        if (location.hash === "#/history") {
          renderHistoryPage();
        } else {
          renderRoom();
        }
      } catch {
        // User cancelled the modal.
      }
    });
  });
  document.querySelectorAll("[data-action='admin']").forEach((button) => {
    button.addEventListener("click", () => navigate("#/admin"));
  });
  document.querySelectorAll("[data-action='digital-humans-admin']").forEach((button) => {
    button.addEventListener("click", () => navigate("#/digital-humans"));
  });
  document.querySelectorAll("[data-action='room']").forEach((button) => {
    button.addEventListener("click", () => navigate("#/room"));
  });
  document.querySelectorAll("[data-action='history']").forEach((button) => {
    button.addEventListener("click", async () => {
      await stopConversation({ render: false });
      stopCamera();
      navigate("#/history");
    });
  });
}

function bindRoom() {
  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  document.querySelectorAll("[data-avatar-view]").forEach((button) => {
    button.addEventListener("click", () => {
      setAvatarView(button.dataset.avatarView);
    });
  });

  document.querySelectorAll("[data-avatar]").forEach((button) => {
    button.addEventListener("click", () => {
      currentAvatarId = button.dataset.avatar;
      renderRoom();
    });
  });

  const backgroundSelect = document.querySelector("[data-action='avatar-background']");
  if (backgroundSelect) {
    backgroundSelect.addEventListener("change", () => {
      currentAvatarBackgroundId = avatarBackgroundById(backgroundSelect.value).id;
      const stage = document.querySelector(".avatar-stage");
      if (stage) {
        stage.classList.remove(...avatarBackgrounds.map((item) => `avatar-bg-${item.id}`));
        stage.classList.add(`avatar-bg-${currentAvatarBackgroundId}`);
        stage.classList.toggle("has-custom-bg", Boolean(avatarBackgroundById(currentAvatarBackgroundId).imageUrl));
        stage.setAttribute("style", `--avatar-color: ${avatar?.color || "#32d0a4"}; ${avatarBackgroundStyle(avatarBackgroundById(currentAvatarBackgroundId))}`);
      }
    });
  }

  document.querySelector("[data-action='toggle-mic']").addEventListener("click", () => {
    micOn = !micOn;
    if (cameraStream) {
      cameraStream.getAudioTracks().forEach((track) => {
        track.enabled = micOn;
      });
    }
    updateMediaControlsDom();
  });

  document.querySelector("[data-action='toggle-camera']").addEventListener("click", async () => {
    const shouldTurnOn = !cameraOn || !hasLiveCameraStream();
    cameraOn = shouldTurnOn;
    if (cameraOn) {
      await ensureCameraStream();
    }
    if (cameraStream) {
      cameraStream.getVideoTracks().forEach((track) => {
        track.enabled = cameraOn;
      });
    }
    updateMediaControlsDom();
  });
  document.querySelector("[data-action='toggle-call-state']").addEventListener("click", () => {
    if (callStopped) {
      resumeConversation().catch((error) => {
        currentReplyText = error.message || String(error);
        renderRoom();
      });
      return;
    }
    stopConversation().catch((error) => {
      currentReplyText = error.message || String(error);
      renderRoom();
    });
  });

  const renderPlayButton = document.querySelector("[data-action='render-play']");
  if (renderPlayButton) renderPlayButton.addEventListener("click", toggleRenderPlayback);
  const renderFitButton = document.querySelector("[data-action='render-fit']");
  if (renderFitButton) renderFitButton.addEventListener("click", () => {
    fitAvatarRenderCamera();
    scheduleRenderPreview(0, "idle");
  });
  const renderSeek = document.querySelector("#avatarRenderSeek");
  if (renderSeek) {
    renderSeek.addEventListener("input", () => {
      const audio = document.querySelector("#avatarRenderAudio");
      if (!audio || !Number.isFinite(audio.duration) || audio.duration <= 0) return;
      audio.currentTime = Number(renderSeek.value) * audio.duration;
      updateRenderTimeline();
      scheduleRenderPreview(0, "seek");
    });
  }
  const renderAudio = document.querySelector("#avatarRenderAudio");
  if (renderAudio) {
    renderAudio.addEventListener("play", startRenderPreviewLoop);
    renderAudio.addEventListener("pause", () => {
      stopRenderPreviewLoop();
      scheduleRenderPreview(80, "idle");
    });
    renderAudio.addEventListener("ended", () => {
      stopRenderPreviewLoop();
      scheduleRenderPreview(80, "idle");
    });
    renderAudio.addEventListener("loadedmetadata", updateRenderTimeline);
  }
  const renderControls = document.querySelector(".avatar-render-controls");
  if (renderControls) {
    ["pointerdown", "pointermove", "pointerup", "mousedown", "mousemove", "mouseup", "touchstart", "touchmove", "touchend"].forEach((eventName) => {
      renderControls.addEventListener(eventName, (event) => event.stopPropagation());
    });
  }
  bindAvatarVideoReadiness();
  loadConversations();
  if (activeAvatarView === "rendered") {
    loadAvatarRenderView();
  }
}

function updateMediaControlsDom() {
  const micButton = document.querySelector("[data-action='toggle-mic']");
  if (micButton) {
    micButton.textContent = micOn ? "Mic" : "Off";
    micButton.classList.toggle("is-off", !micOn);
  }
  const cameraButton = document.querySelector("[data-action='toggle-camera']");
  if (cameraButton) {
    cameraButton.textContent = cameraOn ? "Cam" : "Off";
    cameraButton.classList.toggle("is-off", !cameraOn);
  }
  updateLocalVideo();
}

function pauseAvatarMediaExcept(activeMedia) {
  document.querySelectorAll("#avatarVideo, #avatarAudio, #avatarRenderAudio, .history-video, .recording-player").forEach((media) => {
    if (media === activeMedia) return;
    if (typeof media.pause === "function" && !media.paused) {
      media.pause();
    }
  });
}

function bindMediaExclusion() {
  document.querySelectorAll("#avatarVideo, #avatarAudio, #avatarRenderAudio, .history-video, .recording-player").forEach((media) => {
    media.addEventListener("play", () => pauseAvatarMediaExcept(media));
  });
}

async function startCamera() {
  const video = document.querySelector("#localVideo");
  if (!video) return;

  try {
    await ensureCameraStream();
    updateLocalVideo();
    maybeStartAutoCall();
  } catch (error) {
    stopCamera();
    const video = document.querySelector("#localVideo");
    const placeholder = document.querySelector("#cameraPlaceholder");
    if (video) video.hidden = true;
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.querySelector(".hint").textContent = "Cannot access the camera or microphone. Please check browser permissions.";
    }
  }
}

async function restoreLocalPreview() {
  try {
    await ensureCameraStream();
    updateLocalVideo();
    updateMediaControlsDom();
  } catch (error) {
    const placeholder = document.querySelector("#cameraPlaceholder");
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.querySelector(".hint").textContent = "Cannot access the camera or microphone. Please check browser permissions.";
    }
  }
}

function scheduleLocalPreviewRestore() {
  const restoreAndListen = () => restoreLocalPreview().then(() => maybeStartAutoCall()).catch(() => {});
  restoreAndListen();
  requestAnimationFrame(restoreAndListen);
  setTimeout(restoreAndListen, 250);
  setTimeout(restoreAndListen, 900);
}

function hasLiveCameraStream() {
  return Boolean(cameraStream && cameraStream.getTracks().some((track) => track.readyState === "live"));
}

async function ensureCameraStream() {
  if (!hasLiveCameraStream()) {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  }
  cameraStream.getAudioTracks().forEach((track) => {
    track.enabled = micOn;
  });
  cameraStream.getVideoTracks().forEach((track) => {
    track.enabled = cameraOn;
  });
  updateLocalVideo();
  return cameraStream;
}

function avatarVideoBlocksListening() {
  return Boolean(avatarVideoUrl && (!avatarVideoReadyForListening || waitingForAvatarPlayback || avatarVideoLoadError));
}

function canAutoListen() {
  return Boolean(hasLiveCameraStream() && !callStopped && !generationActive && !autoSubmissionActive && !waitingForAvatarPlayback && !avatarVideoBlocksListening());
}

async function maybeStartAutoCall() {
  if (callStopped || generationActive || autoSubmissionActive || waitingForAvatarPlayback || avatarVideoBlocksListening()) {
    updateListeningPanelDom();
    return;
  }
  try {
    await ensureCameraStream();
  } catch {
    updateListeningPanelDom();
    return;
  }
  if (!canAutoListen()) return;
  startAutoCall();
}

function bindAvatarVideoReadiness() {
  const video = document.querySelector("#avatarVideo");
  if (!video) {
    avatarVideoReadyForListening = true;
    avatarVideoLoadError = false;
    stopAvatarVideoCanvas();
    updateListeningPanelDom();
    return;
  }

  const markReady = () => {
    avatarVideoReadyForListening = !waitingForAvatarPlayback;
    avatarVideoLoadError = false;
    startAvatarVideoCanvas(video);
    restoreLocalPreview();
    updateListeningPanelDom();
  };
  const markEnded = async () => {
    waitingForAvatarPlayback = false;
    avatarVideoReadyForListening = true;
    avatarVideoLoadError = false;
    autoNextAllowedAt = performance.now() + AUTO_SUBMIT_REARM_MS;
    cameraOn = true;
    await restoreLocalPreview();
    updateListeningPanelDom();
    await maybeStartAutoCall();
  };
  const markError = () => {
    avatarVideoReadyForListening = false;
    waitingForAvatarPlayback = false;
    avatarVideoLoadError = true;
    stopAutoCall();
    updateListeningPanelDom();
  };

  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    markReady();
  } else {
    video.addEventListener("loadeddata", markReady, { once: true });
    video.addEventListener("canplay", markReady, { once: true });
    video.addEventListener("canplaythrough", markReady, { once: true });
    video.addEventListener("error", markError, { once: true });
  }
  video.addEventListener("play", () => startAvatarVideoCanvas(video));
  video.addEventListener("pause", () => stopAvatarVideoCanvas({ keepFrame: true }));
  video.addEventListener("ended", markEnded, { once: true });
}

function stopAvatarVideoCanvas(options = {}) {
  if (avatarVideoCanvasRaf) cancelAnimationFrame(avatarVideoCanvasRaf);
  avatarVideoCanvasRaf = 0;
  const video = document.querySelector("#avatarVideo");
  const canvas = document.querySelector("#avatarVideoCanvas");
  if (video && !options.keepFrame) video.classList.remove("is-canvas-source");
  if (canvas && !options.keepFrame) {
    canvas.hidden = true;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (context) context.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function startAvatarVideoCanvas(video) {
  const canvas = document.querySelector("#avatarVideoCanvas");
  if (!video || !canvas || activeAvatarView !== "video") return;
  if (!video.videoWidth || !video.videoHeight) return;
  if (avatarVideoCanvasRaf) cancelAnimationFrame(avatarVideoCanvasRaf);

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.hidden = false;
  video.classList.add("is-canvas-source");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;

  const drawFrame = () => {
    if (activeAvatarView !== "video" || video.hidden || video.ended) {
      avatarVideoCanvasRaf = 0;
      return;
    }
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = context.getImageData(0, 0, canvas.width, canvas.height);
      removeConnectedBlackBackdrop(frame);
      context.putImageData(frame, 0, 0);
    }
    avatarVideoCanvasRaf = requestAnimationFrame(drawFrame);
  };
  drawFrame();
}

function removeConnectedBlackBackdrop(frame) {
  const { data: pixels, width, height } = frame;
  const total = width * height;
  const backdrop = new Uint8Array(total);
  const stack = new Int32Array(total);
  const headProtection = estimateAvatarHeadProtection(pixels, width, height);
  let stackSize = 0;

  const enqueue = (pixelIndex) => {
    if (pixelIndex < 0 || pixelIndex >= total || backdrop[pixelIndex]) return;
    const byteIndex = pixelIndex * 4;
    if (!isAvatarBackdropPixel(pixels, byteIndex, 24)) return;
    if (headProtection && isInsideAvatarHeadProtection(pixelIndex, width, headProtection) && pixelLuma(pixels, byteIndex) > 14) return;
    backdrop[pixelIndex] = 1;
    stack[stackSize] = pixelIndex;
    stackSize += 1;
  };

  for (let x = 0; x < width; x += 1) {
    enqueue(x);
    enqueue((height - 1) * width + x);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueue(y * width);
    enqueue(y * width + width - 1);
  }

  while (stackSize > 0) {
    stackSize -= 1;
    const current = stack[stackSize];
    const x = current % width;
    const y = (current - x) / width;
    if (x > 0) enqueue(current - 1);
    if (x < width - 1) enqueue(current + 1);
    if (y > 0) enqueue(current - width);
    if (y < height - 1) enqueue(current + width);
  }

  for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
    if (backdrop[pixelIndex]) pixels[pixelIndex * 4 + 3] = 0;
  }

  for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
    if (backdrop[pixelIndex]) continue;
    const byteIndex = pixelIndex * 4;
    if (headProtection && isInsideAvatarHeadProtection(pixelIndex, width, headProtection) && pixelLuma(pixels, byteIndex) > 18) continue;
    if (!isAvatarBackdropPixel(pixels, byteIndex, 44)) continue;
    const x = pixelIndex % width;
    const touchesBackdrop =
      (x > 0 && backdrop[pixelIndex - 1]) ||
      (x < width - 1 && backdrop[pixelIndex + 1]) ||
      (pixelIndex >= width && backdrop[pixelIndex - width]) ||
      (pixelIndex < total - width && backdrop[pixelIndex + width]);
    if (!touchesBackdrop) continue;
    const r = pixels[byteIndex];
    const g = pixels[byteIndex + 1];
    const b = pixels[byteIndex + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    pixels[byteIndex + 3] = Math.max(96, Math.min(255, Math.round(((luma - 42) / 34) * 255)));
  }

  featherAvatarMatte(pixels, width, height, backdrop, headProtection);
}

function isAvatarBackdropPixel(pixels, index, lumaLimit) {
  const r = pixels[index];
  const g = pixels[index + 1];
  const b = pixels[index + 2];
  const luma = pixelLuma(pixels, index);
  const channelSpread = Math.max(r, g, b) - Math.min(r, g, b);
  const coolOrNeutralBlack = b >= r - 2 && b >= g - 8 && channelSpread <= 18;
  return luma < lumaLimit && coolOrNeutralBlack;
}

function pixelLuma(pixels, index) {
  return 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
}

function estimateAvatarHeadProtection(pixels, width, height) {
  let left = width;
  let right = 0;
  let top = height;
  let bottom = 0;
  let count = 0;
  const startY = Math.floor(height * 0.06);
  const endY = Math.floor(height * 0.94);
  const startX = Math.floor(width * 0.12);
  const endX = Math.floor(width * 0.88);

  for (let y = startY; y < endY; y += 2) {
    for (let x = startX; x < endX; x += 2) {
      const index = (y * width + x) * 4;
      if (!isAvatarSkinPixel(pixels, index)) continue;
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
      count += 1;
    }
  }

  if (count < 120 || right <= left || bottom <= top) return null;

  const skinWidth = right - left;
  const skinHeight = bottom - top;
  const cx = (left + right) / 2;
  const cy = top + skinHeight * 0.44;
  return {
    cx,
    cy,
    rx: Math.max(skinWidth * 0.78, width * 0.18),
    ry: Math.max(skinHeight * 0.76, height * 0.24),
  };
}

function isAvatarSkinPixel(pixels, index) {
  const r = pixels[index];
  const g = pixels[index + 1];
  const b = pixels[index + 2];
  const luma = pixelLuma(pixels, index);
  return luma > 62 && luma < 230 && r > 72 && g > 42 && b > 28 && r > b + 12 && r >= g - 8 && r - Math.min(g, b) > 16;
}

function isInsideAvatarHeadProtection(pixelIndex, width, protection) {
  const x = pixelIndex % width;
  const y = (pixelIndex - x) / width;
  const dx = (x - protection.cx) / protection.rx;
  const dy = (y - protection.cy) / protection.ry;
  return dx * dx + dy * dy <= 1;
}

function featherAvatarMatte(pixels, width, height, backdrop, headProtection) {
  const total = width * height;
  const nextAlpha = new Uint8ClampedArray(total);

  for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
    nextAlpha[pixelIndex] = pixels[pixelIndex * 4 + 3];
  }

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const pixelIndex = y * width + x;
      const byteIndex = pixelIndex * 4;
      if (pixels[byteIndex + 3] === 0) continue;

      const alphaSamples =
        pixels[(pixelIndex - 1) * 4 + 3] +
        pixels[(pixelIndex + 1) * 4 + 3] +
        pixels[(pixelIndex - width) * 4 + 3] +
        pixels[(pixelIndex + width) * 4 + 3] +
        pixels[(pixelIndex - width - 1) * 4 + 3] +
        pixels[(pixelIndex - width + 1) * 4 + 3] +
        pixels[(pixelIndex + width - 1) * 4 + 3] +
        pixels[(pixelIndex + width + 1) * 4 + 3];
      if (alphaSamples === 8 * 255) continue;

      const touchesBackdrop =
        backdrop[pixelIndex - 1] ||
        backdrop[pixelIndex + 1] ||
        backdrop[pixelIndex - width] ||
        backdrop[pixelIndex + width] ||
        backdrop[pixelIndex - width - 1] ||
        backdrop[pixelIndex - width + 1] ||
        backdrop[pixelIndex + width - 1] ||
        backdrop[pixelIndex + width + 1];
      if (!touchesBackdrop) continue;

      const protectedHair = headProtection && isInsideAvatarHeadProtection(pixelIndex, width, headProtection) && pixelLuma(pixels, byteIndex) > 28;
      const matteAlpha = Math.round((pixels[byteIndex + 3] * 4 + alphaSamples) / 12);
      nextAlpha[pixelIndex] = protectedHair ? Math.max(210, matteAlpha) : Math.max(132, matteAlpha);
    }
  }

  for (let pixelIndex = 0; pixelIndex < total; pixelIndex += 1) {
    pixels[pixelIndex * 4 + 3] = nextAlpha[pixelIndex];
  }
}

function updateLocalVideo() {
  const video = document.querySelector("#localVideo");
  const placeholder = document.querySelector("#cameraPlaceholder");
  if (!video || !placeholder) return;

  const streamReady = hasLiveCameraStream();
  if (streamReady && video.srcObject !== cameraStream) {
    video.srcObject = cameraStream;
  } else if (!streamReady) {
    video.srcObject = null;
  }
  video.hidden = !streamReady || !cameraOn;
  placeholder.hidden = Boolean(streamReady && cameraOn);
  if (!video.hidden) {
    video.play().catch(() => {});
  }
}

function stopCamera() {
  stopAutoCall();
  clearAutoSubmitDeadline();
  if (!cameraStream) return;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
}

async function stopConversation(options = {}) {
  const shouldRender = options.render !== false;
  callStopped = true;
  stopAutoCall();
  clearAutoSubmitDeadline();
  autoSubmissionActive = false;
  waitingForAvatarPlayback = false;
  autoVad.speaking = false;
  autoNextAllowedAt = performance.now() + 1000000;
  if (currentUploadController) {
    currentUploadController.abort();
    currentUploadController = null;
  }
  const jobToCancel = activeJobId;
  activeJobId = "";
  if (generationPollTimer) clearInterval(generationPollTimer);
  generationPollTimer = null;
  if (generationTimer) clearInterval(generationTimer);
  generationTimer = null;
  if (isRecording) {
    await stopRecording();
  }
  recordedChunks = [];
  recordedVideoBlob = null;
  recordedAudioFile = null;
  clearCurrentRecordingPreview();
  stopGenerationProgress("Stopped", 0);
  currentReplyText = "Conversation stopped.";
  currentSubtitleText = "Conversation stopped.";
  if (jobToCancel) {
    await apiFetch(`/api/avatar/jobs/${encodeURIComponent(jobToCancel)}/cancel`, { method: "POST" }).catch(() => {});
  }
  if (shouldRender) {
    renderRoom();
    scheduleLocalPreviewRestore();
  }
}

async function resumeConversation() {
  callStopped = false;
  currentReplyText = "";
  currentSubtitleText = "";
  autoSubmissionActive = false;
  waitingForAvatarPlayback = false;
  autoVad.speaking = false;
  autoNextAllowedAt = performance.now() + 600;
  if (!cameraStream) {
    await startCamera();
  } else {
    cameraStream.getAudioTracks().forEach((track) => {
      track.enabled = micOn;
    });
    cameraStream.getVideoTracks().forEach((track) => {
      track.enabled = cameraOn;
    });
    maybeStartAutoCall();
  }
  renderRoom();
}

function startAutoCall() {
  if (autoCallActive || !canAutoListen()) return;
  const AudioCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtor) return;
  if (currentReplyText === "Conversation stopped.") {
    currentReplyText = "";
    const replyBox = document.querySelector("#replyBox");
    if (replyBox) replyBox.textContent = "";
  }
  autoCallActive = true;
  autoVad.context = new AudioCtor();
  autoVad.source = autoVad.context.createMediaStreamSource(cameraStream);
  autoVad.analyser = autoVad.context.createAnalyser();
  autoVad.analyser.fftSize = 1024;
  autoVad.data = new Uint8Array(autoVad.analyser.fftSize);
  autoVad.source.connect(autoVad.analyser);
  autoVad.speaking = false;
  autoVad.lastVoiceAt = 0;
  autoVad.speechStartedAt = 0;
  autoVad.listenStartedAt = performance.now();
  autoNextAllowedAt = Math.max(autoNextAllowedAt, performance.now() + 400);
  if (!isRecording) {
    startRecording({ silent: true }).then(() => {
      scheduleAutoSubmitDeadline();
    }).catch((error) => {
      currentReplyText = error.message;
    });
  } else {
    scheduleAutoSubmitDeadline();
  }
  autoVadLoop();
}

function stopAutoCall() {
  autoCallActive = false;
  clearAutoSubmitDeadline();
  if (autoVad.raf) cancelAnimationFrame(autoVad.raf);
  autoVad.raf = 0;
  if (autoVad.source) autoVad.source.disconnect();
  if (autoVad.context) autoVad.context.close().catch(() => {});
  autoVad.context = null;
  autoVad.source = null;
  autoVad.analyser = null;
  autoVad.data = null;
  autoVad.speaking = false;
  autoVad.listenStartedAt = 0;
}

function clearAutoSubmitDeadline() {
  if (autoSubmitTimer) {
    clearTimeout(autoSubmitTimer);
    autoSubmitTimer = 0;
  }
}

function scheduleAutoSubmitDeadline() {
  clearAutoSubmitDeadline();
  autoSubmitTimer = window.setTimeout(() => {
    autoSubmitTimer = 0;
    if (autoSubmissionActive || generationActive || waitingForAvatarPlayback || callStopped) return;
    if (!isRecording) return;
    submitAutoTurn().catch((error) => {
      autoSubmissionActive = false;
      currentReplyText = error.message;
      currentSubtitleText = error.message;
      renderRoom();
    });
  }, AUTO_SUBMIT_MAX_RECORDING_MS);
}

function autoVadLoop() {
  if (!autoCallActive || !autoVad.analyser || !autoVad.data) return;
  if (!canAutoListen()) {
    stopAutoCall();
    return;
  }
  autoVad.analyser.getByteTimeDomainData(autoVad.data);
  let sum = 0;
  for (const value of autoVad.data) {
    const centered = (value - 128) / 128;
    sum += centered * centered;
  }
  const rms = Math.sqrt(sum / autoVad.data.length);
  const now = performance.now();
  const threshold = Math.max(0.012, autoVad.baseline * 2.4);
  autoVad.baseline = autoVad.speaking ? autoVad.baseline : autoVad.baseline * 0.96 + rms * 0.04;
  if (!autoVad.listenStartedAt) autoVad.listenStartedAt = now;
  if (!autoSubmissionActive && now >= autoNextAllowedAt && rms > threshold) {
    autoVad.lastVoiceAt = now;
    if (!autoVad.speaking) {
      autoVad.speaking = true;
      autoVad.speechStartedAt = now;
      if (!isRecording) startRecording({ silent: true }).catch((error) => {
        currentReplyText = error.message;
      });
    }
  }
  if (
    isRecording &&
    (
      (autoVad.speaking &&
        now - autoVad.lastVoiceAt > AUTO_SUBMIT_SILENCE_MS &&
        now - autoVad.speechStartedAt > AUTO_SUBMIT_MIN_RECORDING_MS) ||
      now - autoVad.listenStartedAt > AUTO_SUBMIT_MAX_RECORDING_MS
    )
  ) {
    autoVad.speaking = false;
    submitAutoTurn().catch((error) => {
      autoSubmissionActive = false;
      currentReplyText = error.message;
      renderRoom();
    });
  }
  autoVad.raf = requestAnimationFrame(autoVadLoop);
}

function setAvatarView(view) {
  activeAvatarView = view === "rendered" ? "rendered" : "video";
  const video = document.querySelector("#avatarVideo");
  const videoCanvas = document.querySelector("#avatarVideoCanvas");
  const renderView = document.querySelector("#avatarRenderView");
  const toolbar = document.querySelector("#roomToolbar");
  const controlCluster = document.querySelector(".room-control-cluster");
  const caption = document.querySelector(".avatar-caption");
  const statusRow = document.querySelector(".room-control-row");
  document.querySelectorAll("[data-avatar-view]").forEach((button) => {
    const active = button.dataset.avatarView === activeAvatarView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  if (video) {
    video.hidden = activeAvatarView !== "video";
    if (activeAvatarView !== "video") video.pause();
  }
  if (videoCanvas) videoCanvas.hidden = activeAvatarView !== "video";
  if (renderView) renderView.hidden = activeAvatarView !== "rendered";
  if (toolbar) toolbar.hidden = activeAvatarView === "rendered";
  if (controlCluster) controlCluster.hidden = false;
  if (caption) caption.hidden = activeAvatarView === "rendered";
  if (statusRow) statusRow.hidden = activeAvatarView === "rendered";
  if (activeAvatarView === "rendered") {
    loadAvatarRenderView();
  } else {
    const renderAudio = document.querySelector("#avatarRenderAudio");
    if (renderAudio) renderAudio.pause();
    stopRenderPreviewLoop();
    const avatarVideo = document.querySelector("#avatarVideo");
    if (avatarVideo) startAvatarVideoCanvas(avatarVideo);
  }
}

async function loadAvatarRenderView() {
  const notice = document.querySelector("#avatarRenderNotice");
  if (!avatarRunId) {
    if (notice) {
      notice.hidden = false;
      notice.textContent = "Generate an avatar reply first, then open 3D Render.";
    }
    return;
  }
  initAvatarRenderScene();
  if (avatarViewerState.loading || avatarViewerState.runId === avatarRunId) {
    resizeAvatarRender();
    scheduleRenderPreview(0, "idle");
    return;
  }

  avatarViewerState.loading = true;
  avatarViewerState.runId = avatarRunId;
  avatarViewerState.lastRenderedSignature = "";
  avatarViewerState.renderPreviewRequestId += 1;
  if (notice) {
    notice.hidden = false;
    notice.textContent = "Loading 3D render assets...";
  }
  try {
    const assets = await apiFetch(`/api/jobs/${encodeURIComponent(avatarRunId)}/viewer_assets`);
    avatarViewerState.assets = assets;
    avatarViewerState.fps = assets.fps || 25;
    avatarViewerState.frameCount = assets.frame_count || 0;
    const audio = document.querySelector("#avatarRenderAudio");
    if (audio && (assets.audio_url || avatarAudioUrl)) {
      audio.src = assets.audio_url || avatarAudioUrl;
      audio.load();
    }
    if (!assets.point_cloud_url) {
      throw new Error("Point cloud is not available for this run.");
    }
    await loadAvatarPointCloud(assets.point_cloud_url);
    if (notice) notice.hidden = true;
    updateRenderTimeline();
    scheduleRenderPreview(0, "idle");
  } catch (error) {
    if (notice) {
      notice.hidden = false;
      notice.textContent = `3D Render failed: ${error.message || error}`;
    }
  } finally {
    avatarViewerState.loading = false;
  }
}

function initAvatarRenderScene() {
  if (avatarViewerState.renderer) return;
  const canvas = document.querySelector("#avatarRenderCanvas");
  const stage = document.querySelector("#avatarRenderStage");
  if (!canvas) return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x111416, 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111416);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(0, 0.04, 1.25);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.update();
  controls.addEventListener("start", () => {
    avatarViewerState.renderPreviewInteracting = true;
    scheduleRenderPreview(0, "drag");
  });
  controls.addEventListener("change", () => {
    scheduleRenderPreview(avatarViewerState.renderPreviewInteracting ? 35 : 140, avatarViewerState.renderPreviewInteracting ? "drag" : "idle");
  });
  controls.addEventListener("end", () => {
    avatarViewerState.renderPreviewInteracting = false;
    scheduleRenderPreview(50, "drag");
  });

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(0.8, 1.4, 1.0);
  scene.add(key);

  avatarViewerState.renderer = renderer;
  avatarViewerState.scene = scene;
  avatarViewerState.camera = camera;
  avatarViewerState.controls = controls;
  resizeAvatarRender();
  startAvatarRenderLoop();
}

async function loadAvatarPointCloud(url) {
  const loader = new PLYLoader();
  loader.setCustomPropertyNameMapping({
    fdc: ["f_dc_0", "f_dc_1", "f_dc_2"],
  });
  const geometry = await loader.loadAsync(url);
  applyGaussianDcColors(geometry);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const center = new THREE.Vector3();
  if (geometry.boundingBox) geometry.boundingBox.getCenter(center);
  avatarViewerState.centerOffset = [center.x, center.y, center.z];
  geometry.center();
  geometry.computeBoundingSphere();
  const radius = (geometry.boundingSphere && geometry.boundingSphere.radius) || 1;
  const material = new THREE.PointsMaterial({
    color: 0xf1f5f2,
    vertexColors: geometry.hasAttribute("color"),
    size: Math.max(radius / 280, 0.0015),
    sizeAttenuation: true,
  });
  const pointCloud = new THREE.Points(geometry, material);
  if (avatarViewerState.pointCloud) {
    avatarViewerState.scene.remove(avatarViewerState.pointCloud);
    avatarViewerState.pointCloud.geometry.dispose();
    avatarViewerState.pointCloud.material.dispose();
  }
  avatarViewerState.pointCloud = pointCloud;
  avatarViewerState.scene.add(pointCloud);
  fitAvatarRenderCamera();
}

function applyGaussianDcColors(geometry) {
  const fdc = geometry.getAttribute("fdc");
  if (!fdc) return;
  const shC0 = 0.28209479177387814;
  const colors = new Float32Array(fdc.count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < fdc.count; i += 1) {
    color
      .setRGB(
        Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getX(i))),
        Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getY(i))),
        Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getZ(i)))
      )
      .convertSRGBToLinear();
    colors[i * 3 + 0] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

function fitAvatarRenderCamera() {
  const object = avatarViewerState.pointCloud;
  if (!object || !avatarViewerState.camera || !avatarViewerState.controls) return;
  object.geometry.computeBoundingSphere();
  const sphere = object.geometry.boundingSphere;
  const radius = Math.max((sphere && sphere.radius) || 1, 0.01);
  const distance = Math.max(radius * 2.6, 0.8);
  avatarViewerState.defaultCameraDistance = distance;
  avatarViewerState.camera.position.set(0, radius * 0.12, distance);
  avatarViewerState.camera.near = Math.max(radius / 200, 0.001);
  avatarViewerState.camera.far = Math.max(radius * 20, 10);
  avatarViewerState.camera.updateProjectionMatrix();
  avatarViewerState.controls.target.set(0, 0, 0);
  avatarViewerState.controls.update();
}

function startAvatarRenderLoop() {
  if (!avatarViewerState.renderer || avatarViewerState.animationId) return;
  const tick = () => {
    avatarViewerState.animationId = requestAnimationFrame(tick);
    if (avatarViewerState.controls) avatarViewerState.controls.update();
    updateRenderTimeline();
    avatarViewerState.renderer.render(avatarViewerState.scene, avatarViewerState.camera);
  };
  tick();
}

function resizeAvatarRender() {
  const stage = document.querySelector("#avatarRenderStage");
  if (!stage || !avatarViewerState.renderer || !avatarViewerState.camera) return;
  const rect = stage.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  avatarViewerState.renderer.setSize(width, height, false);
  avatarViewerState.camera.aspect = width / height;
  avatarViewerState.camera.updateProjectionMatrix();
}

async function toggleRenderPlayback() {
  if (activeAvatarView !== "rendered") {
    setAvatarView("rendered");
  }
  if (avatarRunId && !avatarViewerState.assets && !avatarViewerState.loading) {
    await loadAvatarRenderView();
  }
  const audio = document.querySelector("#avatarRenderAudio");
  const video = document.querySelector("#avatarVideo");
  const status = document.querySelector("#avatarRenderStatus");
  if (!audio || !audio.src) {
    if (status) status.textContent = "3D Render audio is not ready yet.";
    return;
  }
  if (audio.paused) {
    pauseAvatarMediaExcept(audio);
    if (video) video.pause();
    audio.play().catch((error) => {
      if (status) status.textContent = `Could not play render audio: ${error.message || error}`;
    });
  } else {
    audio.pause();
  }
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function updateRenderTimeline() {
  const audio = document.querySelector("#avatarRenderAudio");
  const seek = document.querySelector("#avatarRenderSeek");
  const time = document.querySelector("#avatarRenderTime");
  const button = document.querySelector("[data-action='render-play']");
  if (!audio || !seek || !time) return;
  const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
  if (duration > 0 && document.activeElement !== seek) {
    seek.value = String(Math.min(1, Math.max(0, current / duration)));
  }
  const totalFrames = avatarViewerState.frameCount || Math.max(0, Math.floor(duration * avatarViewerState.fps));
  const frame = Math.min(Math.max(0, Math.floor(current * avatarViewerState.fps)), Math.max(0, totalFrames - 1));
  time.textContent = `${formatTime(current)} / ${formatTime(duration)} · frame ${frame}${totalFrames ? `/${totalFrames}` : ""}`;
  if (button) button.textContent = audio.paused ? "Play" : "Stop";
}

function currentRenderFrame() {
  const audio = document.querySelector("#avatarRenderAudio");
  const current = audio && Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
  const total = avatarViewerState.frameCount || Math.max(1, Math.floor(((audio && audio.duration) || 0) * avatarViewerState.fps));
  return Math.min(Math.max(0, Math.floor(current * avatarViewerState.fps)), Math.max(0, total - 1));
}

function renderReasonPriority(reason) {
  return { drag: 4, seek: 3, playback: 2, idle: 1 }[reason] || 0;
}

function startRenderPreviewLoop() {
  if (renderPreviewLoopTimer) return;
  const tick = () => {
    const audio = document.querySelector("#avatarRenderAudio");
    if (activeAvatarView !== "rendered" || !audio || audio.paused) {
      renderPreviewLoopTimer = null;
      return;
    }
    scheduleRenderPreview(0, avatarViewerState.renderPreviewInteracting ? "drag" : "playback");
    renderPreviewLoopTimer = setTimeout(tick, avatarViewerState.renderPreviewInteracting ? 180 : 80);
  };
  tick();
}

function stopRenderPreviewLoop() {
  if (!renderPreviewLoopTimer) return;
  clearTimeout(renderPreviewLoopTimer);
  renderPreviewLoopTimer = null;
}

function getRenderPreviewDimensions(reason = "idle") {
  const stage = document.querySelector("#avatarRenderStage");
  const rect = stage ? stage.getBoundingClientRect() : null;
  const baseWidth = Math.max(320, Math.floor((rect && rect.width) || 640));
  const baseHeight = Math.max(320, Math.floor((rect && rect.height) || 640));
  const scale = reason === "drag" ? 0.46 : reason === "playback" ? 0.36 : 0.82;
  const maxSize = reason === "playback" ? 420 : reason === "drag" ? 560 : 900;
  return {
    width: Math.max(224, Math.min(maxSize, Math.round(baseWidth * scale))),
    height: Math.max(224, Math.min(maxSize, Math.round(baseHeight * scale))),
  };
}

function getRenderCameraPayload() {
  if (!avatarViewerState.camera || !avatarViewerState.controls) return null;
  const position = avatarViewerState.camera.position;
  const target = avatarViewerState.controls.target;
  const up = avatarViewerState.camera.up;
  const offset = new THREE.Vector3().subVectors(position, target);
  const distance = Math.max(offset.length(), 1e-6);
  const direction = offset.clone().normalize();
  return {
    position: [position.x, position.y, position.z],
    target: [target.x, target.y, target.z],
    up: [up.x, up.y, up.z],
    direction: [direction.x, direction.y, direction.z],
    radius_norm: Math.min(2.5, Math.max(0.55, distance / Math.max(avatarViewerState.defaultCameraDistance, 1e-6))),
    base_radius: 1.0,
    fov: 20,
    near: avatarViewerState.camera.near,
    far: avatarViewerState.camera.far,
    center_offset: avatarViewerState.centerOffset,
  };
}

function getRenderPreviewSignature(camera, frame, width, height) {
  const round = (value) => Number(value || 0).toFixed(3);
  return JSON.stringify({
    f: frame,
    w: width,
    h: height,
    p: camera.position.map(round),
    t: camera.target.map(round),
    r: round(camera.radius_norm),
  });
}

function setRenderPreviewLoading(isLoading) {
  const loading = document.querySelector("#avatarRenderLoading");
  if (loading) loading.hidden = !isLoading;
}

function scheduleRenderPreview(delay = 500, reason = "idle") {
  if (activeAvatarView !== "rendered" || !avatarRunId || !avatarViewerState.pointCloud) return;
  if (avatarViewerState.renderPreviewLoading) {
    avatarViewerState.renderPreviewDirty = true;
    if (renderReasonPriority(reason) >= renderReasonPriority(avatarViewerState.renderPreviewDirtyReason)) {
      avatarViewerState.renderPreviewDirtyReason = reason;
    }
    return;
  }
  if (renderPreviewTimer) clearTimeout(renderPreviewTimer);
  if (renderReasonPriority(reason) >= renderReasonPriority(avatarViewerState.pendingRenderReason)) {
    avatarViewerState.pendingRenderReason = reason;
  }
  renderPreviewTimer = setTimeout(() => {
    renderPreviewTimer = null;
    const nextReason = avatarViewerState.pendingRenderReason || reason;
    avatarViewerState.pendingRenderReason = "idle";
    requestRenderedFrame(nextReason);
  }, delay);
}

async function fetchWithTimeout(url, options = {}, timeoutMs = RENDER_FRAME_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function responseErrorMessage(response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text);
    return payload.error || payload.detail || text;
  } catch {
    return text || `Request failed: ${response.status}`;
  }
}

async function requestRenderedFrame(reason = "idle") {
  if (avatarViewerState.renderPreviewLoading || activeAvatarView !== "rendered") return;
  const camera = getRenderCameraPayload();
  if (!camera) return;
  const frame = currentRenderFrame();
  const { width, height } = getRenderPreviewDimensions(reason);
  const signature = getRenderPreviewSignature(camera, frame, width, height);
  if (signature === avatarViewerState.lastRenderedSignature && reason !== "idle") return;
  const requestId = avatarViewerState.renderPreviewRequestId + 1;
  avatarViewerState.renderPreviewRequestId = requestId;
  avatarViewerState.renderPreviewLoading = true;
  avatarViewerState.renderPreviewDirty = false;
  avatarViewerState.renderPreviewDirtyReason = "idle";
  setRenderPreviewLoading(true);
  const status = document.querySelector("#avatarRenderStatus");
  if (status) status.textContent = `Rendering frame ${frame}${reason === "playback" ? " (playback)" : reason === "drag" ? " (drag)" : ""}...`;

  try {
    const response = await fetchWithTimeout(`/api/jobs/${encodeURIComponent(avatarRunId)}/viewer/render_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera, frame, width, height, reason }),
    });
    if (!response.ok) throw new Error(await responseErrorMessage(response));
    const frameValue = response.headers.get("X-Frame");
    const frameHeader = Number(frameValue === null ? frame : frameValue);
    const imageBlob = await response.blob();
    const nextSrc = URL.createObjectURL(imageBlob);
    await preloadImage(nextSrc);
    if (requestId !== avatarViewerState.renderPreviewRequestId || activeAvatarView !== "rendered") {
      URL.revokeObjectURL(nextSrc);
      return;
    }
    const image = document.querySelector("#avatarRenderImage");
    const previousObjectUrl = avatarViewerState.renderPreviewObjectUrl;
    if (image) {
      image.classList.add("is-swapping");
      image.hidden = false;
      requestAnimationFrame(() => {
        image.src = nextSrc;
        image.classList.remove("is-swapping");
      });
    }
    avatarViewerState.renderPreviewObjectUrl = nextSrc;
    if (previousObjectUrl) {
      setTimeout(() => URL.revokeObjectURL(previousObjectUrl), 1000);
    }
    avatarViewerState.lastRenderedSignature = signature;
    if (status) status.textContent = `Rendered frame ${frameHeader}.`;
  } catch (error) {
    const message = error.name === "AbortError"
      ? "3D Render timed out. Check gaussian_render_worker on port 8792."
      : `Render preview failed: ${error.message || error}`;
    if (status) status.textContent = message;
  } finally {
    if (requestId === avatarViewerState.renderPreviewRequestId) {
      avatarViewerState.renderPreviewLoading = false;
      setRenderPreviewLoading(false);
      if (avatarViewerState.renderPreviewDirty && activeAvatarView === "rendered") {
        const dirtyReason = avatarViewerState.renderPreviewDirtyReason || "idle";
        avatarViewerState.renderPreviewDirtyReason = "idle";
        scheduleRenderPreview(dirtyReason === "playback" ? 0 : 30, dirtyReason);
      }
    }
  }
}

function preloadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve();
    image.onerror = reject;
    image.src = src;
  });
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function encodeWav(buffers, length, sampleRate) {
  const samples = new Float32Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    samples.set(buffer, offset);
    offset += buffer.length;
  }
  const dataLength = samples.length * 2;
  const arrayBuffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(arrayBuffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataLength, true);
  let pos = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(pos, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    pos += 2;
  }
  return new Blob([arrayBuffer], { type: "audio/wav" });
}

async function startRecording(options = {}) {
  if (!window.MediaRecorder) {
    alert("This browser does not support MediaRecorder recording.");
    return;
  }
  if (!cameraStream) {
    await startCamera();
    if (!cameraStream) {
      alert("Cannot start recording. Please allow camera and microphone access first.");
      return;
    }
  }
  micOn = true;
  cameraOn = true;
  cameraStream.getAudioTracks().forEach((track) => {
    track.enabled = true;
  });
  cameraStream.getVideoTracks().forEach((track) => {
    track.enabled = true;
  });
  clearCurrentRecordingPreview();
  recordedChunks = [];
  recordedVideoBlob = null;
  recordedAudioFile = null;
  recordingBuffers = [];
  recordingLength = 0;
  recordingSampleRate = 0;
  recordedMimeType = getSupportedRecordingType();
  recorder = recordedMimeType
    ? new MediaRecorder(cameraStream, { mimeType: recordedMimeType })
    : new MediaRecorder(cameraStream);
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  recordingSampleRate = audioContext.sampleRate;
  sourceNode = audioContext.createMediaStreamSource(cameraStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  monitorGain = audioContext.createGain();
  monitorGain.gain.value = 0;
  processorNode.onaudioprocess = (event) => {
    if (!micOn) return;
    const channel = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(channel.length);
    copy.set(channel);
    recordingBuffers.push(copy);
    recordingLength += copy.length;
  };
  sourceNode.connect(processorNode);
  processorNode.connect(monitorGain);
  monitorGain.connect(audioContext.destination);
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  });
  recordingStopPromise = new Promise((resolve) => {
    recordingStopResolve = resolve;
  });
  recorder.addEventListener("stop", async () => {
    recordedVideoBlob = new Blob(recordedChunks, { type: recordedMimeType || "video/webm" });
    if (recordingLength > 0 && audioContext) {
      const wavBlob = encodeWav(recordingBuffers, recordingLength, audioContext.sampleRate);
      recordedAudioFile = new File([wavBlob], `3depb-audio-${Date.now()}.wav`, { type: "audio/wav" });
    }
    if (sourceNode) sourceNode.disconnect();
    if (processorNode) processorNode.disconnect();
    if (monitorGain) monitorGain.disconnect();
    if (audioContext) await audioContext.close();
    sourceNode = null;
    processorNode = null;
    monitorGain = null;
    audioContext = null;
    isRecording = false;
    if (!options.silent) updateCurrentRecordingPreview();
    if (recordingStopResolve) recordingStopResolve();
    recordingStopResolve = null;
    recordingStopPromise = null;
    if (!options.silent) renderRoom();
  });
  recorder.start(250);
  isRecording = true;
  autoCaptureStartedAt = Date.now();
  autoVad.listenStartedAt = performance.now();
  if (!options.silent) renderRoom();
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
  recorder = null;
  isRecording = false;
  return recordingStopPromise || Promise.resolve();
}

function getSupportedRecordingType() {
  const types = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function sendTalk() {
  const replyBox = document.querySelector("#replyBox");
  const text = "Recorded video/audio turn";

  stopAutoCall();
  if (isRecording) {
    await stopRecording();
  }
  if (!recordedChunks.length) {
    if (replyBox) replyBox.textContent = "Listening is active. Speak for a moment, then pause to send automatically.";
    return;
  }
  const audioSeconds = recordingSampleRate > 0 ? recordingLength / recordingSampleRate : 0;
  const videoBlob = recordedVideoBlob || new Blob(recordedChunks, { type: recordedMimeType || "video/webm" });
  if (!recordedAudioFile || audioSeconds < MIN_RECORDING_AUDIO_SECONDS) {
    if (replyBox) replyBox.textContent = "Recording was too short or silent. Please speak a little longer, then pause 5 seconds.";
    recordedChunks = [];
    recordedVideoBlob = null;
    recordedAudioFile = null;
    autoNextAllowedAt = performance.now() + AUTO_SUBMIT_REARM_MS;
    return;
  }
  if (videoBlob.size < MIN_RECORDING_VIDEO_BYTES) {
    if (replyBox) replyBox.textContent = "Camera recording was empty. Please try again with the camera on.";
    recordedChunks = [];
    recordedVideoBlob = null;
    recordedAudioFile = null;
    autoNextAllowedAt = performance.now() + AUTO_SUBMIT_REARM_MS;
    return;
  }
  beginGenerationProgress();

  try {
    const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
    const form = new FormData();
    form.append("avatarId", avatar.avatarId || currentAvatarId);
    form.append("ttsSpeakerId", avatar.ttsSpeakerId || "6224");
    form.append("background", currentAvatarBackgroundId);
    form.append("text", text);
    if (recordedAudioFile) {
      form.append("audio", recordedAudioFile, recordedAudioFile.name);
    }
    form.append("video", videoBlob, `3depb-video-${Date.now()}.webm`);
    setGenerationProgress(18, "Uploading media");
    currentUploadController = new AbortController();
    const session = getSession();
    const rawResponse = await fetch("/api/avatar/respond", {
      method: "POST",
      credentials: "same-origin",
      headers: isGuestSession(session) ? { "X-Guest-Id": session.id, "X-Guest-Name": session.username || "Guest" } : {},
      body: form,
      signal: currentUploadController.signal,
    });
    currentUploadController = null;
    const response = await rawResponse.json().catch(() => ({}));
    if (!rawResponse.ok) {
      throw new Error(response.error || "Avatar generation failed");
    }
    showGenerationWarnings(response);
    if (response.jobId) {
      activeJobId = response.jobId;
      await pollAvatarJob(response.jobId);
    } else {
      applyAvatarResult(response);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      stopGenerationProgress("Stopped", 0);
      currentReplyText = "Conversation stopped.";
      if (replyBox) replyBox.textContent = currentReplyText;
      return;
    }
    stopGenerationProgress("Listening", 0);
    currentReplyText = err.message;
    if (replyBox) replyBox.textContent = err.message;
  } finally {
    currentUploadController = null;
    autoSubmissionActive = false;
  }
}

async function submitAutoTurn() {
  if (autoSubmissionActive || !isRecording || generationActive || waitingForAvatarPlayback || callStopped) return;
  clearAutoSubmitDeadline();
  autoSubmissionActive = true;
  await stopRecording();
  if (Date.now() - autoCaptureStartedAt < AUTO_SUBMIT_MIN_RECORDING_MS) {
    recordedChunks = [];
    recordedVideoBlob = null;
    recordedAudioFile = null;
    autoSubmissionActive = false;
    autoNextAllowedAt = performance.now() + AUTO_SUBMIT_REARM_MS;
    return;
  }
  if (recordedChunks.length) {
    await sendTalk();
  }
  autoSubmissionActive = false;
  autoNextAllowedAt = performance.now() + AUTO_SUBMIT_REARM_MS;
}

function workerHealthWarning(payload) {
  if (payload?.workerHealthWarning) return payload.workerHealthWarning;
  const health = payload?.workerHealth || {};
  const offline = Object.entries(health)
    .filter(([, item]) => !item?.ok)
    .map(([name]) => name);
  return offline.length ? `Workers offline: ${offline.join(", ")}` : "";
}

function showGenerationWarnings(payload) {
  const warnings = [payload?.inputVideoWarning, workerHealthWarning(payload)].filter(Boolean);
  if (!warnings.length) return;
  const status = document.querySelector("#avatarRenderStatus");
  const replyBox = document.querySelector("#replyBox");
  const message = warnings.join(" ");
  if (status) status.textContent = message;
  if (replyBox && !generationActive) replyBox.textContent = message;
  console.warn("[3depb]", message);
}

async function pollAvatarJob(jobId) {
  if (generationPollTimer) clearInterval(generationPollTimer);
  activeJobId = jobId;
  const tick = async () => {
    try {
      const response = await apiFetch(`/api/avatar/jobs/${encodeURIComponent(jobId)}`);
      showGenerationWarnings(response);
      applyPartialAvatarResult(response);
      setGenerationProgress(response.progress || generationProgress, response.stageLabel || response.stage || "Running", true);
      currentRunId = response.runId || currentRunId;
      if (response.status === "done" || response.videoUrl) {
        clearInterval(generationPollTimer);
        generationPollTimer = null;
        activeJobId = "";
        applyAvatarResult(response);
      } else if (response.status === "failed") {
        clearInterval(generationPollTimer);
        generationPollTimer = null;
        activeJobId = "";
        throw new Error(response.error || "Avatar generation failed");
      } else if (response.status === "cancelled") {
        clearInterval(generationPollTimer);
        generationPollTimer = null;
        activeJobId = "";
        stopGenerationProgress("Stopped", 0);
        currentReplyText = "Conversation stopped.";
        const replyBox = document.querySelector("#replyBox");
        if (replyBox) replyBox.textContent = currentReplyText;
      }
    } catch (err) {
      clearInterval(generationPollTimer);
      generationPollTimer = null;
      activeJobId = "";
      stopGenerationProgress("Listening", 0);
      currentReplyText = err.message;
      const replyBox = document.querySelector("#replyBox");
      if (replyBox) replyBox.textContent = err.message;
    }
  };
  await tick();
  generationPollTimer = setInterval(tick, 2000);
}

function applyPartialAvatarResult(response) {
  const nextSubtitle = subtitleTextFromResponse(response);
  const shouldUpdateReply = response.replyText && response.replyText !== currentReplyText;
  const shouldUpdateSubtitle = nextSubtitle && nextSubtitle !== currentSubtitleText;
  if (shouldUpdateReply || shouldUpdateSubtitle) {
    if (response.replyText) {
      currentReplyText = response.replyText;
    }
    currentSubtitleText = nextSubtitle;
    const replyBox = document.querySelector("#replyBox");
    if (replyBox) replyBox.textContent = currentSubtitleText || currentReplyText;
  }
}

async function applyAvatarResult(response) {
  setGenerationProgress(100, "Avatar ready");
  avatarVideoUrl = response.videoUrl || "";
  waitingForAvatarPlayback = Boolean(avatarVideoUrl);
  avatarVideoReadyForListening = !avatarVideoUrl;
  avatarVideoLoadError = false;
  avatarAudioUrl = response.audioUrl || "";
  currentRunId = response.runId || currentRunId;
  avatarRunId = response.runId || response.run_id || currentRunId || avatarRunId;
  activeAvatarView = "video";
  currentReplyText = response.replyText || "The digital human did not return a response yet.";
  currentSubtitleText = subtitleTextFromResponse(response);
  const historyItem = buildHistoryItemFromResponse(response);
  recordedChunks = [];
  recordedVideoBlob = null;
  recordedAudioFile = null;
  clearCurrentRecordingPreview();
  if (isGuestSession()) {
    addGuestConversation(historyItem);
  }
  renderRoom();
  scheduleLocalPreviewRestore();
  await loadConversations();
  requestAnimationFrame(() => {
    const video = document.querySelector("#avatarVideo");
    if (video) {
      video.currentTime = 0;
      pauseAvatarMediaExcept(video);
      video.play().catch(() => {
        video.muted = true;
        video.play().catch(() => {
          waitingForAvatarPlayback = false;
          avatarVideoReadyForListening = true;
          updateListeningPanelDom();
          maybeStartAutoCall();
        });
      });
    } else {
      waitingForAvatarPlayback = false;
    }
  });
  setTimeout(() => {
    stopGenerationProgress("Ready", 0);
    updateListeningPanelDom();
  }, 1200);
}

function buildHistoryItemFromResponse(response) {
  const avatar = avatars.find((item) => item.id === currentAvatarId || item.avatarId === currentAvatarId) || avatars[0];
  const session = getSession() || { id: getGuestId(), username: "Guest" };
  const subtitleText = subtitleTextFromResponse(response);
  return {
    id: response.jobId || response.id || `guest-turn-${Date.now()}`,
    userId: session.id,
    username: session.username || "Guest",
    avatarId: response.avatarId || avatar.avatarId || currentAvatarId,
    avatarName: avatar.name,
    userText: "Recorded video/audio turn",
    replyText: subtitleText || response.replyText || "",
    subtitleText,
    userVideoUrl: response.inputVideoUrl || "",
    videoUrl: response.videoUrl || "",
    audioUrl: response.audioUrl || "",
    combinedVideoUrl: response.combinedVideoUrl || "",
    runId: response.runId || response.run_id || "",
    createdAt: response.createdAt || new Date().toISOString(),
  };
}

function getGuestConversations() {
  return guestConversations;
}

function saveGuestConversations(items) {
  guestConversations = items.slice(0, 80);
}

function addGuestConversation(item) {
  const next = [item, ...getGuestConversations().filter((entry) => entry.id !== item.id)];
  saveGuestConversations(next);
}

function beginGenerationProgress() {
  stopAutoCall();
  clearAutoSubmitDeadline();
  generationActive = true;
  generationProgress = 8;
  generationServerUpdatedAt = 0;
  waitingForAvatarPlayback = false;
  avatarAudioUrl = "";
  generationStatus = "Preparing upload";
  if (generationTimer) clearInterval(generationTimer);
  generationTimer = setInterval(() => {
    if (!generationActive) return;
    const stages = [
      [25, "Analyzing your video and voice"],
      [45, "Planning emotional response"],
      [65, "Synthesizing companion voice"],
      [82, "Driving avatar motion"],
      [94, "Rendering digital human video"],
    ];
    const next = stages.find(([limit]) => generationProgress < limit);
    if (next) {
      generationProgress = Math.min(next[0], generationProgress + 2);
      if (Date.now() - generationServerUpdatedAt > 2500) {
        generationStatus = next[1];
      }
      updateGenerationProgressDom();
    }
  }, 1400);
  updateGenerationProgressDom();
}

function setGenerationProgress(value, status, fromServer = false) {
  generationActive = value < 100;
  generationProgress = generationActive ? Math.max(generationProgress, value) : value;
  if (fromServer) generationServerUpdatedAt = Date.now();
  generationStatus = status;
  updateGenerationProgressDom();
}

function stopGenerationProgress(status = "Ready", value = 0) {
  if (generationTimer) clearInterval(generationTimer);
  if (generationPollTimer) clearInterval(generationPollTimer);
  generationTimer = null;
  generationPollTimer = null;
  generationActive = false;
  generationProgress = value;
  generationServerUpdatedAt = 0;
  generationStatus = status;
  updateGenerationProgressDom();
}

function listeningPanelTitle() {
  if (callStopped) return "Stopped";
  if (waitingForAvatarPlayback) return "Playing response";
  if (avatarVideoLoadError) return "Video unavailable";
  if (avatarVideoUrl && !avatarVideoReadyForListening) return "Loading video";
  return "Listening";
}

function updateListeningPanelDom() {
  const listening = document.querySelector(".listening-panel");
  if (!listening) return;
  listening.hidden = generationActive;
  const title = listening.querySelector("strong");
  const wave = listening.querySelector(".voice-wave");
  const shouldShowWave = !callStopped && !waitingForAvatarPlayback && !avatarVideoLoadError && !(avatarVideoUrl && !avatarVideoReadyForListening);
  if (title) title.textContent = listeningPanelTitle();
  if (wave) wave.hidden = !shouldShowWave;
}

function updateGenerationProgressDom() {
  const wrap = document.querySelector(".generation-progress");
  const listening = document.querySelector(".listening-panel");
  const badge = document.querySelector(".avatar-status-badge");
  if (badge) {
    badge.textContent = generationActive ? "Generating avatar" : "Digital human online";
    badge.classList.toggle("is-generating", generationActive);
  }
  if (listening) {
    listening.hidden = generationActive;
    const title = listening.querySelector("strong");
    const wave = listening.querySelector(".voice-wave");
    if (title) title.textContent = listeningPanelTitle();
    if (wave) wave.hidden = callStopped || waitingForAvatarPlayback || avatarVideoLoadError || (avatarVideoUrl && !avatarVideoReadyForListening);
  }
  if (!wrap) return;
  wrap.hidden = !generationActive;
  const title = wrap.querySelector("strong");
  const percent = wrap.querySelector(".generation-percent");
  const bar = wrap.querySelector(".generation-track span");
  if (title) title.textContent = generationStatus;
  if (percent) percent.textContent = `${Math.round(generationProgress)}%`;
  if (bar) bar.style.width = `${Math.max(4, generationProgress)}%`;
}

async function loadConversations() {
  const historyList = document.querySelector("#historyList");
  if (historyList) {
    historyList.innerHTML = '<div class="history-empty">Loading...</div>';
  }
  if (isGuestSession()) {
    conversations = getGuestConversations().filter((item) => !currentAvatarId || String(item.avatarId) === String(currentAvatarId));
    if (historyList) historyList.innerHTML = renderConversationHistory(conversations);
    bindMediaExclusion();
    return;
  }
  try {
    const payload = await apiFetch(`/api/conversations?avatarId=${encodeURIComponent(currentAvatarId)}`);
    conversations = payload.conversations || [];
    if (historyList) {
      historyList.innerHTML = renderConversationHistory(conversations);
      bindMediaExclusion();
    }
  } catch (err) {
    if (historyList) {
      historyList.innerHTML = `<div class="history-empty">${escapeHtml(err.message)}</div>`;
    }
  }
}

function renderConversationHistory(items) {
  if (!items.length) {
    return '<div class="history-empty">No conversation history yet</div>';
  }
  return items
    .map(
      (item) => `
        <article class="history-item">
          <div class="history-meta">
            <span>${escapeHtml(item.avatarName)}</span>
            <time>${formatDate(item.createdAt)}</time>
          </div>
          <p class="history-user">${escapeHtml(item.username)}: ${escapeHtml(item.userText)}</p>
          <p class="history-reply">${escapeHtml(item.subtitleText || item.replyText)}</p>
          ${renderHistoryVideos(item)}
        </article>
      `
    )
    .join("");
}

function renderHistoryVideos(item) {
  if (!item.combinedVideoUrl) return "";
  return `
    <div class="history-videos">
      <div class="history-video-block">
        <span>Video call recording</span>
        <video class="history-video" src="${escapeHtml(item.combinedVideoUrl)}" controls preload="metadata"></video>
      </div>
    </div>
  `;
}

function historyUserVideoUrl(item) {
  if (item.userVideoUrl) return item.userVideoUrl;
  if (!item.id) return "";
  return `/outputs/3depb_uploads/${encodeURIComponent(item.id)}/input_video.webm`;
}

async function clearHistory() {
  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  if (!confirm(`Clear history and referenced videos for ${avatar.name}?`)) return;
  const button = document.querySelector("[data-action='clear-history']");
  if (button) {
    button.disabled = true;
    button.textContent = "Clearing...";
  }
  try {
    if (isGuestSession()) {
      const remaining = getGuestConversations().filter((item) => String(item.avatarId) !== String(currentAvatarId));
      saveGuestConversations(remaining);
      conversations = [];
      const historyList = document.querySelector("#historyList");
      if (historyList) historyList.innerHTML = renderConversationHistory(conversations);
      return;
    }
    await apiFetch(`/api/history?avatarId=${encodeURIComponent(currentAvatarId)}`, { method: "DELETE" });
    conversations = [];
    recordings = [];
    clearCurrentRecordingPreview();
    const historyList = document.querySelector("#historyList");
    if (historyList) historyList.innerHTML = renderConversationHistory(conversations);
  } catch (err) {
    alert(err.message);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Clear";
    }
  }
}

async function exportHistory() {
  let session = requireSignedIn();
  if (!session) {
    try {
      session = await openAuthModal("Exporting all history requires login or registration.");
    } catch {
      return;
    }
  }
  const items = isGuestSession() ? [] : conversations;
  const exportItems = conversations.length ? conversations : items;
  const button = document.querySelector("[data-action='export-history']");
  if (button) {
    button.disabled = true;
    button.textContent = "Exporting...";
  }
  try {
    const payload = await apiFetch("/api/history/export", {
      method: "POST",
      body: JSON.stringify({
        avatarId: currentAvatarId,
        entries: exportItems,
      }),
    });
    if (payload.url) {
      window.open(payload.url, "_blank", "noopener,noreferrer");
    }
  } catch (err) {
    alert(err.message);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Export All";
    }
  }
}

function avatarBackgroundById(id) {
  return avatarBackgrounds.find((item) => item.id === id) || avatarBackgrounds[0] || DEFAULT_AVATAR_BACKGROUNDS[0];
}

function maybePromptAvatarSelection() {
  if (avatarSelectionPrompted || !avatars.length) return Promise.resolve();
  return openAvatarSelectionModal({ requireSelection: true, showCancel: false });
}

function openAvatarSelectionModal(options = {}) {
  const requireSelection = options.requireSelection !== false;
  const showCancel = options.showCancel !== false;
  return new Promise((resolve, reject) => {
  const existing = document.querySelector("#avatarModal");
  if (existing) existing.remove();
  const wrap = document.createElement("div");
  wrap.id = "avatarModal";
  wrap.className = "modal-backdrop";
  wrap.innerHTML = `
    <section class="modal-card avatar-picker-modal">
      <h2>Choose Avatar</h2>
      <div class="avatar-picker-list">
        ${avatars
          .map(
            (item) => `
              <div class="avatar-option ${item.id === currentAvatarId ? "is-active" : ""}" role="button" tabindex="0" data-pick-avatar="${escapeHtml(item.id)}">
                ${avatarSwatchMarkup(item)}
                <span class="avatar-meta">
                  <strong>${escapeHtml(item.name)}</strong>
                  <span>${escapeHtml(avatarDisplayDescription(item))}</span>
                </span>
                <button class="secondary-btn avatar-preview-btn" type="button" data-preview-speaker="${escapeHtml(item.ttsSpeakerId || "6224")}">Preview voice</button>
              </div>
            `
          )
          .join("")}
      </div>
      ${showCancel ? '<div class="avatar-picker-actions"><button class="secondary-btn" type="button" data-modal-close>Cancel</button></div>' : ""}
    </section>
  `;
  document.body.appendChild(wrap);
  avatarSelectionPrompted = true;
  const finish = (avatarId) => {
    currentAvatarId = avatarId;
    wrap.remove();
    resolve(avatarId);
  };
  wrap.querySelectorAll("[data-pick-avatar]").forEach((button) => {
    button.addEventListener("click", () => {
      finish(button.dataset.pickAvatar);
    });
    button.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      finish(button.dataset.pickAvatar);
    });
  });
  const closeButton = wrap.querySelector("[data-modal-close]");
  if (closeButton) {
    closeButton.addEventListener("click", () => {
      wrap.remove();
      if (!requireSelection) {
        resolve(currentAvatarId);
        return;
      }
      reject(new Error("Avatar selection cancelled"));
    });
  }
  const previewButtons = Array.from(wrap.querySelectorAll("[data-preview-speaker]"));
  previewButtons.forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      previewAvatarVoice(button).catch((error) => {
        button.textContent = "Preview failed";
        button.title = error.message || String(error);
        button.disabled = false;
        console.warn("[3depb] voice preview failed", error);
      });
    });
  });
  warmAvatarVoicePreviews(previewButtons.map((button) => button.dataset.previewSpeaker || "6224"));
  });
}

async function previewAvatarVoice(button) {
  const speakerId = normalizePreviewSpeakerId(button.dataset.previewSpeaker || "6224");
  document.querySelectorAll(".avatar-preview-btn").forEach((item) => {
    if (item !== button) {
      item.disabled = false;
      item.textContent = "Preview voice";
    }
  });
  button.disabled = true;
  button.textContent = "Loading...";
  button.title = "";
  try {
    const payload = await getAvatarVoicePreview(speakerId);
    if (!payload.audioUrl) throw new Error("No preview audio returned");
    if (avatarPreviewAudio) {
      avatarPreviewAudio.pause();
      avatarPreviewAudio.currentTime = 0;
    }
    avatarPreviewAudio = payload.audio || new Audio(payload.audioUrl);
    avatarPreviewAudio.currentTime = 0;
    avatarPreviewAudio.onended = () => {
      button.textContent = "Preview voice";
      button.disabled = false;
    };
    await avatarPreviewAudio.play();
    button.textContent = "Playing...";
  } catch (error) {
    button.textContent = "Preview voice";
    throw error;
  } finally {
    if (!avatarPreviewAudio || avatarPreviewAudio.paused) {
      button.disabled = false;
    }
  }
}

function normalizePreviewSpeakerId(speakerId) {
  return String(speakerId || "6224").trim() || "6224";
}

async function getAvatarVoicePreview(speakerId) {
  const key = normalizePreviewSpeakerId(speakerId);
  const cached = avatarVoicePreviewCache.get(key);
  if (cached) return cached;
  const existingRequest = avatarVoicePreviewRequests.get(key);
  if (existingRequest) return existingRequest;

  const request = apiFetch(`/api/tts_preview?speakerId=${encodeURIComponent(key)}`).then((payload) => {
    if (!payload.audioUrl) throw new Error("No preview audio returned");
    const audio = new Audio(payload.audioUrl);
    audio.preload = "auto";
    audio.load();
    const cachedPayload = { ...payload, audio };
    avatarVoicePreviewCache.set(key, cachedPayload);
    return cachedPayload;
  });
  avatarVoicePreviewRequests.set(key, request);
  try {
    return await request;
  } finally {
    avatarVoicePreviewRequests.delete(key);
  }
}

function warmAvatarVoicePreviews(speakerIds) {
  window.clearTimeout(avatarVoicePreviewPreloadTimer);
  const uniqueSpeakerIds = [...new Set(speakerIds.map(normalizePreviewSpeakerId))].slice(0, 6);
  avatarVoicePreviewPreloadTimer = window.setTimeout(async () => {
    const queue = [...uniqueSpeakerIds];
    const preloadOne = async () => {
      while (queue.length) {
        const speakerId = queue.shift();
        try {
          await getAvatarVoicePreview(speakerId);
        } catch (error) {
          console.warn("[3depb] voice preview preload failed", speakerId, error);
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(2, queue.length) }, preloadOne));
  }, 300);
}

async function loadRecordings() {
  const recordingsList = document.querySelector("#recordingsList");
  if (recordingsList) {
    recordingsList.innerHTML = renderRecordings(recordings);
  }
}

function updateCurrentRecordingPreview() {
  clearCurrentRecordingPreview();
  if (!recordedChunks.length) {
    recordings = [];
    return;
  }
  const blob = recordedVideoBlob || new Blob(recordedChunks, { type: recordedMimeType || "video/webm" });
  currentRecordingPreviewUrl = URL.createObjectURL(blob);
  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  const session = getSession() || { username: "You" };
  recordings = [
    {
      id: "current-recording",
      avatarName: avatar.name,
      username: session.username,
      sizeBytes: blob.size,
      createdAt: new Date().toISOString(),
      url: currentRecordingPreviewUrl,
    },
  ];
}

function clearCurrentRecordingPreview() {
  if (currentRecordingPreviewUrl) {
    URL.revokeObjectURL(currentRecordingPreviewUrl);
    currentRecordingPreviewUrl = "";
  }
  recordings = [];
}

function renderRecordings(items) {
  if (!items.length) {
    return '<div class="history-empty">No current recording yet</div>';
  }
  return items.slice(0, 1)
    .map(
      (item) => `
        <article class="recording-item">
          <div class="history-meta">
            <span>${escapeHtml(item.avatarName)}</span>
            <time>${formatDate(item.createdAt)}</time>
          </div>
          <p class="history-user">${escapeHtml(item.username)} · ${formatBytes(item.sizeBytes)}</p>
          <video class="recording-player" src="${escapeHtml(item.url)}" controls preload="metadata"></video>
        </article>
      `
    )
    .join("");
}

function formatBytes(value) {
  const bytes = Number(value) || 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function renderAdmin() {
  const session = requireAuth();
  if (!session) return;
  if (session.role !== "admin") {
    navigate("#/room");
    return;
  }

  let users = [];
  try {
    ({ users } = await apiFetch("/api/users"));
  } catch (err) {
    alert(err.message);
    navigate("#/room");
    return;
  }
  document.querySelector("#app").innerHTML = `
    <main class="page">
      ${topbar("User Account Management", session)}
      <section class="admin-page-stack">
        <section class="admin-account-layout">
          <form class="admin-form" id="userForm">
            <h2>New User</h2>
            <div class="field">
              <label for="newUsername">Username</label>
              <input id="newUsername" name="username" required />
            </div>
            <div class="field">
              <label for="newPassword">Password</label>
              <input id="newPassword" name="password" type="password" required />
            </div>
            <div class="field">
              <label for="newRole">Role</label>
              <select id="newRole" name="role">
                <option value="user">Standard User</option>
                <option value="admin">Administrator</option>
              </select>
            </div>
            <button class="primary-btn" type="submit">Create Account</button>
            <p class="error" id="adminError"></p>
            <p class="hint">Account data is managed by the backend SQLite database. Passwords are stored with PBKDF2 hashes.</p>
          </form>
          <article class="admin-panel">
            <div class="pane-head">
              <div class="pane-title">
                <strong>All Users</strong>
                <span>${users.length} account${users.length === 1 ? "" : "s"}</span>
              </div>
            </div>
            <div class="table-wrap users-table-wrap">
              ${renderUsersTable(users)}
            </div>
          </article>
        </section>
      </section>
    </main>
  `;

  bindTopbar();
  bindUserAdmin();
}

async function renderDigitalHumansAdmin() {
  const session = requireAuth();
  if (!session) return;
  if (session.role !== "admin") {
    navigate("#/room");
    return;
  }

  let digitalHumans = [];
  let backgrounds = [];
  let speakers = [];
  let speakerLoadError = "";
  try {
    const payload = await apiFetch("/api/digital_humans");
    digitalHumans = payload.digitalHumans || [];
    const backgroundPayload = await apiFetch("/api/backgrounds").catch(() => ({ backgrounds: avatarBackgrounds }));
    backgrounds = normalizeBoothBackgrounds(backgroundPayload.backgrounds || []);
  } catch (err) {
    digitalHumans = avatars;
    backgrounds = normalizeBoothBackgrounds(avatarBackgrounds);
    speakerLoadError = err.message;
  }
  try {
    ({ speakers } = await apiFetch("/api/tts_speakers"));
  } catch (err) {
    speakers = [];
    speakerLoadError = err.message;
  }

  document.querySelector("#app").innerHTML = `
    <main class="page">
      ${topbar("Digital Humans Management", session)}
      <section class="admin-page-stack">
        <article class="admin-panel background-management">
          <div class="pane-head">
            <div class="pane-title">
              <strong>Background Management</strong>
              <span>${backgrounds.length} room background${backgrounds.length === 1 ? "" : "s"}</span>
            </div>
          </div>
          ${renderBackgroundManagement(backgrounds)}
        </article>
        <article class="admin-panel digital-human-management">
          <div class="pane-head">
            <div class="pane-title">
              <strong>Digital Humans Management</strong>
              <span>${digitalHumans.length} registered avatar${digitalHumans.length === 1 ? "" : "s"}</span>
            </div>
          </div>
          <div class="table-wrap digital-humans-table-wrap">
            ${speakerLoadError ? `<div class="empty">${escapeHtml(speakerLoadError)}</div>` : renderDigitalHumansTable(digitalHumans, speakers)}
          </div>
        </article>
      </section>
    </main>
  `;

  bindTopbar();
  bindDigitalHumansAdmin();
  bindBackgroundAdmin();
}

function renderBackgroundManagement(items) {
  const rows = (items.length ? items : DEFAULT_AVATAR_BACKGROUNDS)
    .map(
      (item) => `
        <div class="background-admin-row" data-background-row="${escapeHtml(item.id)}">
          ${adminBackgroundPreviewMarkup(item)}
          <input class="table-input" data-background-label="${escapeHtml(item.id)}" value="${escapeHtml(item.label)}" />
          <input class="table-file" type="file" accept="image/png,image/jpeg,image/webp" data-background-image="${escapeHtml(item.id)}" />
          <button class="secondary-btn compact-btn" type="button" data-background-save="${escapeHtml(item.id)}">Save</button>
          <button class="danger-btn compact-btn" type="button" data-background-delete="${escapeHtml(item.id)}" ${DEFAULT_AVATAR_BACKGROUNDS.some((bg) => bg.id === item.id) ? "disabled" : ""}>Delete</button>
        </div>
      `
    )
    .join("");
  return `
    <div class="background-admin-list">${rows}</div>
    <form class="background-admin-create" id="backgroundCreateForm">
      <input class="table-input" name="label" maxlength="32" placeholder="New background name" required />
      <input class="table-file" name="image" type="file" accept="image/png,image/jpeg,image/webp" />
      <button class="primary-btn compact-btn" type="submit">Add Background</button>
    </form>
  `;
}

function renderDigitalHumansTable(items, speakers) {
  if (!items.length) return '<div class="empty">No registered digital humans yet</div>';
  const speakerOptions = (selectedId) =>
    speakers
      .map(
        (speaker) => `
          <option value="${escapeHtml(speaker.id)}" ${String(speaker.id) === String(selectedId) ? "selected" : ""}>
            ${escapeHtml(speaker.label || speaker.id)}
          </option>
        `
      )
      .join("");

  return `
    <table>
      <thead>
        <tr>
          <th>Avatar</th>
          <th>Name</th>
          <th>Description</th>
          <th>Voice</th>
          <th>Speaker</th>
          <th>Image</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map(
            (item) => `
              <tr>
                <td>
                  ${adminAvatarPreviewMarkup(item)}
                  <div class="table-subtext">${escapeHtml(item.id)}</div>
                </td>
                <td>
                  <input class="table-input" data-human-name="${escapeHtml(item.id)}" value="${escapeHtml(item.name)}" />
                </td>
                <td>
                  <input class="table-input" data-human-role="${escapeHtml(item.id)}" value="${escapeHtml(item.role)}" />
                </td>
                <td><span class="badge">${escapeHtml(item.ttsSpeakerId)}</span></td>
                <td>
                  <select class="table-select" data-human-speaker="${escapeHtml(item.id)}">
                    ${speakerOptions(item.ttsSpeakerId)}
                  </select>
                </td>
                <td>
                  <input class="table-file" type="file" accept="image/png,image/jpeg,image/webp,image/gif" data-human-image="${escapeHtml(item.id)}" />
                </td>
                <td>
                  <button class="secondary-btn compact-btn" data-human-save="${escapeHtml(item.id)}">Save</button>
                </td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderUsersTable(users) {
  if (!users.length) return '<div class="empty">No users yet</div>';

  return `
    <table>
      <thead>
        <tr>
          <th>Username</th>
          <th>Role</th>
          <th>Status</th>
          <th>Last Login</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${users
          .map(
            (user) => `
              <tr>
                <td>${escapeHtml(user.username)}</td>
                <td><span class="badge">${user.role === "admin" ? "Administrator" : "Standard User"}</span></td>
                <td><span class="badge ${user.status === "active" ? "" : "off"}">${user.status === "active" ? "Active" : "Disabled"}</span></td>
                <td>${user.lastLoginAt ? formatDate(user.lastLoginAt) : "Never"}</td>
                <td>
                  <div class="row-actions">
                    <button class="secondary-btn" data-user-action="reset" data-user-id="${user.id}">Reset Password</button>
                    <button class="secondary-btn" data-user-action="status" data-user-id="${user.id}">${user.status === "active" ? "Disable" : "Enable"}</button>
                    <button class="danger-btn" data-user-action="delete" data-user-id="${user.id}">Delete</button>
                  </div>
                </td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function bindUserAdmin() {
  document.querySelector("#userForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = String(form.get("username")).trim();
    const password = String(form.get("password")).trim();
    const role = String(form.get("role"));
    const error = document.querySelector("#adminError");
    const button = event.currentTarget.querySelector("button[type='submit']");

    error.textContent = "";
    button.disabled = true;
    button.textContent = "Creating...";
    try {
      await apiFetch("/api/users", {
        method: "POST",
        body: JSON.stringify({ username, password, role })
      });
      await renderAdmin();
    } catch (err) {
      error.textContent = err.message;
    } finally {
      button.disabled = false;
      button.textContent = "Create Account";
    }
  });

  document.querySelectorAll("[data-user-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.userId;
      const action = button.dataset.userAction;
      const session = getSession();
      try {
        if (action === "delete") {
          if (id === session.id) {
            alert("You cannot delete the currently signed-in administrator account.");
            return;
          }
          await apiFetch(`/api/users/${id}`, { method: "DELETE" });
        }

        if (action === "status") {
          if (id === session.id && button.textContent === "Disable") {
            alert("You cannot disable the currently signed-in administrator account.");
            return;
          }
          await apiFetch(`/api/users/${id}/status`, {
            method: "PATCH",
            body: JSON.stringify({ status: button.textContent === "Disable" ? "disabled" : "active" })
          });
        }

        if (action === "reset") {
          const nextPassword = prompt("Enter a new password");
          if (!nextPassword) return;
          await apiFetch(`/api/users/${id}/password`, {
            method: "PATCH",
            body: JSON.stringify({ password: nextPassword })
          });
        }

        await renderAdmin();
      } catch (err) {
        alert(err.message);
      }
    });
  });
}

function bindDigitalHumansAdmin() {
  document.querySelectorAll("[data-human-image]").forEach((input) => {
    input.addEventListener("change", () => {
      const avatarId = input.dataset.humanImage;
      if (!input.files || !input.files[0]) return;
      const previous = pendingDigitalHumanImages.get(avatarId);
      if (previous?.previewUrl) URL.revokeObjectURL(previous.previewUrl);
      const localPreviewUrl = URL.createObjectURL(input.files[0]);
      pendingDigitalHumanImages.set(avatarId, { file: input.files[0], previewUrl: localPreviewUrl });
      const preview = input.closest("tr")?.querySelector(".admin-avatar-preview");
      if (preview) {
        preview.innerHTML = `<img src="${escapeHtml(localPreviewUrl)}" alt="Selected avatar preview" />`;
      }
    });
  });

  document.querySelectorAll("[data-human-save]").forEach((button) => {
    button.addEventListener("click", async () => {
      const avatarId = button.dataset.humanSave;
      const nameInput = document.querySelector(`[data-human-name="${cssEscape(avatarId)}"]`);
      const roleInput = document.querySelector(`[data-human-role="${cssEscape(avatarId)}"]`);
      const name = nameInput ? nameInput.value.trim() : "";
      const role = roleInput ? roleInput.value.trim() : "";
      const speaker = document.querySelector(`[data-human-speaker="${cssEscape(avatarId)}"]`);
      const image = document.querySelector(`[data-human-image="${cssEscape(avatarId)}"]`);
      const pendingImage = pendingDigitalHumanImages.get(avatarId);
      const ttsSpeakerId = (speaker && speaker.value) || "6224";
      button.disabled = true;
      button.textContent = "Saving...";
      try {
        let latestPayload = await apiFetch(`/api/digital_humans/${encodeURIComponent(avatarId)}`, {
          method: "PATCH",
          body: JSON.stringify({ name, role, ttsSpeakerId })
        });
        applyDigitalHumansPayload(latestPayload);
        if (pendingImage?.file) {
          const payload = await uploadDigitalHumanImage(avatarId, pendingImage.file);
          latestPayload = payload;
          if (payload.imageUrl) {
            updateAvatarImageInMemory(avatarId, payload.imageUrl);
          }
          applyDigitalHumansPayload(payload);
          if (pendingImage.previewUrl) URL.revokeObjectURL(pendingImage.previewUrl);
          pendingDigitalHumanImages.delete(avatarId);
          if (image) image.value = "";
        }
        if (!Array.isArray(latestPayload?.digitalHumans)) {
          digitalHumansLoaded = false;
          await loadDigitalHumans();
        }
        await renderDigitalHumansAdmin();
      } catch (err) {
        alert(err.message);
        await renderDigitalHumansAdmin();
      }
    });
  });
}

function bindBackgroundAdmin() {
  document.querySelectorAll("[data-background-image]").forEach((input) => {
    input.addEventListener("change", () => {
      const backgroundId = input.dataset.backgroundImage;
      if (!input.files || !input.files[0]) return;
      const previous = pendingBoothBackgroundImages.get(backgroundId);
      if (previous?.previewUrl) URL.revokeObjectURL(previous.previewUrl);
      const localPreviewUrl = URL.createObjectURL(input.files[0]);
      pendingBoothBackgroundImages.set(backgroundId, { file: input.files[0], previewUrl: localPreviewUrl });
      const preview = input.closest("[data-background-row]")?.querySelector(".admin-background-preview");
      if (preview) {
        preview.classList.add("has-image");
        preview.innerHTML = `<img src="${escapeHtml(localPreviewUrl)}" alt="Selected background preview" />`;
      }
    });
  });

  document.querySelectorAll("[data-background-save]").forEach((button) => {
    button.addEventListener("click", async () => {
      const backgroundId = button.dataset.backgroundSave;
      const labelInput = document.querySelector(`[data-background-label="${cssEscape(backgroundId)}"]`);
      const pending = pendingBoothBackgroundImages.get(backgroundId);
      button.disabled = true;
      button.textContent = "Saving...";
      try {
        const profilePayload = await apiFetch(`/api/backgrounds/${encodeURIComponent(backgroundId)}`, {
          method: "PATCH",
          body: JSON.stringify({ label: labelInput ? labelInput.value.trim() : "" }),
        });
        applyBoothBackgroundsPayload(profilePayload);
        if (pending?.file) {
          const uploadPayload = await uploadBoothBackground(backgroundId, pending.file);
          applyBoothBackgroundsPayload(uploadPayload);
          if (pending.previewUrl) URL.revokeObjectURL(pending.previewUrl);
          pendingBoothBackgroundImages.delete(backgroundId);
        }
        digitalHumansLoaded = false;
        await loadDigitalHumans();
        await renderDigitalHumansAdmin();
      } catch (err) {
        alert(err.message);
        button.disabled = false;
        button.textContent = "Save";
      }
    });
  });

  document.querySelectorAll("[data-background-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      const backgroundId = button.dataset.backgroundDelete;
      if (!confirm("Delete this background?")) return;
      button.disabled = true;
      try {
        await apiFetch(`/api/backgrounds/${encodeURIComponent(backgroundId)}`, { method: "DELETE" });
        if (currentAvatarBackgroundId === backgroundId) {
          currentAvatarBackgroundId = "study";
        }
        digitalHumansLoaded = false;
        await loadDigitalHumans();
        await renderDigitalHumansAdmin();
      } catch (err) {
        alert(err.message);
        button.disabled = false;
      }
    });
  });

  const createForm = document.querySelector("#backgroundCreateForm");
  if (createForm) {
    createForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(createForm);
      const button = createForm.querySelector("button[type='submit']");
      if (button) {
        button.disabled = true;
        button.textContent = "Adding...";
      }
      try {
        const payload = await apiFetch("/api/backgrounds", {
          method: "POST",
          body: JSON.stringify({ label: String(data.get("label") || "").trim() }),
        });
        const backgroundId = payload.background?.id;
        applyBoothBackgroundsPayload(payload);
        const image = data.get("image");
        if (backgroundId && image instanceof File && image.size > 0) {
          const uploadPayload = await uploadBoothBackground(backgroundId, image);
          applyBoothBackgroundsPayload(uploadPayload);
        }
        digitalHumansLoaded = false;
        await loadDigitalHumans();
        await renderDigitalHumansAdmin();
      } catch (err) {
        alert(err.message);
        if (button) {
          button.disabled = false;
          button.textContent = "Add Background";
        }
      }
    });
  }
}

async function uploadDigitalHumanImage(avatarId, file) {
  const form = new FormData();
  form.append("image", file, file.name);
  const response = await fetch(`/api/digital_humans/${encodeURIComponent(avatarId)}/image`, {
    method: "POST",
    credentials: "same-origin",
    body: form
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || "Image upload failed");
  return payload;
}

async function uploadBoothBackground(backgroundId, file) {
  const form = new FormData();
  form.append("background", file, file.name);
  return apiFetch(`/api/backgrounds/${encodeURIComponent(backgroundId)}/image`, {
    method: "POST",
    body: form
  });
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function looksChinese(value) {
  return /[\u4e00-\u9fff]/.test(String(value || ""));
}

function subtitleTextFromResponse(response) {
  const subtitle = response?.subtitleText || response?.ttsText || response?.spokenText || response?.englishTtsText || "";
  if (subtitle && !looksChinese(subtitle)) return subtitle;
  const reply = response?.replyText || "";
  if (reply && !looksChinese(reply)) return reply;
  return "";
}

function cssEscape(value) {
  if (window.CSS && window.CSS.escape) return window.CSS.escape(String(value));
  return String(value).replace(/["\\]/g, "\\$&");
}

async function render() {
  const route = location.hash || "#/";
  if (route === "#/room" || route === "#/history" || route === "#/admin" || route === "#/digital-humans") {
    await hydrateSession();
  }
  if (route === "#/room") {
    await loadDigitalHumans();
    if (!getSession()) {
      renderLogin();
      return;
    }
    if (!avatarSelectionPrompted) {
      renderPreCallSetup();
      await maybePromptAvatarSelection();
    }
    renderRoom();
    return;
  }
  if (route === "#/history") {
    await loadDigitalHumans();
    renderHistoryPage();
    return;
  }
  if (route === "#/admin") {
    await renderAdmin();
    return;
  }
  if (route === "#/digital-humans") {
    await renderDigitalHumansAdmin();
    return;
  }
  renderLogin();
}

function renderPreCallSetup() {
  const session = requireAuth();
  if (!session) return;
  document.querySelector("#app").innerHTML = `
    <main class="page login-page">
      <section class="login-shell">
        <div class="brand-stage">
          <div class="booth-visual" aria-hidden="true"></div>
          <div class="brand-copy">
            <h1 class="brand-title">Choose Digital Human</h1>
            <p class="brand-subtitle">Hi, ${escapeHtml(session.username)}. Select a digital human to start the call.</p>
          </div>
        </div>
      </section>
    </main>
  `;
}

window.addEventListener("hashchange", () => {
  render().catch(showBootError);
});
window.addEventListener("beforeunload", stopCamera);
render().catch(showBootError);

function showBootError(error) {
  const app = document.querySelector("#app");
  if (!app) return;
  app.innerHTML = `
    <main class="page login-page">
      <section class="login-shell">
        <form class="auth-card">
          <h2>Interface failed to load</h2>
          <p class="error">${escapeHtml(error && error.message ? error.message : error || "Unknown browser error")}</p>
        </form>
      </section>
    </main>
  `;
}
