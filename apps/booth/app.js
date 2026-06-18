import * as THREE from "./vendor/three.module.js?v=3depb-render-drag-20260614-03";
import { OrbitControls } from "./vendor/OrbitControls.js?v=3depb-render-drag-20260614-03";
import { PLYLoader } from "./vendor/PLYLoader.js?v=3depb-render-drag-20260614-03";

const SESSION_KEY = "3depb_session";

let avatars = [
  {
    id: "306",
    avatarId: "306",
    ttsSpeakerId: "6224",
    name: "Avatar 306",
    role: "Avatar 306 · Voice 6224",
    speakerLabel: "6224 · voice",
    color: "#32d0a4",
    reply: "I hear what you are feeling. We can slow down and spend a little more time with the most important part."
  }
];
let digitalHumansLoaded = false;

let cameraStream = null;
let recorder = null;
let recordedChunks = [];
let recordedMimeType = "video/webm";
let recordedAudioFile = null;
let recordingStopPromise = null;
let recordingStopResolve = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let monitorGain = null;
let recordingBuffers = [];
let recordingLength = 0;
let currentAvatarId = "306";
let cameraOn = true;
let micOn = true;
let isRecording = false;
let avatarVideoUrl = "";
let avatarAudioUrl = "";
let avatarRunId = "";
let activeAvatarView = "video";
let currentReplyText = "";
let conversations = [];
let recordings = [];
let currentRecordingPreviewUrl = "";
let generationActive = false;
let generationProgress = 0;
let generationStatus = "Ready";
let generationTimer = null;
let renderPreviewTimer = null;
let renderPreviewLoopTimer = null;
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

function getSession() {
  const saved = sessionStorage.getItem(SESSION_KEY);
  return saved ? JSON.parse(saved) : null;
}

function setSession(user) {
  sessionStorage.setItem(
    SESSION_KEY,
    JSON.stringify({ id: user.id, username: user.username, role: user.role })
  );
}

function clearSession() {
  sessionStorage.removeItem(SESSION_KEY);
}

async function hydrateSession() {
  if (getSession()) return getSession();
  try {
    const { user } = await apiFetch("/api/auth/me");
    setSession(user);
    return user;
  } catch {
    return null;
  }
}

async function apiFetch(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: isFormData
      ? { ...(options.headers || {}) }
      : {
          "Content-Type": "application/json",
          ...(options.headers || {})
        },
    ...options
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

async function loadDigitalHumans() {
  if (digitalHumansLoaded) return;
  try {
    const payload = await apiFetch("/api/digital_humans");
    const items = Array.isArray(payload.digitalHumans) ? payload.digitalHumans : [];
    if (items.length) {
      avatars = items.map((item, index) => ({
        id: String(item.id || item.avatarId),
        avatarId: String(item.avatarId || item.id),
        ttsSpeakerId: String(item.ttsSpeakerId || "6224"),
        name: String(item.name || item.label || `Avatar ${item.avatarId || item.id}`),
        role: String(item.role || `Avatar ${item.avatarId || item.id} · Voice ${item.ttsSpeakerId || "6224"}`),
        speakerLabel: String(item.speakerLabel || `Voice ${item.ttsSpeakerId || "6224"}`),
        color: String(item.color || ["#32d0a4", "#6fb7ff", "#ffb84d", "#f0798d"][index % 4]),
        imageUrl: String(item.imageUrl || ""),
        reply: String(item.reply || "I hear what you are feeling. We can slow down and spend a little more time with the most important part."),
      }));
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
  return `${color} background-image: url("${escapeHtml(item.imageUrl)}");`;
}

function requireAuth() {
  const session = getSession();
  if (!session) {
    navigate("#/");
    return null;
  }
  return session;
}

function topbar(title, session) {
  const adminButton =
    session.role === "admin"
      ? `
        <button class="secondary-btn" data-action="digital-humans-admin">Digital Humans Management</button>
        <button class="secondary-btn" data-action="admin">User Management</button>
      `
      : "";
  return `
    <header class="topbar">
      <div class="topbar-title"><span class="status-dot"></span>${escapeHtml(title)}</div>
      <div class="topbar-actions">
        <span class="badge">${escapeHtml(session.username)}</span>
        ${adminButton}
        <button class="secondary-btn" data-action="room">Conversation</button>
        <button class="danger-btn" data-action="logout">Log Out</button>
      </div>
    </header>
  `;
}

function renderLogin(mode = "login") {
  stopCamera();
  const isRegister = mode === "register";
  document.querySelector("#app").innerHTML = `
    <main class="page login-page">
      <section class="login-shell">
        <div class="brand-stage">
          <div class="booth-visual" aria-hidden="true"></div>
          <div class="brand-copy">
            <h1 class="brand-title">3D Emotional Phone Booth</h1>
            <p class="brand-subtitle">Have immersive emotional conversations with your chosen digital human, then record, review, and manage every session.</p>
          </div>
        </div>
        <form class="auth-card" id="loginForm">
          <h2>${isRegister ? "Register" : "Log In"}</h2>
          <div class="field">
            <label for="username">Username</label>
            <input id="username" name="username" autocomplete="username" placeholder="Enter username" required />
          </div>
          <div class="field">
            <label for="password">Password</label>
            <input id="password" name="password" type="password" autocomplete="${isRegister ? "new-password" : "current-password"}" placeholder="Enter password" required />
          </div>
          <button class="primary-btn" type="submit">${isRegister ? "Register" : "Log In"}</button>
          <button class="secondary-btn auth-switch-btn" type="button" data-auth-mode="${isRegister ? "login" : "register"}">
            ${isRegister ? "Back to Log In" : "Create an Account"}
          </button>
          <p class="error" id="loginError"></p>
        </form>
      </section>
    </main>
  `;

  document.querySelector("[data-auth-mode]").addEventListener("click", (event) => {
    renderLogin(event.currentTarget.dataset.authMode);
  });

  document.querySelector("#loginForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = String(form.get("username")).trim();
    const password = String(form.get("password"));
    const error = document.querySelector("#loginError");
    const button = event.currentTarget.querySelector("button[type='submit']");

    error.textContent = "";
    button.disabled = true;
    button.textContent = isRegister ? "Registering..." : "Logging in...";
    try {
      const { user } = await apiFetch(isRegister ? "/api/auth/register" : "/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password })
      });
      setSession(user);
      navigate("#/room");
    } catch (err) {
      error.textContent =
        isRegister && err.message === "Not found"
          ? "Registration service is not loaded. Refresh this page and try again."
          : err.message;
    } finally {
      button.disabled = false;
      button.textContent = isRegister ? "Register" : "Log In";
    }
  });
}

function renderRoom() {
  const session = requireAuth();
  if (!session) return;

  const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
  document.querySelector("#app").innerHTML = `
    <main class="page">
      ${topbar("3D Emotional Phone Booth", session)}
      <section class="room">
        <article class="video-pane">
          <div class="pane-head">
            <div class="pane-title">
              <strong>My Video</strong>
              <span>${micOn ? "Microphone on" : "Microphone off"} · ${cameraOn ? "Camera on" : "Camera off"}</span>
            </div>
            ${isRecording ? '<span class="recording-badge">Recording</span>' : ""}
          </div>
          <div class="video-frame">
            <video id="localVideo" autoplay muted playsinline ${cameraStream && cameraOn ? "" : "hidden"}></video>
            <div class="camera-placeholder" id="cameraPlaceholder" ${cameraStream && cameraOn ? "hidden" : ""}>
              <div>
                <h2>${escapeHtml(session.username)}</h2>
                <p class="hint">${cameraOn ? "Requesting camera and microphone access" : "Camera is off"}</p>
              </div>
            </div>
          </div>
        </article>

        <article class="video-pane">
          <div class="pane-head">
            <div class="pane-title">
              <strong>${avatar.name}</strong>
              <span>${avatar.role} · ${avatar.speakerLabel || `Voice ${avatar.ttsSpeakerId || ""}`} · Avatar system pipeline</span>
            </div>
            <div class="topbar-actions">
              <div class="avatar-view-tabs" role="tablist" aria-label="Avatar view">
                <button class="${activeAvatarView === "video" ? "active" : ""}" type="button" data-avatar-view="video" aria-selected="${activeAvatarView === "video"}">Video</button>
                <button class="${activeAvatarView === "rendered" ? "active" : ""}" type="button" data-avatar-view="rendered" aria-selected="${activeAvatarView === "rendered"}">3D Render</button>
              </div>
              <span class="badge ${generationActive ? "is-generating" : ""}">${generationActive ? "Generating avatar" : "Digital human online"}</span>
            </div>
          </div>
          <div class="video-frame">
            <div class="avatar-stage" style="--avatar-color: ${avatar.color}">
              ${
                avatarVideoUrl
                  ? `<video id="avatarVideo" class="avatar-video" src="${escapeHtml(avatarVideoUrl)}" autoplay playsinline controls ${activeAvatarView === "video" ? "" : "hidden"}></video>`
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
              <div class="avatar-caption" ${activeAvatarView === "rendered" ? "hidden" : ""}>
                <div class="reply-box" id="replyBox">${escapeHtml(currentReplyText || avatar.reply)}</div>
              </div>
              <div class="generation-progress" ${generationActive ? "" : "hidden"}>
                <div class="generation-progress-head">
                  <strong>${escapeHtml(generationStatus)}</strong>
                  <span class="generation-percent">${Math.round(generationProgress)}%</span>
                </div>
                <div class="generation-track">
                  <span style="width: ${Math.max(4, generationProgress)}%"></span>
                </div>
              </div>
            </div>
            <div class="toolbar" id="roomToolbar" ${activeAvatarView === "rendered" ? "hidden" : ""}>
              <button class="icon-btn ${micOn ? "" : "is-off"}" title="Toggle audio" data-action="toggle-mic">${micOn ? "Mic" : "Off"}</button>
              <button class="icon-btn ${cameraOn ? "" : "is-off"}" title="Toggle video" data-action="toggle-camera">${cameraOn ? "Cam" : "Off"}</button>
              <button class="icon-btn ${isRecording ? "is-off" : ""}" title="Toggle recording" data-action="toggle-record">${isRecording ? "Stop" : "Rec"}</button>
              <button class="danger-btn" data-action="end-call">End</button>
            </div>
          </div>
        </article>

        <aside class="avatar-sidebar">
          <div class="pane-head">
            <div class="pane-title">
              <strong>Digital Human</strong>
              <span>Choose who you want to talk with</span>
            </div>
          </div>
          <div class="avatar-list">
            ${avatars
              .map(
                (item) => `
                  <button class="avatar-option ${item.id === avatar.id ? "is-active" : ""}" data-avatar="${item.id}">
                    <span class="avatar-swatch" style="${avatarSwatchStyle(item)}"></span>
                    <span class="avatar-meta">
                      <strong>${escapeHtml(item.name)}</strong>
                      <span>${escapeHtml(item.role)} · ${escapeHtml(item.speakerLabel || `Voice ${item.ttsSpeakerId || ""}`)}</span>
                    </span>
                  </button>
                `
              )
              .join("")}
          </div>
          <section class="history-panel">
            <div class="history-head">
              <strong>History</strong>
              <div class="history-actions">
                <button class="secondary-btn compact-btn" data-action="refresh-history">Refresh</button>
                <button class="danger-btn compact-btn" data-action="clear-history">Clear</button>
              </div>
            </div>
            <div class="history-list" id="historyList">
              ${renderConversationHistory(conversations)}
            </div>
          </section>
          <section class="recordings-panel">
            <div class="history-head">
              <strong>Recordings</strong>
              <button class="secondary-btn compact-btn" data-action="refresh-recordings">Refresh</button>
            </div>
            <div class="recordings-list" id="recordingsList">
              ${renderRecordings(recordings)}
            </div>
          </section>
          <div class="talk-input">
            <textarea id="talkText" placeholder="Record your video and audio, then send it to generate an avatar reply."></textarea>
            <button class="primary-btn" data-action="send-talk">Send to Digital Human</button>
            <audio id="avatarAudio" ${avatarAudioUrl && !avatarVideoUrl ? `src="${escapeHtml(avatarAudioUrl)}" autoplay` : ""}></audio>
          </div>
        </aside>
      </section>
    </main>
  `;

  bindTopbar();
  bindRoom();
  bindMediaExclusion();
  startCamera();
}

function bindTopbar() {
  document.querySelectorAll("[data-action='logout']").forEach((button) => {
    button.addEventListener("click", async () => {
      await apiFetch("/api/auth/logout", { method: "POST" }).catch(() => {});
      clearSession();
      stopCamera();
      navigate("#/");
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
}

function bindRoom() {
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

  document.querySelector("[data-action='toggle-mic']").addEventListener("click", () => {
    micOn = !micOn;
    if (cameraStream) {
      cameraStream.getAudioTracks().forEach((track) => {
        track.enabled = micOn;
      });
    }
    renderRoom();
  });

  document.querySelector("[data-action='toggle-camera']").addEventListener("click", () => {
    cameraOn = !cameraOn;
    if (cameraStream) {
      cameraStream.getVideoTracks().forEach((track) => {
        track.enabled = cameraOn;
      });
    }
    updateLocalVideo();
    renderRoom();
  });

  document.querySelector("[data-action='toggle-record']").addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  document.querySelector("[data-action='end-call']").addEventListener("click", () => {
    stopRecording();
    stopCamera();
    navigate("#/");
  });
  document.querySelector("[data-action='send-talk']").addEventListener("click", sendTalk);
  document.querySelector("[data-action='refresh-history']").addEventListener("click", loadConversations);
  document.querySelector("[data-action='clear-history']").addEventListener("click", clearHistory);
  document.querySelector("[data-action='refresh-recordings']").addEventListener("click", loadRecordings);
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
    renderAudio.addEventListener("pause", stopRenderPreviewLoop);
    renderAudio.addEventListener("ended", stopRenderPreviewLoop);
    renderAudio.addEventListener("loadedmetadata", updateRenderTimeline);
  }
  loadConversations();
  loadRecordings();
  if (activeAvatarView === "rendered") {
    loadAvatarRenderView();
  }
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
    if (!cameraStream) {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      cameraStream.getAudioTracks().forEach((track) => {
        track.enabled = micOn;
      });
      cameraStream.getVideoTracks().forEach((track) => {
        track.enabled = cameraOn;
      });
    }
    updateLocalVideo();
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

function updateLocalVideo() {
  const video = document.querySelector("#localVideo");
  const placeholder = document.querySelector("#cameraPlaceholder");
  if (!video || !placeholder) return;

  video.srcObject = cameraStream;
  video.hidden = !cameraStream || !cameraOn;
  placeholder.hidden = Boolean(cameraStream && cameraOn);
}

function stopCamera() {
  if (!cameraStream) return;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
}

function setAvatarView(view) {
  activeAvatarView = view === "rendered" ? "rendered" : "video";
  const video = document.querySelector("#avatarVideo");
  const renderView = document.querySelector("#avatarRenderView");
  const toolbar = document.querySelector("#roomToolbar");
  document.querySelectorAll("[data-avatar-view]").forEach((button) => {
    const active = button.dataset.avatarView === activeAvatarView;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  if (video) {
    video.hidden = activeAvatarView !== "video";
    if (activeAvatarView !== "video") video.pause();
  }
  if (renderView) renderView.hidden = activeAvatarView !== "rendered";
  if (toolbar) toolbar.hidden = activeAvatarView === "rendered";
  if (activeAvatarView === "rendered") {
    loadAvatarRenderView();
  } else {
    const renderAudio = document.querySelector("#avatarRenderAudio");
    if (renderAudio) renderAudio.pause();
    stopRenderPreviewLoop();
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
  const controls = new OrbitControls(camera, stage || renderer.domElement);
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

function toggleRenderPlayback() {
  const audio = document.querySelector("#avatarRenderAudio");
  const video = document.querySelector("#avatarVideo");
  if (!audio || !audio.src) return;
  if (audio.paused) {
    pauseAvatarMediaExcept(audio);
    if (video) video.pause();
    audio.play().catch(() => {});
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
    renderPreviewLoopTimer = setTimeout(tick, avatarViewerState.renderPreviewInteracting ? 220 : 140);
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
  const scale = reason === "drag" ? 0.72 : reason === "playback" ? 0.82 : 1;
  return {
    width: Math.max(256, Math.min(960, Math.round(baseWidth * scale))),
    height: Math.max(256, Math.min(1280, Math.round(baseHeight * scale))),
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
    const response = await fetch(`/api/jobs/${encodeURIComponent(avatarRunId)}/viewer/render_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera, frame, width, height }),
    });
    if (!response.ok) throw new Error(await response.text());
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
    if (status) status.textContent = `Render preview failed: ${error.message || error}`;
  } finally {
    if (requestId === avatarViewerState.renderPreviewRequestId) {
      avatarViewerState.renderPreviewLoading = false;
      setRenderPreviewLoading(false);
      if (avatarViewerState.renderPreviewDirty && activeAvatarView === "rendered") {
        const dirtyReason = avatarViewerState.renderPreviewDirtyReason || "idle";
        avatarViewerState.renderPreviewDirtyReason = "idle";
        scheduleRenderPreview(30, dirtyReason);
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

async function startRecording() {
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
  recordedAudioFile = null;
  recordingBuffers = [];
  recordingLength = 0;
  recordedMimeType = getSupportedRecordingType();
  recorder = recordedMimeType
    ? new MediaRecorder(cameraStream, { mimeType: recordedMimeType })
    : new MediaRecorder(cameraStream);
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
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
    updateCurrentRecordingPreview();
    if (recordingStopResolve) recordingStopResolve();
    recordingStopResolve = null;
    recordingStopPromise = null;
    renderRoom();
  });
  recorder.start();
  isRecording = true;
  renderRoom();
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
  const input = document.querySelector("#talkText");
  const replyBox = document.querySelector("#replyBox");
  const sendButton = document.querySelector("[data-action='send-talk']");
  const text = input.value.trim() || "Recorded video/audio turn";

  if (isRecording) {
    await stopRecording();
  }
  if (!recordedChunks.length) {
    replyBox.textContent = "Please click Rec, speak for a moment, then click Stop or Send again.";
    return;
  }
  beginGenerationProgress();
  replyBox.textContent = "Uploading your video and audio...";
  sendButton.disabled = true;
  sendButton.textContent = "Generating...";

  try {
    const videoBlob = new Blob(recordedChunks, { type: recordedMimeType || "video/webm" });
    const avatar = avatars.find((item) => item.id === currentAvatarId) || avatars[0];
    const form = new FormData();
    form.append("avatarId", avatar.avatarId || currentAvatarId);
    form.append("ttsSpeakerId", avatar.ttsSpeakerId || "6224");
    form.append("text", text);
    if (recordedAudioFile) {
      form.append("audio", recordedAudioFile, recordedAudioFile.name);
    }
    form.append("video", videoBlob, `3depb-video-${Date.now()}.webm`);
    setGenerationProgress(18, "Uploading media");
    const rawResponse = await fetch("/api/avatar/respond", {
      method: "POST",
      credentials: "same-origin",
      body: form
    });
    const response = await rawResponse.json().catch(() => ({}));
    if (!rawResponse.ok) {
      throw new Error(response.error || "Avatar generation failed");
    }
    if (response.jobId) {
      input.value = "";
      await pollAvatarJob(response.jobId);
    } else {
      applyAvatarResult(response);
    }
  } catch (err) {
    stopGenerationProgress("Generation failed", generationProgress);
    currentReplyText = err.message;
    replyBox.textContent = err.message;
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send to Digital Human";
  }
}

async function pollAvatarJob(jobId) {
  if (generationPollTimer) clearInterval(generationPollTimer);
  const tick = async () => {
    try {
      const response = await apiFetch(`/api/avatar/jobs/${encodeURIComponent(jobId)}`);
      setGenerationProgress(response.progress || generationProgress, response.stageLabel || response.stage || "Running");
      currentRunId = response.runId || currentRunId;
      if (response.status === "done") {
        clearInterval(generationPollTimer);
        generationPollTimer = null;
        applyAvatarResult(response);
      } else if (response.status === "failed") {
        clearInterval(generationPollTimer);
        generationPollTimer = null;
        throw new Error(response.error || "Avatar generation failed");
      }
    } catch (err) {
      clearInterval(generationPollTimer);
      generationPollTimer = null;
      stopGenerationProgress("Generation failed", generationProgress);
      currentReplyText = err.message;
      const replyBox = document.querySelector("#replyBox");
      if (replyBox) replyBox.textContent = err.message;
    }
  };
  await tick();
  generationPollTimer = setInterval(tick, 2000);
}

async function applyAvatarResult(response) {
  setGenerationProgress(100, "Avatar ready");
  avatarVideoUrl = response.videoUrl || "";
  avatarAudioUrl = response.audioUrl || "";
  currentRunId = response.runId || currentRunId;
  avatarRunId = response.runId || response.run_id || currentRunId || avatarRunId;
  activeAvatarView = "video";
  currentReplyText = response.replyText || "The digital human did not return a response yet.";
  recordedChunks = [];
  recordedAudioFile = null;
  clearCurrentRecordingPreview();
  renderRoom();
  await loadConversations();
  requestAnimationFrame(() => {
    const video = document.querySelector("#avatarVideo");
    if (video) {
      video.currentTime = 0;
      pauseAvatarMediaExcept(video);
      video.play().catch(() => {
        video.muted = true;
        video.play().catch(() => {});
      });
    }
  });
  if (avatarAudioUrl && !avatarVideoUrl) {
    const avatarAudio = document.querySelector("#avatarAudio");
    if (avatarAudio) {
      pauseAvatarMediaExcept(avatarAudio);
      avatarAudio.play().catch(() => {});
    }
  }
  setTimeout(() => stopGenerationProgress("Ready", 0), 1200);
}

function beginGenerationProgress() {
  generationActive = true;
  generationProgress = 8;
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
      generationStatus = next[1];
      updateGenerationProgressDom();
    }
  }, 1400);
  updateGenerationProgressDom();
}

function setGenerationProgress(value, status) {
  generationActive = value < 100;
  generationProgress = value;
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
  generationStatus = status;
  updateGenerationProgressDom();
}

function updateGenerationProgressDom() {
  const wrap = document.querySelector(".generation-progress");
  const badge = document.querySelector(".pane-head .badge");
  if (badge) {
    badge.textContent = generationActive ? "Generating avatar" : "Digital human online";
    badge.classList.toggle("is-generating", generationActive);
  }
  if (!wrap) return;
  wrap.hidden = !generationActive && generationProgress <= 0;
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
  try {
    const payload = await apiFetch(`/api/conversations?avatarId=${encodeURIComponent(currentAvatarId)}`);
    conversations = payload.conversations || [];
    if (historyList) {
      historyList.innerHTML = renderConversationHistory(conversations);
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
          <p class="history-reply">${escapeHtml(item.replyText)}</p>
          ${renderHistoryVideos(item)}
        </article>
      `
    )
    .join("");
}

function renderHistoryVideos(item) {
  const userVideoUrl = historyUserVideoUrl(item);
  const userVideo = userVideoUrl
    ? `
        <div class="history-video-block">
          <span>Your recording</span>
          <video class="history-video" src="${escapeHtml(userVideoUrl)}" controls preload="metadata"></video>
        </div>
      `
    : "";
  const avatarVideo = item.videoUrl
    ? `
        <div class="history-video-block">
          <span>Generated avatar</span>
          <video class="history-video" src="${escapeHtml(item.videoUrl)}" controls preload="metadata"></video>
        </div>
      `
    : "";
  if (!userVideo && !avatarVideo) return "";
  return `<div class="history-videos">${userVideo}${avatarVideo}</div>`;
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
    await apiFetch(`/api/history?avatarId=${encodeURIComponent(currentAvatarId)}`, { method: "DELETE" });
    conversations = [];
    recordings = [];
    clearCurrentRecordingPreview();
    const historyList = document.querySelector("#historyList");
    const recordingsList = document.querySelector("#recordingsList");
    if (historyList) historyList.innerHTML = renderConversationHistory(conversations);
    if (recordingsList) recordingsList.innerHTML = renderRecordings(recordings);
  } catch (err) {
    alert(err.message);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Clear";
    }
  }
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
  const blob = new Blob(recordedChunks, { type: recordedMimeType || "video/webm" });
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
  let speakers = [];
  let speakerLoadError = "";
  try {
    ({ digitalHumans } = await apiFetch("/api/digital_humans"));
  } catch (err) {
    digitalHumans = avatars;
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
                  <span class="admin-avatar-preview" style="${avatarSwatchStyle(item)}"></span>
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
  document.querySelectorAll("[data-human-save]").forEach((button) => {
    button.addEventListener("click", async () => {
      const avatarId = button.dataset.humanSave;
      const nameInput = document.querySelector(`[data-human-name="${cssEscape(avatarId)}"]`);
      const roleInput = document.querySelector(`[data-human-role="${cssEscape(avatarId)}"]`);
      const name = nameInput ? nameInput.value.trim() : "";
      const role = roleInput ? roleInput.value.trim() : "";
      const speaker = document.querySelector(`[data-human-speaker="${cssEscape(avatarId)}"]`);
      const image = document.querySelector(`[data-human-image="${cssEscape(avatarId)}"]`);
      const ttsSpeakerId = (speaker && speaker.value) || "6224";
      button.disabled = true;
      button.textContent = "Saving...";
      try {
        await apiFetch(`/api/digital_humans/${encodeURIComponent(avatarId)}`, {
          method: "PATCH",
          body: JSON.stringify({ name, role, ttsSpeakerId })
        });
        if (image && image.files && image.files[0]) {
          const form = new FormData();
          form.append("image", image.files[0], image.files[0].name);
          const response = await fetch(`/api/digital_humans/${encodeURIComponent(avatarId)}/image`, {
            method: "POST",
            credentials: "same-origin",
            body: form
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(payload.error || "Image upload failed");
        }
        digitalHumansLoaded = false;
        await loadDigitalHumans();
        await renderDigitalHumansAdmin();
      } catch (err) {
        alert(err.message);
        await renderDigitalHumansAdmin();
      }
    });
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

function cssEscape(value) {
  if (window.CSS && window.CSS.escape) return window.CSS.escape(String(value));
  return String(value).replace(/["\\]/g, "\\$&");
}

async function render() {
  const route = location.hash || "#/";
  if (route === "#/room" || route === "#/admin" || route === "#/digital-humans") {
    await hydrateSession();
  }
  if (route === "#/room") {
    await loadDigitalHumans();
    renderRoom();
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
