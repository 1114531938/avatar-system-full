const els = {
  historyBtn: document.getElementById("historyBtn"),
  exportBtn: document.getElementById("exportBtn"),
  logoutBtn: document.getElementById("logoutBtn"),
  avatarStage: document.getElementById("avatarStage"),
  replyVideo: document.getElementById("replyVideo"),
  stageEmpty: document.getElementById("stageEmpty"),
  userPreview: document.getElementById("userPreview"),
  cameraBtn: document.getElementById("cameraBtn"),
  recordBtn: document.getElementById("recordBtn"),
  stopBtn: document.getElementById("stopBtn"),
  sendBtn: document.getElementById("sendBtn"),
  replayBtn: document.getElementById("replayBtn"),
  statusText: document.getElementById("statusText"),
  detailText: document.getElementById("detailText"),
  audioPreview: document.getElementById("audioPreview"),
  usernameInput: document.getElementById("usernameInput"),
  passwordInput: document.getElementById("passwordInput"),
  newPasswordInput: document.getElementById("newPasswordInput"),
  resetTokenInput: document.getElementById("resetTokenInput"),
  loginBtn: document.getElementById("loginBtn"),
  registerBtn: document.getElementById("registerBtn"),
  changePasswordBtn: document.getElementById("changePasswordBtn"),
  resetPasswordBtn: document.getElementById("resetPasswordBtn"),
  confirmResetBtn: document.getElementById("confirmResetBtn"),
  sessionTitle: document.getElementById("sessionTitle"),
  backgroundSelect: document.getElementById("backgroundSelect"),
  backgroundImageInput: document.getElementById("backgroundImageInput"),
  newSessionBtn: document.getElementById("newSessionBtn"),
  matchCard: document.getElementById("matchCard"),
  conversationList: document.getElementById("conversationList"),
  turnCount: document.getElementById("turnCount"),
  historyDrawer: document.getElementById("historyDrawer"),
  closeHistoryBtn: document.getElementById("closeHistoryBtn"),
  historyList: document.getElementById("historyList"),
};

let currentUser = null;
let currentSession = null;
let cameraStream = null;
let recorder = null;
let recordedVideoBlob = null;
let recordedAudioFile = null;
let recordedChunks = [];
let pollTimer = null;
let exportPollTimer = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let recordingBuffers = [];
let recordingLength = 0;
let lastReplyUrl = "";

function setStatus(status, detail = "") {
  els.statusText.textContent = status;
  els.detailText.textContent = detail;
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

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function updateAuthUi() {
  els.logoutBtn.hidden = !currentUser;
  els.recordBtn.disabled = !currentUser || !cameraStream;
  els.sendBtn.disabled = !currentUser || !recordedAudioFile;
  if (currentUser) {
    setStatus("Signed in", `Welcome ${currentUser.username}. Create or continue a session.`);
  }
}

async function loadConfig() {
  const payload = await fetchJson("/api/booth/config");
  currentUser = payload.user || null;
  els.backgroundSelect.innerHTML = "";
  for (const bg of payload.backgrounds || []) {
    const option = document.createElement("option");
    option.value = bg.id;
    option.textContent = bg.label;
    els.backgroundSelect.appendChild(option);
  }
  applyBackground();
  updateAuthUi();
  if (currentUser) {
    await ensureSession();
    await loadHistory();
  }
}

async function ensureSession() {
  if (currentSession) return currentSession;
  const sessions = await fetchJson("/api/booth/sessions");
  if (sessions.sessions && sessions.sessions.length) {
    currentSession = sessions.sessions[0];
    await refreshSession();
    return currentSession;
  }
  return createSession();
}

async function createSession() {
  const formData = new FormData();
  formData.append("title", els.sessionTitle.value || "Emotional Avatar Booth Demo");
  formData.append("background_id", els.backgroundSelect.value || "soft_studio");
  const payload = await fetchJson("/api/booth/sessions", { method: "POST", body: formData });
  currentSession = payload.session;
  renderSession();
  await loadHistory();
  return currentSession;
}

function applyBackground() {
  els.avatarStage.dataset.bg = els.backgroundSelect.value || "soft_studio";
  const file = els.backgroundImageInput.files && els.backgroundImageInput.files[0];
  if (file) {
    els.avatarStage.style.backgroundImage = `url("${URL.createObjectURL(file)}")`;
    els.avatarStage.style.backgroundSize = "cover";
    els.avatarStage.style.backgroundPosition = "center";
  } else {
    els.avatarStage.style.backgroundImage = "";
  }
}

async function enableCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
  });
  els.userPreview.srcObject = cameraStream;
  els.cameraBtn.textContent = "Camera On";
  els.recordBtn.disabled = !currentUser;
  setStatus("Camera ready", "Record a short video message for the avatar.");
}

async function startRecording() {
  if (!cameraStream) await enableCamera();
  recordedChunks = [];
  recordedVideoBlob = null;
  recordedAudioFile = null;
  recordingBuffers = [];
  recordingLength = 0;

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(cameraStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(channel);
    recordingBuffers.push(copy);
    recordingLength += copy.length;
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
    ? "video/webm;codecs=vp9,opus"
    : "video/webm";
  recorder = new MediaRecorder(cameraStream, { mimeType });
  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
  };
  recorder.onstop = async () => {
    recordedVideoBlob = new Blob(recordedChunks, { type: "video/webm" });
    const wavBlob = encodeWav(recordingBuffers, recordingLength, audioContext.sampleRate);
    recordedAudioFile = new File([wavBlob], `booth_turn_${Date.now()}.wav`, { type: "audio/wav" });
    els.audioPreview.src = URL.createObjectURL(recordedAudioFile);
    els.audioPreview.hidden = false;
    els.sendBtn.disabled = !currentUser;
    els.recordBtn.disabled = !currentUser;
    els.stopBtn.disabled = true;
    if (processorNode) processorNode.disconnect();
    if (sourceNode) sourceNode.disconnect();
    await audioContext.close();
    setStatus("Turn recorded", "Send it to generate the avatar reply.");
  };
  recorder.start(250);
  els.recordBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.sendBtn.disabled = true;
  setStatus("Recording", "Speak naturally. Your video will be attached to this turn.");
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

async function sendTurn() {
  if (!recordedAudioFile) {
    setStatus("No recording", "Record a turn before sending.");
    return;
  }
  await ensureSession();
  const formData = new FormData();
  formData.append("audio", recordedAudioFile, recordedAudioFile.name);
  if (recordedVideoBlob) {
    formData.append("video", new File([recordedVideoBlob], `booth_turn_${Date.now()}.webm`, { type: "video/webm" }));
  }
  formData.append("background_id", els.backgroundSelect.value || "soft_studio");
  const backgroundFile = els.backgroundImageInput.files && els.backgroundImageInput.files[0];
  if (backgroundFile) {
    formData.append("background_image", backgroundFile, backgroundFile.name);
  }
  els.sendBtn.disabled = true;
  setStatus("Generating", "Input Agent and Plan Agent are preparing the avatar reply.");
  const payload = await fetchJson(`/api/booth/sessions/${encodeURIComponent(currentSession.id)}/turns`, {
    method: "POST",
    body: formData,
  });
  currentSession.turns = [...(currentSession.turns || []), payload.turn];
  renderSession();
  startPolling();
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshSession, 3000);
  refreshSession();
}

async function refreshSession() {
  if (!currentSession) return;
  const payload = await fetchJson(`/api/booth/sessions/${encodeURIComponent(currentSession.id)}`);
  currentSession = payload.session;
  renderSession();
  const hasRunning = (currentSession.turns || []).some((turn) => ["queued", "running", "unknown"].includes(turn.status));
  if (!hasRunning && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function renderSession() {
  const turns = currentSession?.turns || [];
  els.turnCount.textContent = `${turns.length} ${turns.length === 1 ? "turn" : "turns"}`;
  els.conversationList.innerHTML = "";
  for (const turn of turns.slice().reverse()) {
    const card = document.createElement("article");
    card.className = "turn-card";
    const match = turn.match_result || {};
    const title = document.createElement("strong");
    title.textContent = `Turn ${turns.indexOf(turn) + 1} · ${turn.status}`;
    const detail = document.createElement("p");
    detail.textContent = turn.reply_text || turn.error || match.reason || "Waiting for generated reply.";
    card.append(title, detail);
    if (turn.reply_video_url) {
      const link = document.createElement("a");
      link.href = turn.reply_video_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Open reply video";
      card.appendChild(link);
    }
    els.conversationList.appendChild(card);
  }
  const latestReady = turns.slice().reverse().find((turn) => turn.reply_video_url);
  if (latestReady) {
    if (latestReady.reply_video_url !== lastReplyUrl) {
      lastReplyUrl = latestReady.reply_video_url;
      playReply(lastReplyUrl);
    }
    els.replayBtn.disabled = false;
    const match = latestReady.match_result || {};
    els.matchCard.textContent = `Avatar ${match.avatar_id || "-"} and voice ${match.tts_speaker_id || "-"} selected. ${match.reason || ""}`;
    setStatus("Reply ready", latestReady.reply_text || "Avatar response generated.");
  }
  if (currentSession?.export_video_url) {
    els.detailText.innerHTML = `Conversation export ready: <a href="${currentSession.export_video_url}" target="_blank" rel="noreferrer">download video</a>`;
  } else if (currentSession?.export_status === "running" || currentSession?.export_status === "queued") {
    setStatus("Exporting", "The full conversation video is being assembled.");
  } else if (currentSession?.export_status === "failed") {
    setStatus("Export failed", "Check the Booth service log or session export log.");
  }
}

function playReply(url) {
  els.stageEmpty.hidden = true;
  els.replyVideo.hidden = false;
  els.replyVideo.src = `${url}?t=${Date.now()}`;
  els.replyVideo.load();
  const playPromise = els.replyVideo.play();
  if (playPromise) playPromise.catch(() => {});
}

async function loadHistory() {
  if (!currentUser) return;
  const payload = await fetchJson("/api/booth/sessions");
  els.historyList.innerHTML = "";
  for (const session of payload.sessions || []) {
    const item = document.createElement("article");
    item.className = "history-item";
    const title = document.createElement("strong");
    title.textContent = session.title;
    const meta = document.createElement("p");
    meta.textContent = `${session.updated_at} · export: ${session.export_status || "idle"}`;
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "Open";
    open.addEventListener("click", async () => {
      currentSession = session;
      await refreshSession();
      els.historyDrawer.hidden = true;
    });
    item.append(title, meta, open);
    if (session.export_video_url) {
      const link = document.createElement("a");
      link.href = session.export_video_url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "Export video";
      item.appendChild(link);
    }
    els.historyList.appendChild(item);
  }
}

async function exportSession() {
  if (!currentSession) return;
  await fetchJson(`/api/booth/sessions/${encodeURIComponent(currentSession.id)}/export`, { method: "POST" });
  setStatus("Export queued", "The full conversation video is being assembled.");
  if (exportPollTimer) clearInterval(exportPollTimer);
  exportPollTimer = setInterval(async () => {
    await refreshSession();
    await loadHistory();
    if (!["queued", "running"].includes(currentSession?.export_status)) {
      clearInterval(exportPollTimer);
      exportPollTimer = null;
    }
  }, 2500);
}

async function loginOrRegister(mode) {
  const payload = await fetchJson(`/api/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: els.usernameInput.value,
      password: els.passwordInput.value,
    }),
  });
  currentUser = payload.user;
  updateAuthUi();
  await ensureSession();
  await loadHistory();
}

els.loginBtn.addEventListener("click", () => loginOrRegister("login").catch((error) => setStatus("Login failed", error.message)));
els.registerBtn.addEventListener("click", () => loginOrRegister("register").catch((error) => setStatus("Register failed", error.message)));
els.logoutBtn.addEventListener("click", async () => {
  await fetchJson("/api/auth/logout", { method: "POST" });
  currentUser = null;
  currentSession = null;
  updateAuthUi();
  renderSession();
});
els.changePasswordBtn.addEventListener("click", async () => {
  await fetchJson("/api/auth/change_password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: els.passwordInput.value,
      new_password: els.newPasswordInput.value,
    }),
  });
  setStatus("Password changed", "Use the new password next time.");
});
els.resetPasswordBtn.addEventListener("click", async () => {
  const payload = await fetchJson("/api/auth/reset_password/request", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: els.usernameInput.value }),
  });
  if (payload.reset_token) els.resetTokenInput.value = payload.reset_token;
  setStatus("Reset token issued", "The local reset token is also written to the service log.");
});
els.confirmResetBtn.addEventListener("click", async () => {
  await fetchJson("/api/auth/reset_password/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      token: els.resetTokenInput.value,
      new_password: els.newPasswordInput.value,
    }),
  });
  setStatus("Password reset", "Login with the new password.");
});
els.cameraBtn.addEventListener("click", () => enableCamera().catch((error) => setStatus("Camera failed", error.message)));
els.recordBtn.addEventListener("click", () => startRecording().catch((error) => setStatus("Record failed", error.message)));
els.stopBtn.addEventListener("click", stopRecording);
els.sendBtn.addEventListener("click", () => sendTurn().catch((error) => setStatus("Send failed", error.message)));
els.replayBtn.addEventListener("click", () => {
  if (lastReplyUrl) playReply(lastReplyUrl);
});
els.newSessionBtn.addEventListener("click", () => createSession().catch((error) => setStatus("Session failed", error.message)));
els.backgroundSelect.addEventListener("change", applyBackground);
els.backgroundImageInput.addEventListener("change", applyBackground);
els.exportBtn.addEventListener("click", () => exportSession().catch((error) => setStatus("Export failed", error.message)));
els.historyBtn.addEventListener("click", async () => {
  await loadHistory();
  els.historyDrawer.hidden = false;
});
els.closeHistoryBtn.addEventListener("click", () => {
  els.historyDrawer.hidden = true;
});

loadConfig().catch((error) => setStatus("Booth unavailable", error.message));
