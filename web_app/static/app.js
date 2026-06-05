import * as THREE from "/static/vendor/three.module.js";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { PLYLoader } from "/static/vendor/PLYLoader.js";

const els = {
  avatarPreset: document.getElementById("avatarPreset"),
  avatarId: document.getElementById("avatarId"),
  ttsSpeaker: document.getElementById("ttsSpeaker"),
  ttsPreviewBtn: document.getElementById("ttsPreviewBtn"),
  ttsPreviewStatus: document.getElementById("ttsPreviewStatus"),
  ttsPreviewAudio: document.getElementById("ttsPreviewAudio"),
  apiKey: document.getElementById("apiKey"),
  baseUrl: document.getElementById("baseUrl"),
  modelName: document.getElementById("modelName"),
  saveSettingsBtn: document.getElementById("saveSettingsBtn"),
  keyStatus: document.getElementById("keyStatus"),
  recordBtn: document.getElementById("recordBtn"),
  stopBtn: document.getElementById("stopBtn"),
  meterFill: document.getElementById("meterFill"),
  recordPreview: document.getElementById("recordPreview"),
  fileInput: document.getElementById("fileInput"),
  fileLabel: document.getElementById("fileLabel"),
  submitBtn: document.getElementById("submitBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  noLlm: document.getElementById("noLlm"),
  prepareOnly: document.getElementById("prepareOnly"),
  noVideo: document.getElementById("noVideo"),
  serverStatus: document.getElementById("serverStatus"),
  jobSummary: document.getElementById("jobSummary"),
  videoTab: document.getElementById("videoTab"),
  renderedTab: document.getElementById("renderedTab"),
  debugTab: document.getElementById("debugTab"),
  finalVideo: document.getElementById("finalVideo"),
  viewerPanel: document.getElementById("viewerPanel"),
  viewerStage: document.getElementById("viewerStage"),
  viewerCanvas: document.getElementById("viewerCanvas"),
  viewerGpuCanvas: document.getElementById("viewerGpuCanvas"),
  renderPreviewImage: document.getElementById("renderPreviewImage"),
  renderPreviewLoading: document.getElementById("renderPreviewLoading"),
  viewerNotice: document.getElementById("viewerNotice"),
  viewerPlayBtn: document.getElementById("viewerPlayBtn"),
  viewerSeek: document.getElementById("viewerSeek"),
  viewerTime: document.getElementById("viewerTime"),
  viewerResetBtn: document.getElementById("viewerResetBtn"),
  viewerFitBtn: document.getElementById("viewerFitBtn"),
  viewerExportBtn: document.getElementById("viewerExportBtn"),
  debugControls: document.getElementById("debugControls"),
  debugPointBtn: document.getElementById("debugPointBtn"),
  debugSplatBtn: document.getElementById("debugSplatBtn"),
  splatControls: document.getElementById("splatControls"),
  splatSize: document.getElementById("splatSize"),
  splatOpacity: document.getElementById("splatOpacity"),
  renderPreviewStatus: document.getElementById("renderPreviewStatus"),
  viewerExportStatus: document.getElementById("viewerExportStatus"),
  viewerAudio: document.getElementById("viewerAudio"),
  downloads: document.getElementById("downloads"),
  paths: document.getElementById("paths"),
  logBox: document.getElementById("logBox"),
};

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let recordingBuffers = [];
let recordingLength = 0;
let recordedFile = null;
let pollTimer = null;
let currentRunId = null;
let activeView = "video";
let debugView = "pointcloud";
let renderPreviewTimer = null;
let renderPreviewLoopTimer = null;
let splatSortTimer = null;
let viewerState = {
  runId: null,
  assets: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  pointCloud: null,
  splatCloud: null,
  webgpuReady: false,
  webgpuSupported: false,
  webgpuContext: null,
  webgpuDevice: null,
  webgpuFormat: null,
  webgpuPipeline: null,
  webgpuUniformBuffer: null,
  webgpuPointBuffer: null,
  webgpuPointBufferSize: 0,
  webgpuOrderBuffer: null,
  webgpuOrderBufferSize: 0,
  webgpuBindGroup: null,
  webgpuPointCount: 0,
  webgpuOrderArray: null,
  webgpuData: null,
  webgpuDirty: false,
  centerOffset: [0, 0, 0],
  defaultCameraDistance: 1,
  animationId: null,
  fps: 25,
  frameCount: 0,
  loading: false,
  exportPollTimer: null,
  renderPreviewLoading: false,
  renderPreviewDirty: false,
  renderPreviewDirtyReason: "idle",
  pendingRenderReason: "idle",
  renderPreviewRequestId: 0,
  lastRenderedFrame: -1,
  lastRenderedSignature: "",
  renderPreviewInteracting: false,
  splatBaseScale: 1,
  splatSortVersion: 0,
  splatMotionLoading: false,
  splatMotionMeta: null,
  splatMotionData: null,
  splatMotionCenter: [0, 0, 0],
  splatMotionRadius: 0,
  currentSplatMotionFrame: -1,
  exportedVideoUrl: null,
  renderPreviewObjectUrl: null,
};

function setStatus(text) {
  els.serverStatus.textContent = text;
}

function avatarPresetLabel(item) {
  if (!item) return "";
  const parts = [item.id];
  if (item.label && item.label !== item.id) parts.push(item.label);
  return parts.join(" · ");
}

function ttsSpeakerLabel(item) {
  if (!item) return "";
  return item.label || item.id || "";
}

function syncAvatarPresetFromInput() {
  const value = (els.avatarId.value || "").trim();
  const match = Array.from(els.avatarPreset.options).find((option) => option.value === value);
  els.avatarPreset.value = match ? value : "__custom__";
}

async function loadTtsSpeakers() {
  try {
    const response = await fetch("/api/tts_speakers");
    if (!response.ok) return;
    const payload = await response.json();
    const speakers = Array.isArray(payload.speakers) ? payload.speakers : [];
    const defaultSpeakerId = String(payload.default_speaker_id || "6224");
    els.ttsSpeaker.innerHTML = "";
    for (const speaker of speakers) {
      const option = document.createElement("option");
      option.value = String(speaker.id);
      option.textContent = ttsSpeakerLabel(speaker);
      els.ttsSpeaker.appendChild(option);
    }
    if (speakers.some((speaker) => String(speaker.id) === defaultSpeakerId)) {
      els.ttsSpeaker.value = defaultSpeakerId;
    }
  } catch (error) {
    console.warn("Failed to load TTS speakers", error);
  }
}

async function loadAvatarOptions() {
  try {
    const response = await fetch("/api/avatars");
    if (!response.ok) return;
    const payload = await response.json();
    const avatars = Array.isArray(payload.avatars) ? payload.avatars : [];
    const currentValue = (els.avatarId.value || "306").trim();
    els.avatarPreset.innerHTML = "";
    for (const avatar of avatars) {
      const option = document.createElement("option");
      option.value = String(avatar.id);
      option.textContent = avatarPresetLabel(avatar);
      els.avatarPreset.appendChild(option);
    }
    const customOption = document.createElement("option");
    customOption.value = "__custom__";
    customOption.textContent = "Custom ID...";
    els.avatarPreset.appendChild(customOption);
    const known = avatars.some((avatar) => String(avatar.id) === currentValue);
    els.avatarPreset.value = known ? currentValue : "__custom__";
  } catch (error) {
    console.warn("Failed to load avatar options", error);
  }
}

async function loadSettings() {
  const response = await fetch("/api/settings");
  if (!response.ok) return;
  const settings = await response.json();
  els.baseUrl.value = settings.openai_base_url || "https://openrouter.ai/api/v1";
  els.modelName.value = settings.llm_model || "openai/gpt-oss-120b:free";
  els.keyStatus.textContent = settings.has_openai_api_key
    ? `Key configured (${settings.openai_key_preview})`
    : "Key missing";
}

els.avatarPreset.addEventListener("change", () => {
  if (els.avatarPreset.value === "__custom__") {
    els.avatarId.focus();
    els.avatarId.select();
    return;
  }
  els.avatarId.value = els.avatarPreset.value;
});

els.avatarId.addEventListener("input", () => {
  syncAvatarPresetFromInput();
});

els.ttsPreviewBtn.addEventListener("click", async () => {
  const speakerId = els.ttsSpeaker.value || "6224";
  els.ttsPreviewBtn.disabled = true;
  els.ttsPreviewStatus.textContent = "Generating preview...";
  try {
    const response = await fetch("/api/tts_preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_id: speakerId }),
    });
    if (!response.ok) {
      const detail = await response.text();
      let message = detail || "Preview failed";
      try {
        const payload = JSON.parse(detail);
        message = payload.detail || message;
      } catch {
        // Keep the raw response text when the server did not return JSON.
      }
      throw new Error(message);
    }
    const payload = await response.json();
    els.ttsPreviewAudio.src = `${payload.audio_url}?t=${Date.now()}`;
    els.ttsPreviewAudio.hidden = false;
    els.ttsPreviewStatus.textContent = `Preview ready: ${payload.speaker_id}`;
    await els.ttsPreviewAudio.play();
  } catch (error) {
    els.ttsPreviewStatus.textContent = `Preview failed: ${error.message || error}`;
  } finally {
    els.ttsPreviewBtn.disabled = false;
  }
});

els.saveSettingsBtn.addEventListener("click", async () => {
  els.saveSettingsBtn.disabled = true;
  els.keyStatus.textContent = "Saving...";
  const response = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      openai_api_key: els.apiKey.value,
      openai_base_url: els.baseUrl.value,
      llm_model: els.modelName.value,
    }),
  });
  if (!response.ok) {
    els.keyStatus.textContent = "Save failed";
    els.saveSettingsBtn.disabled = false;
    return;
  }
  els.apiKey.value = "";
  els.saveSettingsBtn.disabled = false;
  loadSettings();
});

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

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

els.recordBtn.addEventListener("click", async () => {
  audioContext = new AudioContext();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  recordingBuffers = [];
  recordingLength = 0;

  processorNode.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0);
    const copy = new Float32Array(channel);
    recordingBuffers.push(copy);
    recordingLength += copy.length;

    let peak = 0;
    for (let i = 0; i < channel.length; i += 1) {
      peak = Math.max(peak, Math.abs(channel[i]));
    }
    els.meterFill.style.width = `${Math.min(100, peak * 140)}%`;
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
  recordedFile = null;
  els.fileInput.value = "";
  els.fileLabel.textContent = "Recording...";
  els.recordBtn.disabled = true;
  els.stopBtn.disabled = false;
});

els.stopBtn.addEventListener("click", async () => {
  if (processorNode) processorNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());

  const wavBlob = encodeWav(recordingBuffers, recordingLength, audioContext.sampleRate);
  recordedFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: "audio/wav" });
  els.recordPreview.src = URL.createObjectURL(recordedFile);
  els.recordPreview.hidden = false;
  els.fileLabel.textContent = recordedFile.name;
  els.meterFill.style.width = "0%";
  els.recordBtn.disabled = false;
  els.stopBtn.disabled = true;

  await audioContext.close();
});

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  if (!file) return;
  recordedFile = null;
  els.fileLabel.textContent = file.name;
  els.recordPreview.hidden = true;
});

els.submitBtn.addEventListener("click", async () => {
  const file = recordedFile || els.fileInput.files[0];
  if (!file) {
    els.jobSummary.textContent = "Record audio or choose a WAV file first.";
    return;
  }

  const formData = new FormData();
  formData.append("audio", file, file.name);
  formData.append("avatar_id", els.avatarId.value || "306");
  formData.append("tts_speaker_id", els.ttsSpeaker.value || "6224");
  formData.append("no_llm", els.noLlm.checked ? "true" : "false");
  formData.append("prepare_only", els.prepareOnly.checked ? "true" : "false");
  formData.append("no_video_export", els.noVideo.checked ? "true" : "false");

  els.submitBtn.disabled = true;
  setStatus("Starting");
  els.jobSummary.textContent = "Submitting job...";
  els.logBox.textContent = "";

  const response = await fetch("/api/jobs", { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.text();
    els.jobSummary.textContent = `Submit failed: ${detail}`;
    els.submitBtn.disabled = false;
    setStatus("Error");
    return;
  }
  const payload = await response.json();
  currentRunId = payload.run_id;
  viewerState.exportedVideoUrl = null;
  els.refreshBtn.disabled = false;
  pollJob();
  pollTimer = setInterval(pollJob, 3000);
});

els.refreshBtn.addEventListener("click", () => {
  if (currentRunId) pollJob();
});

els.videoTab.addEventListener("click", () => setActiveView("video"));
els.renderedTab.addEventListener("click", () => setActiveView("rendered"));
els.debugTab.addEventListener("click", () => setActiveView(debugView));
els.debugPointBtn.addEventListener("click", () => setActiveView("pointcloud"));
els.debugSplatBtn.addEventListener("click", () => setActiveView("splat"));
els.viewerPlayBtn.addEventListener("click", () => toggleViewerPlayback());
els.viewerResetBtn.addEventListener("click", () => resetViewerCamera());
els.viewerFitBtn.addEventListener("click", () => fitCurrentSubject());
els.viewerExportBtn.addEventListener("click", () => startViewerExport());
els.splatSize.addEventListener("input", () => updateSplatUniforms());
els.splatOpacity.addEventListener("input", () => updateSplatUniforms());
els.viewerSeek.addEventListener("input", () => {
  const duration = Number.isFinite(els.viewerAudio.duration) ? els.viewerAudio.duration : 0;
  if (duration > 0) {
    els.viewerAudio.currentTime = Number(els.viewerSeek.value) * duration;
    updateViewerTimeline();
    if (activeView === "rendered") scheduleRenderPreview(0, "seek");
  }
});

els.viewerAudio.addEventListener("play", () => {
  els.viewerPlayBtn.textContent = "❚❚";
  if (activeView === "rendered") {
    startRenderPreviewLoop();
    scheduleRenderPreview(0, "playback");
  }
});

els.viewerAudio.addEventListener("pause", () => {
  els.viewerPlayBtn.textContent = "▶";
  stopRenderPreviewLoop();
  if (activeView === "rendered") scheduleRenderPreview(0, "idle");
});

els.viewerAudio.addEventListener("loadedmetadata", updateViewerTimeline);
els.viewerAudio.addEventListener("timeupdate", updateViewerTimeline);
els.viewerAudio.addEventListener("seeked", () => {
  if (activeView === "rendered") scheduleRenderPreview(0, "seek");
});

window.addEventListener("resize", () => {
  resizeViewer();
});

async function pollJob() {
  if (!currentRunId) return;
  const response = await fetch(`/api/jobs/${encodeURIComponent(currentRunId)}`);
  if (!response.ok) return;
  const payload = await response.json();
  renderJob(payload);

  if (payload.status === "done" || payload.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    els.submitBtn.disabled = false;
  }
}

function renderJob(payload) {
  setStatus(payload.status);
  const manifest = payload.manifest || {};
  const stage = payload.state?.current_stage || "";
  const replyText = manifest.reply_text ? `\nReply: ${manifest.reply_text}` : "";
  els.jobSummary.textContent = `Run: ${payload.run_id}\nStatus: ${payload.status}${stage ? `\nStage: ${stage}` : ""}${replyText}`;
  els.logBox.textContent = payload.log_tail || "No logs yet.";

  const urls = payload.artifact_urls || {};
  if (urls.output_video && !viewerState.exportedVideoUrl) {
    els.finalVideo.src = urls.output_video;
    els.finalVideo.hidden = activeView !== "video";
  }
  if (payload.status === "done" && payload.run_id) {
    prepareViewer(payload.run_id);
  }

  const links = [
    ["final_video.mp4", urls.output_video],
    ["white_model.mp4", urls.output_white_model_video],
    ["reply.wav", urls.artifact_reply_wav],
    ["reply_enhanced.wav", urls.artifact_enhanced_reply_wav],
    ["flame_motion.npz", urls.artifact_flame_motion_npz],
  ].filter((item) => item[1]);

  els.downloads.innerHTML = "";
  for (const [label, url] of links) {
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    a.textContent = label;
    els.downloads.appendChild(a);
  }

  const pathLines = [];
  if (payload.run_dir) pathLines.push(`run_dir: ${payload.run_dir}`);
  if (manifest.artifact_dir) pathLines.push(`artifact_dir: ${manifest.artifact_dir}`);
  if (manifest.output_video) pathLines.push(`final_video: ${manifest.output_video}`);
  if (manifest.output_white_model_video) pathLines.push(`white_model: ${manifest.output_white_model_video}`);
  if (manifest.artifact_reply_wav) pathLines.push(`reply_wav: ${manifest.artifact_reply_wav}`);
  els.paths.textContent = pathLines.join("\n");
}

Promise.all([loadSettings(), loadAvatarOptions(), loadTtsSpeakers()]).finally(() => {
  syncAvatarPresetFromInput();
});

function setActiveView(view) {
  if (view === "debug") view = debugView;
  activeView = view;
  const isVideo = view === "video";
  const isRendered = view === "rendered";
  const isSplat = view === "splat";
  const isPointCloud = view === "pointcloud";
  const isDebug = isSplat || isPointCloud;
  if (isDebug) debugView = view;
  els.videoTab.classList.toggle("active", isVideo);
  els.renderedTab.classList.toggle("active", isRendered);
  els.debugTab.classList.toggle("active", isDebug);
  els.videoTab.setAttribute("aria-selected", String(isVideo));
  els.renderedTab.setAttribute("aria-selected", String(isRendered));
  els.debugTab.setAttribute("aria-selected", String(isDebug));
  els.debugPointBtn.classList.toggle("active", isPointCloud);
  els.debugSplatBtn.classList.toggle("active", isSplat);
  els.debugPointBtn.setAttribute("aria-pressed", String(isPointCloud));
  els.debugSplatBtn.setAttribute("aria-pressed", String(isSplat));
  els.finalVideo.hidden = !isVideo || !els.finalVideo.src;
  els.viewerPanel.hidden = isVideo;
  els.renderPreviewImage.hidden = !isRendered || !els.renderPreviewImage.src;
  els.debugControls.hidden = !isDebug;
  els.splatControls.hidden = !isSplat;
  els.viewerCanvas.style.opacity = isRendered ? "0" : "1";
  els.viewerCanvas.hidden = isSplat;
  els.viewerGpuCanvas.hidden = !isSplat;
  if (viewerState.pointCloud) viewerState.pointCloud.visible = isPointCloud;
  if (viewerState.splatCloud) viewerState.splatCloud.visible = isSplat;
  setRenderPreviewLoading(false);
  if (isSplat) {
    els.renderPreviewStatus.textContent = "WebGPU Gaussian debug preview.";
    if (els.splatSize.value === "0.6") els.splatSize.value = "1.1";
    if (els.splatOpacity.value === "1") els.splatOpacity.value = "1.15";
    ensureWebGPU();
    prepareSplatMotion();
    scheduleSplatSort(30);
    renderWebGPUScene();
  } else if (!isRendered) {
    els.renderPreviewStatus.textContent = "";
  }
  if (isRendered) {
    if (els.viewerAudio.paused) {
      stopRenderPreviewLoop();
      scheduleRenderPreview(0, "idle");
    } else {
      startRenderPreviewLoop();
      scheduleRenderPreview(0, "playback");
    }
  } else {
    stopRenderPreviewLoop();
  }
  if (!isVideo) {
    resizeViewer();
    startViewerLoop();
  }
  if (isPointCloud || isRendered) {
    fitCameraToObject(viewerState.pointCloud, 2.6);
  } else if (isSplat && !viewerState.splatMotionData) {
    fitCameraToObject(viewerState.splatCloud, 2.4);
  }
  if (isRendered) scheduleRenderPreview(100, "idle");
}

function clearSplatMotionState() {
  viewerState.splatMotionMeta = null;
  viewerState.splatMotionData = null;
  viewerState.splatMotionCenter = [0, 0, 0];
  viewerState.splatMotionRadius = 0;
  viewerState.currentSplatMotionFrame = -1;
}

function setRenderPreviewLoading(isLoading) {
  els.renderPreviewLoading.hidden = !isLoading || activeView !== "rendered";
}

async function prepareViewer(runId) {
  if (viewerState.runId === runId || viewerState.loading) return;
  viewerState.loading = true;
  els.viewerNotice.hidden = false;
  els.viewerNotice.textContent = "Loading 3D assets...";

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(runId)}/viewer_assets`);
    if (!response.ok) throw new Error(await response.text());
    const assets = await response.json();
    viewerState.runId = runId;
    viewerState.assets = assets;
    viewerState.fps = assets.fps || 25;
    viewerState.frameCount = assets.frame_count || 0;
    viewerState.lastRenderedFrame = -1;
    viewerState.lastRenderedSignature = "";
    viewerState.pendingRenderReason = "idle";
    viewerState.renderPreviewDirtyReason = "idle";
    if (viewerState.renderPreviewObjectUrl) {
      URL.revokeObjectURL(viewerState.renderPreviewObjectUrl);
      viewerState.renderPreviewObjectUrl = null;
    }
    els.renderPreviewImage.removeAttribute("src");
    els.renderPreviewImage.hidden = true;
    clearSplatMotionState();

    if (assets.audio_url && els.viewerAudio.src !== new URL(assets.audio_url, window.location.origin).href) {
      els.viewerAudio.src = assets.audio_url;
      els.viewerAudio.load();
    }

    initViewerScene();
    if (assets.point_cloud_url) {
      await loadPointCloud(assets.point_cloud_url);
      els.viewerNotice.hidden = true;
    } else {
      els.viewerNotice.textContent = "Point cloud is not available for this run.";
    }
    updateViewerTimeline();
  } catch (error) {
    els.viewerNotice.hidden = false;
    els.viewerNotice.textContent = `3D viewer failed: ${error.message || error}`;
  } finally {
    viewerState.loading = false;
  }
}

function initViewerScene() {
  if (viewerState.renderer) return;

  const renderer = new THREE.WebGLRenderer({
    canvas: els.viewerCanvas,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x111416, 1);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111416);

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(0, 0.04, 1.25);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.update();
  controls.addEventListener("start", () => {
    viewerState.renderPreviewInteracting = true;
    if (activeView === "rendered") scheduleRenderPreview(0, "drag");
  });
  controls.addEventListener("change", () => {
    if (activeView === "rendered") {
      scheduleRenderPreview(viewerState.renderPreviewInteracting ? 25 : 120, viewerState.renderPreviewInteracting ? "drag" : "idle");
    } else if (activeView === "splat") {
      scheduleSplatSort(180);
      renderWebGPUScene();
    }
  });
  controls.addEventListener("end", () => {
    viewerState.renderPreviewInteracting = false;
    if (activeView === "rendered") {
      scheduleRenderPreview(40, "drag");
      scheduleRenderPreview(220, "idle");
    }
  });

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambient);
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(0.8, 1.4, 1.0);
  scene.add(key);

  viewerState.renderer = renderer;
  viewerState.scene = scene;
  viewerState.camera = camera;
  viewerState.controls = controls;
  resizeViewer();
  startViewerLoop();
}

async function loadPointCloud(url) {
  const loader = new PLYLoader();
  loader.setCustomPropertyNameMapping({
    fdc: ["f_dc_0", "f_dc_1", "f_dc_2"],
    sh1: ["f_rest_0", "f_rest_15", "f_rest_30"],
    sh2: ["f_rest_1", "f_rest_16", "f_rest_31"],
    sh3: ["f_rest_2", "f_rest_17", "f_rest_32"],
    gaussianScale: ["scale_0", "scale_1", "scale_2"],
    gaussianRotation: ["rot_0", "rot_1", "rot_2", "rot_3"],
    gaussianOpacity: ["opacity"],
  });
  const geometry = await loader.loadAsync(url);
  applyGaussianDcColors(geometry);
  prepareGaussianSplatAttributes(geometry);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const center = new THREE.Vector3();
  geometry.boundingBox?.getCenter(center);
  viewerState.centerOffset = [center.x, center.y, center.z];
  geometry.center();
  geometry.computeBoundingSphere();

  const radius = geometry.boundingSphere?.radius || 1;
  const pointGeometry = geometry.clone();
  const splatGeometry = geometry.clone();
  prepareWebGPUData(pointGeometry);
  geometry.dispose();
  const material = new THREE.PointsMaterial({
    color: 0xf1f5f2,
    vertexColors: pointGeometry.hasAttribute("color"),
    size: Math.max(radius / 280, 0.0015),
    sizeAttenuation: true,
  });
  const pointCloud = new THREE.Points(pointGeometry, material);
  const splatCloud = new THREE.Points(splatGeometry, createSplatMaterial(radius));

  if (viewerState.pointCloud) {
    viewerState.scene.remove(viewerState.pointCloud);
    viewerState.pointCloud.geometry.dispose();
    viewerState.pointCloud.material.dispose();
  }
  if (viewerState.splatCloud) {
    viewerState.scene.remove(viewerState.splatCloud);
    viewerState.splatCloud.material.dispose();
    viewerState.splatCloud.geometry.dispose();
  }
  viewerState.pointCloud = pointCloud;
  viewerState.splatCloud = splatCloud;
  viewerState.splatBaseScale = 1;
  updateSplatUniforms();
  pointCloud.visible = activeView === "pointcloud";
  splatCloud.visible = activeView === "splat";
  viewerState.scene.add(pointCloud);
  viewerState.scene.add(splatCloud);
  resizeViewer();

  const cameraDistance = Math.max(radius * 2.6, 0.8);
  viewerState.defaultCameraDistance = cameraDistance;
  viewerState.camera.position.set(0, radius * 0.12, cameraDistance);
  viewerState.camera.near = Math.max(radius / 200, 0.001);
  viewerState.camera.far = Math.max(radius * 20, 10);
  viewerState.camera.updateProjectionMatrix();
  viewerState.controls.target.set(0, 0, 0);
  viewerState.controls.update();
  if (activeView === "rendered") scheduleRenderPreview(100);
  if (activeView === "splat") {
    ensureWebGPU();
    scheduleSplatSort(50);
    renderWebGPUScene();
  }
}

function applyGaussianDcColors(geometry) {
  const fdc = geometry.getAttribute("fdc");
  if (!fdc) return;

  const shC0 = 0.28209479177387814;
  const colors = new Float32Array(fdc.count * 3);
  const color = new THREE.Color();

  for (let i = 0; i < fdc.count; i += 1) {
    const r = Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getX(i)));
    const g = Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getY(i)));
    const b = Math.min(1, Math.max(0, 0.5 + shC0 * fdc.getZ(i)));
    color.setRGB(r, g, b).convertSRGBToLinear();
    colors[i * 3 + 0] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

function prepareGaussianSplatAttributes(geometry) {
  const count = geometry.getAttribute("position").count;
  const scale = geometry.getAttribute("gaussianScale");
  const opacity = geometry.getAttribute("gaussianOpacity");
  const radii = new Float32Array(count);
  const alphas = new Float32Array(count);

  for (let i = 0; i < count; i += 1) {
    if (scale) {
      const maxLogScale = Math.max(scale.getX(i), scale.getY(i), scale.getZ(i));
      radii[i] = Math.min(Math.max(Math.exp(maxLogScale), 0.00008), 0.012);
    } else {
      radii[i] = 0.004;
    }

    if (opacity) {
      const rawOpacity = opacity.getX(i);
      alphas[i] = Math.min(0.42, Math.max(0.012, 1 / (1 + Math.exp(-rawOpacity))));
    } else {
      alphas[i] = 0.18;
    }
  }

  geometry.setAttribute("gaussianRadius", new THREE.BufferAttribute(radii, 1));
  geometry.setAttribute("gaussianAlpha", new THREE.BufferAttribute(alphas, 1));
}

function prepareWebGPUData(geometry) {
  const position = geometry.getAttribute("position");
  const color = geometry.getAttribute("color");
  const radius = geometry.getAttribute("gaussianRadius");
  const alpha = geometry.getAttribute("gaussianAlpha");
  const scale = geometry.getAttribute("gaussianScale");
  if (!position || !color || !radius || !alpha) {
    viewerState.webgpuData = null;
    viewerState.webgpuPointCount = 0;
    return;
  }

  const bounds = robustBoundsFromPosition(position, 0.08, 0.92);
  const center = bounds.center;
  const radiusBound = Math.max(bounds.radius, 0.01);
  const extentX = radiusBound * 1.25;
  const extentY = radiusBound * 1.45;
  const extentZ = radiusBound * 1.2;
  const alphaValues = Array.from(alpha.array).filter(Number.isFinite).sort((a, b) => a - b);
  const alphaFloor = alphaValues.length
    ? alphaValues[Math.floor(alphaValues.length * 0.3)]
    : 0.03;
  const minAlpha = Math.max(0.03, Math.min(alphaFloor, 0.12));

  const kept = [];
  for (let i = 0; i < position.count; i += 1) {
    const x = position.getX(i);
    const y = position.getY(i);
    const z = position.getZ(i);
    const a = alpha.getX(i);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) || !Number.isFinite(a)) continue;
    if (Math.abs(x - center.x) > extentX) continue;
    if (Math.abs(y - center.y) > extentY) continue;
    if (Math.abs(z - center.z) > extentZ) continue;
    if (a < minAlpha) continue;
    kept.push(i);
  }

  if (!kept.length) {
    for (let i = 0; i < position.count; i += 1) kept.push(i);
  }

  const keptCount = kept.length;
  const positions = new Float32Array(keptCount * 3);
  const colors = new Float32Array(keptCount * 3);
  const radii = new Float32Array(keptCount);
  const alphas = new Float32Array(keptCount);
  const scales = scale ? new Float32Array(keptCount * 3) : null;

  for (let j = 0; j < keptCount; j += 1) {
    const i = kept[j];
    const src3 = i * 3;
    const dst3 = j * 3;
    positions[dst3 + 0] = position.array[src3 + 0];
    positions[dst3 + 1] = position.array[src3 + 1];
    positions[dst3 + 2] = position.array[src3 + 2];
    colors[dst3 + 0] = color.array[src3 + 0];
    colors[dst3 + 1] = color.array[src3 + 1];
    colors[dst3 + 2] = color.array[src3 + 2];
    radii[j] = Math.max(radius.array[i] * 1.75, 0.0012);
    alphas[j] = Math.min(alpha.array[i] * 1.35, 0.9);
    if (scales) {
      scales[dst3 + 0] = scale.array[src3 + 0];
      scales[dst3 + 1] = scale.array[src3 + 1];
      scales[dst3 + 2] = scale.array[src3 + 2];
    }
  }

  viewerState.webgpuData = {
    positions,
    colors,
    radii,
    alphas,
    scales,
    sourceIndices: new Uint32Array(kept),
  };
  viewerState.webgpuPointCount = keptCount;
  viewerState.webgpuOrderArray = new Uint32Array(keptCount);
  for (let i = 0; i < viewerState.webgpuOrderArray.length; i += 1) {
    viewerState.webgpuOrderArray[i] = i;
  }
  viewerState.webgpuDirty = true;
}

async function ensureWebGPU() {
  if (viewerState.webgpuReady) return true;
  if (!("gpu" in navigator)) {
    viewerState.webgpuSupported = false;
    els.renderPreviewStatus.textContent = "WebGPU is not available in this browser. Falling back to legacy debug preview.";
    els.viewerGpuCanvas.hidden = true;
    els.viewerCanvas.hidden = false;
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");
    const device = await adapter.requestDevice();
    const context = els.viewerGpuCanvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: "opaque",
    });

    const shaderModule = device.createShaderModule({
      code: `
struct Uniforms {
  view_proj: mat4x4<f32>,
  cam_right: vec4<f32>,
  cam_up: vec4<f32>,
  params: vec4<f32>,
};

struct GaussianPoint {
  pos_radius: vec4<f32>,
  color_alpha: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<GaussianPoint>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VsOut {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) local_uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
};

fn corner(vid: u32) -> vec2<f32> {
  switch vid {
    case 0u: { return vec2<f32>(-1.0, -1.0); }
    case 1u: { return vec2<f32>( 1.0, -1.0); }
    case 2u: { return vec2<f32>(-1.0,  1.0); }
    case 3u: { return vec2<f32>(-1.0,  1.0); }
    case 4u: { return vec2<f32>( 1.0, -1.0); }
    default: { return vec2<f32>( 1.0,  1.0); }
  }
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VsOut {
  let point_index = order[instance_index];
  let point = points[point_index];
  let uv = corner(vertex_index);
  let radius = point.pos_radius.w * uniforms.params.x;
  let world = point.pos_radius.xyz
    + uniforms.cam_right.xyz * uv.x * radius
    + uniforms.cam_up.xyz * uv.y * radius;

  var out: VsOut;
  out.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
  out.local_uv = uv;
  out.color = point.color_alpha.xyz;
  out.alpha = point.color_alpha.w * uniforms.params.y;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let dist2 = dot(in.local_uv, in.local_uv);
  if (dist2 > 1.0) {
    discard;
  }
  let gaussian = exp(-3.5 * dist2);
  let alpha = clamp(in.alpha * gaussian, 0.0, 1.0);
  if (alpha < 0.01) {
    discard;
  }
  return vec4<f32>(in.color, alpha);
}
      `,
    });

    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
    });

    const uniformBuffer = device.createBuffer({
      size: 64 + 16 + 16 + 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    viewerState.webgpuSupported = true;
    viewerState.webgpuReady = true;
    viewerState.webgpuContext = context;
    viewerState.webgpuDevice = device;
    viewerState.webgpuFormat = format;
    viewerState.webgpuPipeline = pipeline;
    viewerState.webgpuUniformBuffer = uniformBuffer;
    viewerState.webgpuDirty = true;
    syncWebGPUResources();
    resizeViewer();
    return true;
  } catch (error) {
    viewerState.webgpuSupported = false;
    els.renderPreviewStatus.textContent = `WebGPU init failed: ${error.message || error}`;
    els.viewerGpuCanvas.hidden = true;
    els.viewerCanvas.hidden = false;
    return false;
  }
}

function syncWebGPUResources() {
  if (!viewerState.webgpuReady || !viewerState.webgpuData) return;
  const { webgpuDevice: device, webgpuPipeline: pipeline, webgpuUniformBuffer: uniformBuffer, webgpuData } = viewerState;
  const count = viewerState.webgpuPointCount;
  if (!count) return;

  const packed = new Float32Array(count * 8);
  for (let i = 0; i < count; i += 1) {
    const dst = i * 8;
    const pos = i * 3;
    packed[dst + 0] = webgpuData.positions[pos + 0];
    packed[dst + 1] = webgpuData.positions[pos + 1];
    packed[dst + 2] = webgpuData.positions[pos + 2];
    packed[dst + 3] = webgpuData.radii[i];
    packed[dst + 4] = webgpuData.colors[pos + 0];
    packed[dst + 5] = webgpuData.colors[pos + 1];
    packed[dst + 6] = webgpuData.colors[pos + 2];
    packed[dst + 7] = webgpuData.alphas[i];
  }

  const pointBufferSize = packed.byteLength;
  if (!viewerState.webgpuPointBuffer || viewerState.webgpuPointBufferSize !== pointBufferSize) {
    viewerState.webgpuPointBuffer?.destroy();
    viewerState.webgpuPointBuffer = device.createBuffer({
      size: pointBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    viewerState.webgpuPointBufferSize = pointBufferSize;
  }
  device.queue.writeBuffer(viewerState.webgpuPointBuffer, 0, packed);

  const orderArray = viewerState.webgpuOrderArray || new Uint32Array(count);
  if (!viewerState.webgpuOrderArray) {
    viewerState.webgpuOrderArray = orderArray;
    for (let i = 0; i < count; i += 1) orderArray[i] = i;
  }
  const orderBufferSize = orderArray.byteLength;
  if (!viewerState.webgpuOrderBuffer || viewerState.webgpuOrderBufferSize !== orderBufferSize) {
    viewerState.webgpuOrderBuffer?.destroy();
    viewerState.webgpuOrderBuffer = device.createBuffer({
      size: orderBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    viewerState.webgpuOrderBufferSize = orderBufferSize;
  }
  device.queue.writeBuffer(viewerState.webgpuOrderBuffer, 0, orderArray);

  viewerState.webgpuBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: viewerState.webgpuPointBuffer } },
      { binding: 2, resource: { buffer: viewerState.webgpuOrderBuffer } },
    ],
  });
  viewerState.webgpuDirty = false;
}

function sortWebGPUByDepth() {
  if (!viewerState.webgpuData || !viewerState.camera || !viewerState.webgpuOrderArray) return;
  const matrix = viewerState.camera.matrixWorldInverse.elements;
  const positions = viewerState.webgpuData.positions;
  const count = viewerState.webgpuPointCount;
  const order = Array.from({ length: count }, (_, i) => i);

  order.sort((a, b) => {
    const ax = positions[a * 3 + 0];
    const ay = positions[a * 3 + 1];
    const az = positions[a * 3 + 2];
    const bx = positions[b * 3 + 0];
    const by = positions[b * 3 + 1];
    const bz = positions[b * 3 + 2];
    const ad = matrix[2] * ax + matrix[6] * ay + matrix[10] * az + matrix[14];
    const bd = matrix[2] * bx + matrix[6] * by + matrix[10] * bz + matrix[14];
    return ad - bd;
  });

  for (let i = 0; i < count; i += 1) {
    viewerState.webgpuOrderArray[i] = order[i];
  }
  viewerState.webgpuDirty = true;
}

function renderWebGPUScene() {
  if (activeView !== "splat" || !viewerState.webgpuReady || !viewerState.webgpuBindGroup || !viewerState.camera) return;
  if (viewerState.webgpuDirty) {
    syncWebGPUResources();
  }
  if (!viewerState.webgpuBindGroup) return;

  const device = viewerState.webgpuDevice;
  const camera = viewerState.camera;
  camera.updateMatrixWorld();
  const viewProj = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();
  const uniforms = new Float32Array(28);
  uniforms.set(viewProj.elements, 0);
  uniforms.set([right.x, right.y, right.z, 0], 16);
  uniforms.set([up.x, up.y, up.z, 0], 20);
  uniforms.set([
    Number.parseFloat(els.splatSize.value || "1"),
    Number.parseFloat(els.splatOpacity.value || "1"),
    0,
    0,
  ], 24);
  device.queue.writeBuffer(viewerState.webgpuUniformBuffer, 0, uniforms);

  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: viewerState.webgpuContext.getCurrentTexture().createView(),
        clearValue: { r: 17 / 255, g: 20 / 255, b: 22 / 255, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  renderPass.setPipeline(viewerState.webgpuPipeline);
  renderPass.setBindGroup(0, viewerState.webgpuBindGroup);
  renderPass.draw(6, viewerState.webgpuPointCount, 0, 0);
  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);
}

function createSplatMaterial(radius) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uResolution: { value: new THREE.Vector2(1, 1) },
      uSplatScale: { value: 1 },
      uOpacityScale: { value: 1 },
    },
    vertexShader: `
      attribute vec3 fdc;
      attribute vec3 sh1;
      attribute vec3 sh2;
      attribute vec3 sh3;
      attribute vec3 gaussianScale;
      attribute vec4 gaussianRotation;
      attribute float gaussianAlpha;
      varying vec3 vColor;
      varying float vAlpha;
      varying vec3 vConic;
      varying float vExtent;
      uniform vec2 uResolution;
      uniform float uSplatScale;
      uniform float uOpacityScale;

      vec3 rotateByQuat(vec4 q, vec3 v) {
        q = normalize(q);
        return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
      }

      vec3 shColor(vec3 dir) {
        const float C0 = 0.28209479177387814;
        const float C1 = 0.4886025119029199;
        vec3 color = 0.5 + C0 * fdc;
        color += (-C1 * dir.y) * sh1;
        color += ( C1 * dir.z) * sh2;
        color += (-C1 * dir.x) * sh3;
        return clamp(color, vec3(0.0), vec3(1.0));
      }

      void main() {
        vec3 viewDir = normalize(position - cameraPosition);
        vColor = shColor(viewDir);
        vAlpha = gaussianAlpha * uOpacityScale;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        float dist = max(-mvPosition.z, 0.001);

        vec3 scale = clamp(exp(gaussianScale) * uSplatScale, vec3(0.00002), vec3(0.018));
        vec4 quat = vec4(gaussianRotation.y, gaussianRotation.z, gaussianRotation.w, gaussianRotation.x);
        vec3 axisX = (modelViewMatrix * vec4(rotateByQuat(quat, vec3(scale.x, 0.0, 0.0)), 0.0)).xyz;
        vec3 axisY = (modelViewMatrix * vec4(rotateByQuat(quat, vec3(0.0, scale.y, 0.0)), 0.0)).xyz;
        vec3 axisZ = (modelViewMatrix * vec4(rotateByQuat(quat, vec3(0.0, 0.0, scale.z)), 0.0)).xyz;
        float pixelScale = 0.5 * uResolution.y * projectionMatrix[1][1] / dist;
        vec2 ax = axisX.xy * pixelScale;
        vec2 ay = axisY.xy * pixelScale;
        vec2 az = axisZ.xy * pixelScale;
        float covXX = dot(vec3(ax.x, ay.x, az.x), vec3(ax.x, ay.x, az.x)) + 0.18;
        float covXY = dot(vec3(ax.x, ay.x, az.x), vec3(ax.y, ay.y, az.y));
        float covYY = dot(vec3(ax.y, ay.y, az.y), vec3(ax.y, ay.y, az.y)) + 0.18;
        float mid = 0.5 * (covXX + covYY);
        float root = sqrt(max(0.0, 0.25 * (covXX - covYY) * (covXX - covYY) + covXY * covXY));
        float lambdaMax = max(mid + root, 0.18);
        vExtent = clamp(3.0 * sqrt(lambdaMax), 0.75, 24.0);

        float det = max(covXX * covYY - covXY * covXY, 1e-4);
        vConic = vec3(covYY / det, -covXY / det, covXX / det);
        gl_PointSize = 2.0 * vExtent;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;
      varying vec3 vConic;
      varying float vExtent;

      void main() {
        vec2 p = (gl_PointCoord * 2.0 - 1.0) * vExtent;
        float q = vConic.x * p.x * p.x + 2.0 * vConic.y * p.x * p.y + vConic.z * p.y * p.y;
        if (q > 9.0) discard;
        float alpha = vAlpha * exp(-0.5 * q);
        if (alpha < 0.002) discard;
        gl_FragColor = vec4(vColor, alpha);
      }
    `,
    vertexColors: true,
    transparent: true,
    blending: THREE.NormalBlending,
    depthTest: false,
    depthWrite: false,
  });
}

function updateSplatUniforms() {
  const material = viewerState.splatCloud?.material;
  if (!material?.uniforms) return;
  const size = Number.parseFloat(els.splatSize.value || "1");
  const opacity = Number.parseFloat(els.splatOpacity.value || "1");
  material.uniforms.uSplatScale.value = viewerState.splatBaseScale * Math.max(0.05, size);
  material.uniforms.uOpacityScale.value = Math.max(0.05, opacity);
}

function scheduleSplatSort(delay = 160) {
  if (activeView !== "splat" || !viewerState.splatCloud || !viewerState.camera) return;
  if (splatSortTimer) clearTimeout(splatSortTimer);
  splatSortTimer = setTimeout(() => {
    splatSortTimer = null;
    sortSplatByDepth();
    sortWebGPUByDepth();
    renderWebGPUScene();
  }, delay);
}

function sortSplatByDepth() {
  const cloud = viewerState.splatCloud;
  const camera = viewerState.camera;
  const geometry = cloud?.geometry;
  const position = geometry?.getAttribute("position");
  if (!cloud || !camera || !geometry || !position) return;

  cloud.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const viewMatrix = new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, cloud.matrixWorld);
  const point = new THREE.Vector3();
  const order = new Array(position.count);
  for (let i = 0; i < position.count; i += 1) {
    point.fromBufferAttribute(position, i).applyMatrix4(viewMatrix);
    order[i] = { index: i, depth: point.z };
  }
  order.sort((a, b) => a.depth - b.depth);

  const indices = new Uint32Array(order.length);
  for (let i = 0; i < order.length; i += 1) {
    indices[i] = order[i].index;
  }
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.index.needsUpdate = true;
  viewerState.splatSortVersion += 1;
  els.renderPreviewStatus.textContent = `WebGPU Gaussian sorted (${position.count.toLocaleString()} points).`;
}

async function prepareSplatMotion() {
  if (!viewerState.runId || !viewerState.splatCloud || viewerState.splatMotionData || viewerState.splatMotionLoading) return;
  viewerState.splatMotionLoading = true;
  els.renderPreviewStatus.textContent = "Preparing dynamic splat motion...";
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(viewerState.runId)}/viewer/splat_motion`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_stride: 1,
        max_frames: viewerState.frameCount || 360,
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    const meta = await response.json();
    const motionResponse = await fetch(`${meta.motion_url}?t=${Date.now()}`);
    if (!motionResponse.ok) throw new Error(await motionResponse.text());
    const buffer = await motionResponse.arrayBuffer();
    const valuesPerPoint = Number(meta.values_per_point || 3);
    const expectedValues = Number(meta.point_count) * Number(meta.frame_count) * valuesPerPoint;
    const data = new Uint16Array(buffer);
    if (data.length < expectedValues) {
      throw new Error(`Motion buffer is incomplete (${data.length}/${expectedValues}).`);
    }
    viewerState.splatMotionMeta = meta;
    viewerState.splatMotionData = data;
    const bounds = computeSplatMotionBounds(data, meta, 0);
    viewerState.splatMotionCenter = bounds.center;
    viewerState.splatMotionRadius = bounds.radius;
    viewerState.currentSplatMotionFrame = -1;
    applySplatMotionFrame(currentViewerFrame(), true);
    resetSplatCameraToMotion();
    els.renderPreviewStatus.textContent = `Dynamic splat ready (${meta.frame_count} frames).`;
  } catch (error) {
    els.renderPreviewStatus.textContent = `Dynamic splat failed: ${error.message || error}`;
  } finally {
    viewerState.splatMotionLoading = false;
  }
}

function applySplatMotionFrame(frame, force = false) {
  const meta = viewerState.splatMotionMeta;
  const data = viewerState.splatMotionData;
  const cloud = viewerState.splatCloud;
  const position = cloud?.geometry?.getAttribute("position");
  if (!meta || !data || !position) return;

  const stride = Math.max(1, Number(meta.frame_stride || 1));
  const frameIndex = Math.min(
    Math.max(0, Math.floor(frame / stride)),
    Math.max(0, Number(meta.frame_count || 1) - 1),
  );
  if (!force && frameIndex === viewerState.currentSplatMotionFrame) return;

  const pointCount = Math.min(Number(meta.point_count || 0), position.count);
  const valuesPerPoint = Number(meta.values_per_point || 3);
  const srcOffset = frameIndex * Number(meta.point_count || 0) * valuesPerPoint;
  const dst = position.array;
  const scale = cloud.geometry.getAttribute("gaussianScale");
  const rotation = cloud.geometry.getAttribute("gaussianRotation");
  const center = viewerState.splatMotionCenter || [0, 0, 0];
  const webgpu = viewerState.webgpuData;
  for (let i = 0; i < pointCount; i += 1) {
    const src = srcOffset + i * valuesPerPoint;
    const dstIndex = i * 3;
    const x = halfToFloat(data[src + 0]) - center[0];
    const y = halfToFloat(data[src + 1]) - center[1];
    const z = halfToFloat(data[src + 2]) - center[2];
    dst[dstIndex + 0] = x;
    dst[dstIndex + 1] = y;
    dst[dstIndex + 2] = z;
    if (valuesPerPoint >= 10 && scale && rotation) {
      const sx = halfToFloat(data[src + 3]);
      const sy = halfToFloat(data[src + 4]);
      const sz = halfToFloat(data[src + 5]);
      scale.setXYZ(i, sx, sy, sz);
      rotation.setXYZW(
        i,
        halfToFloat(data[src + 6]),
        halfToFloat(data[src + 7]),
        halfToFloat(data[src + 8]),
        halfToFloat(data[src + 9]),
      );
    }
  }
  if (webgpu?.sourceIndices) {
    for (let j = 0; j < webgpu.sourceIndices.length; j += 1) {
      const srcPoint = webgpu.sourceIndices[j];
      if (srcPoint >= pointCount) continue;
      const src = srcOffset + srcPoint * valuesPerPoint;
      const dst3 = j * 3;
      const x = halfToFloat(data[src + 0]) - center[0];
      const y = halfToFloat(data[src + 1]) - center[1];
      const z = halfToFloat(data[src + 2]) - center[2];
      webgpu.positions[dst3 + 0] = x;
      webgpu.positions[dst3 + 1] = y;
      webgpu.positions[dst3 + 2] = z;
      if (valuesPerPoint >= 10 && webgpu.scales) {
        const sx = halfToFloat(data[src + 3]);
        const sy = halfToFloat(data[src + 4]);
        const sz = halfToFloat(data[src + 5]);
        webgpu.scales[dst3 + 0] = sx;
        webgpu.scales[dst3 + 1] = sy;
        webgpu.scales[dst3 + 2] = sz;
        webgpu.radii[j] = Math.max(Math.min(Math.exp(Math.max(sx, sy, sz)) * 1.75, 0.028), 0.0012);
      }
    }
  }
  position.needsUpdate = true;
  if (valuesPerPoint >= 10) {
    if (scale) scale.needsUpdate = true;
    if (rotation) rotation.needsUpdate = true;
  }
  viewerState.currentSplatMotionFrame = frameIndex;
  viewerState.webgpuDirty = true;
  if (activeView === "splat") {
    scheduleSplatSort(20);
    renderWebGPUScene();
  }
}

function computeSplatMotionBounds(data, meta, frameIndex) {
  const pointCount = Number(meta.point_count || 0);
  const valuesPerPoint = Number(meta.values_per_point || 3);
  const srcOffset = Math.min(Math.max(0, frameIndex), Math.max(0, Number(meta.frame_count || 1) - 1)) * pointCount * valuesPerPoint;
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < pointCount; i += 1) {
    const src = srcOffset + i * valuesPerPoint;
    const x = halfToFloat(data[src + 0]);
    const y = halfToFloat(data[src + 1]);
    const z = halfToFloat(data[src + 2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }
  if (!xs.length) {
    return { center: [0, 0, 0], radius: 1 };
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const lo = Math.floor(xs.length * 0.04);
  const hi = Math.max(lo, Math.floor(xs.length * 0.96));
  const minX = xs[lo];
  const minY = ys[lo];
  const minZ = zs[lo];
  const maxX = xs[hi];
  const maxY = ys[hi];
  const maxZ = zs[hi];
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const radius = Math.max(
    Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2,
    0.01,
  );
  return { center, radius };
}

function resetSplatCameraToMotion() {
  if (!viewerState.camera || !viewerState.controls || activeView !== "splat") return;
  const radius = Math.max(viewerState.splatMotionRadius || viewerState.splatCloud?.geometry?.boundingSphere?.radius || 1, 0.01);
  const cameraDistance = Math.max(radius * 1.7, 0.2);
  viewerState.defaultCameraDistance = cameraDistance;
  viewerState.splatBaseScale = 1;
  updateSplatUniforms();
  viewerState.camera.position.set(0, radius * 0.08, cameraDistance);
  viewerState.camera.near = Math.max(radius / 200, 0.0005);
  viewerState.camera.far = Math.max(radius * 30, 5);
  viewerState.camera.updateProjectionMatrix();
  viewerState.controls.target.set(0, 0, 0);
  viewerState.controls.update();
  scheduleSplatSort(30);
  renderWebGPUScene();
}

function fitCameraToObject(object, distanceMultiplier = 2.6) {
  if (!viewerState.camera || !viewerState.controls || !object?.geometry) return;
  object.geometry.computeBoundingSphere();
  const sphere = object.geometry.boundingSphere;
  const radius = Math.max(sphere?.radius || 1, 0.01);
  const center = sphere?.center || new THREE.Vector3();
  const cameraDistance = Math.max(radius * distanceMultiplier, 0.25);
  viewerState.defaultCameraDistance = cameraDistance;
  viewerState.camera.position.set(center.x, center.y + radius * 0.1, center.z + cameraDistance);
  viewerState.camera.near = Math.max(radius / 200, 0.0005);
  viewerState.camera.far = Math.max(radius * 30, 5);
  viewerState.camera.updateProjectionMatrix();
  viewerState.controls.target.copy(center);
  viewerState.controls.update();
  if (activeView === "splat") renderWebGPUScene();
}

function fitCurrentSubject() {
  const object = activeView === "splat" ? viewerState.splatCloud : viewerState.pointCloud;
  const position = object?.geometry?.getAttribute("position");
  if (!viewerState.camera || !viewerState.controls || !position) return;
  const bounds = robustBoundsFromPosition(position, 0.12, 0.88);
  const radius = Math.max(bounds.radius, 0.01);
  const cameraDistance = Math.max(radius * 1.6, 0.18);
  viewerState.defaultCameraDistance = cameraDistance;
  viewerState.camera.position.set(bounds.center.x, bounds.center.y + radius * 0.08, bounds.center.z + cameraDistance);
  viewerState.camera.near = Math.max(radius / 200, 0.0005);
  viewerState.camera.far = Math.max(radius * 40, 5);
  viewerState.camera.updateProjectionMatrix();
  viewerState.controls.target.copy(bounds.center);
  viewerState.controls.update();
  if (activeView === "splat") {
    scheduleSplatSort(30);
    renderWebGPUScene();
  } else if (activeView === "rendered") {
    scheduleRenderPreview(0, "idle");
  }
}

function robustBoundsFromPosition(position, low = 0.08, high = 0.92) {
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < position.count; i += 1) {
    const x = position.getX(i);
    const y = position.getY(i);
    const z = position.getZ(i);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    xs.push(x);
    ys.push(y);
    zs.push(z);
  }
  if (!xs.length) {
    return { center: new THREE.Vector3(), radius: 1 };
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const lo = Math.floor(xs.length * low);
  const hi = Math.max(lo, Math.floor(xs.length * high));
  const minX = xs[lo];
  const minY = ys[lo];
  const minZ = zs[lo];
  const maxX = xs[hi];
  const maxY = ys[hi];
  const maxZ = zs[hi];
  return {
    center: new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2),
    radius: Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2, 0.01),
  };
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }
  if (exponent === 31) {
    return fraction ? NaN : sign * Infinity;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function startViewerLoop() {
  if (!viewerState.renderer || viewerState.animationId) return;
  const tick = () => {
    viewerState.animationId = requestAnimationFrame(tick);
    viewerState.controls?.update();
    updateViewerTimeline();
    if (activeView === "splat") {
      if (viewerState.webgpuReady) {
        renderWebGPUScene();
      } else {
        viewerState.renderer.render(viewerState.scene, viewerState.camera);
      }
    } else if (activeView !== "rendered") {
      viewerState.renderer.render(viewerState.scene, viewerState.camera);
    }
  };
  tick();
}

function resizeViewer() {
  if (!viewerState.renderer || els.viewerPanel.hidden) return;
  const rect = els.viewerStage.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  viewerState.renderer.setSize(width, height, false);
  els.viewerGpuCanvas.width = width * Math.min(window.devicePixelRatio || 1, 2);
  els.viewerGpuCanvas.height = height * Math.min(window.devicePixelRatio || 1, 2);
  viewerState.camera.aspect = width / height;
  viewerState.camera.updateProjectionMatrix();
  if (viewerState.splatCloud?.material?.uniforms?.uResolution) {
    viewerState.splatCloud.material.uniforms.uResolution.value.set(width, height);
  }
  if (activeView === "splat") renderWebGPUScene();
  if (activeView === "rendered") scheduleRenderPreview(0, "idle");
}

function toggleViewerPlayback() {
  if (!els.viewerAudio.src) return;
  if (els.viewerAudio.paused) {
    els.finalVideo.pause();
    els.viewerAudio.play();
  } else {
    els.viewerAudio.pause();
  }
}

function resetViewerCamera() {
  if (!viewerState.camera || !viewerState.controls) return;
  if (activeView === "splat" && viewerState.splatMotionData) {
    resetSplatCameraToMotion();
    return;
  }
  fitCameraToObject(activeView === "splat" ? viewerState.splatCloud : viewerState.pointCloud, 2.6);
  if (activeView === "rendered") scheduleRenderPreview(0, "idle");
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function updateViewerTimeline() {
  const duration = Number.isFinite(els.viewerAudio.duration) ? els.viewerAudio.duration : 0;
  const current = Number.isFinite(els.viewerAudio.currentTime) ? els.viewerAudio.currentTime : 0;
  if (duration > 0 && document.activeElement !== els.viewerSeek) {
    els.viewerSeek.value = String(Math.min(1, Math.max(0, current / duration)));
  }
  const frame = Math.min(
    Math.max(0, Math.floor(current * viewerState.fps)),
    Math.max(0, (viewerState.frameCount || Math.floor(duration * viewerState.fps)) - 1),
  );
  const totalFrames = viewerState.frameCount || Math.max(0, Math.floor(duration * viewerState.fps));
  els.viewerTime.textContent = `${formatTime(current)} / ${formatTime(duration)} · frame ${frame}${totalFrames ? `/${totalFrames}` : ""}`;
  if (activeView === "splat" && viewerState.splatMotionData) {
    applySplatMotionFrame(frame);
  }
}

function currentViewerFrame() {
  const current = Number.isFinite(els.viewerAudio.currentTime) ? els.viewerAudio.currentTime : 0;
  const total = viewerState.frameCount || Math.max(1, Math.floor((els.viewerAudio.duration || 0) * viewerState.fps));
  return Math.min(Math.max(0, Math.floor(current * viewerState.fps)), Math.max(0, total - 1));
}

function renderReasonPriority(reason) {
  return { drag: 4, seek: 3, playback: 2, idle: 1 }[reason] || 0;
}

function startRenderPreviewLoop() {
  if (renderPreviewLoopTimer) return;
  const tick = () => {
    if (activeView !== "rendered" || els.viewerAudio.paused) {
      renderPreviewLoopTimer = null;
      return;
    }
    scheduleRenderPreview(0, viewerState.renderPreviewInteracting ? "drag" : "playback");
    renderPreviewLoopTimer = setTimeout(tick, viewerState.renderPreviewInteracting ? 180 : 120);
  };
  tick();
}

function stopRenderPreviewLoop() {
  if (!renderPreviewLoopTimer) return;
  clearTimeout(renderPreviewLoopTimer);
  renderPreviewLoopTimer = null;
}

function getRenderPreviewDimensions(reason = "idle") {
  const rect = els.viewerStage.getBoundingClientRect();
  const baseWidth = Math.max(320, Math.floor(rect.width || 550));
  const baseHeight = Math.max(320, Math.floor(rect.height || 802));
  const scale = reason === "drag" ? 0.72 : reason === "playback" ? 0.82 : 1;
  return {
    width: Math.max(256, Math.min(960, Math.round(baseWidth * scale))),
    height: Math.max(256, Math.min(1280, Math.round(baseHeight * scale))),
  };
}

function getRenderPreviewSignature(camera, frame, width, height) {
  const round = (value, places = 3) => Number(value || 0).toFixed(places);
  return JSON.stringify({
    f: frame,
    w: width,
    h: height,
    p: camera.position.map((v) => round(v)),
    t: camera.target.map((v) => round(v)),
    r: round(camera.radius_norm, 3),
  });
}

function scheduleRenderPreview(delay = 500, reason = "idle") {
  if (activeView !== "rendered" || !viewerState.runId || !viewerState.pointCloud) return;
  if (viewerState.renderPreviewLoading) {
    viewerState.renderPreviewDirty = true;
    if (renderReasonPriority(reason) >= renderReasonPriority(viewerState.renderPreviewDirtyReason)) {
      viewerState.renderPreviewDirtyReason = reason;
    }
    return;
  }
  if (renderPreviewTimer) clearTimeout(renderPreviewTimer);
  if (renderReasonPriority(reason) >= renderReasonPriority(viewerState.pendingRenderReason)) {
    viewerState.pendingRenderReason = reason;
  }
  renderPreviewTimer = setTimeout(() => {
    renderPreviewTimer = null;
    const nextReason = viewerState.pendingRenderReason || reason;
    viewerState.pendingRenderReason = "idle";
    requestRenderedFrame(nextReason);
  }, delay);
}

async function requestRenderedFrame(reason = "idle") {
  if (viewerState.renderPreviewLoading || activeView !== "rendered") return;
  const camera = getViewerCameraPayload();
  if (!camera) return;
  const frame = currentViewerFrame();
  const { width, height } = getRenderPreviewDimensions(reason);
  const signature = getRenderPreviewSignature(camera, frame, width, height);
  if (signature === viewerState.lastRenderedSignature && reason !== "idle") return;
  const requestId = viewerState.renderPreviewRequestId + 1;
  viewerState.renderPreviewRequestId = requestId;
  viewerState.renderPreviewLoading = true;
  viewerState.renderPreviewDirty = false;
  viewerState.renderPreviewDirtyReason = "idle";
  setRenderPreviewLoading(true);
  els.renderPreviewStatus.textContent = `Rendering frame ${frame}${reason === "playback" ? " (playback)" : reason === "drag" ? " (drag)" : ""}...`;

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(viewerState.runId)}/viewer/render_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera,
        frame,
        width,
        height,
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    const frameHeader = Number(response.headers.get("X-Frame") ?? frame);
    const imageBlob = await response.blob();
    const nextSrc = URL.createObjectURL(imageBlob);
    await preloadImage(nextSrc);
    if (requestId !== viewerState.renderPreviewRequestId || activeView !== "rendered") {
      URL.revokeObjectURL(nextSrc);
      return;
    }
    const previousObjectUrl = viewerState.renderPreviewObjectUrl;
    els.renderPreviewImage.classList.add("is-swapping");
    els.renderPreviewImage.hidden = false;
    requestAnimationFrame(() => {
      els.renderPreviewImage.src = nextSrc;
      els.renderPreviewImage.classList.remove("is-swapping");
    });
    viewerState.renderPreviewObjectUrl = nextSrc;
    if (previousObjectUrl) {
      setTimeout(() => URL.revokeObjectURL(previousObjectUrl), 1000);
    }
    viewerState.lastRenderedFrame = frameHeader;
    viewerState.lastRenderedSignature = signature;
    els.renderPreviewStatus.textContent = `Rendered frame ${viewerState.lastRenderedFrame}.`;
  } catch (error) {
    els.renderPreviewStatus.textContent = `Render preview failed: ${error.message || error}`;
  } finally {
    if (requestId === viewerState.renderPreviewRequestId) {
      viewerState.renderPreviewLoading = false;
      setRenderPreviewLoading(false);
      if (viewerState.renderPreviewDirty && activeView === "rendered") {
        const dirtyReason = viewerState.renderPreviewDirtyReason || "idle";
        viewerState.renderPreviewDirtyReason = "idle";
        scheduleRenderPreview(30, dirtyReason);
      }
    }
  }
}

function preloadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = resolve;
    image.onerror = reject;
    image.src = src;
  });
}

function getViewerCameraPayload() {
  if (!viewerState.camera || !viewerState.controls) return null;
  const position = viewerState.camera.position;
  const target = viewerState.controls.target;
  const up = viewerState.camera.up;
  const offset = new THREE.Vector3().subVectors(position, target);
  const distance = Math.max(offset.length(), 1e-6);
  const direction = offset.clone().normalize();
  const radiusNorm = Math.min(2.5, Math.max(0.55, distance / Math.max(viewerState.defaultCameraDistance, 1e-6)));
  return {
    position: [position.x, position.y, position.z],
    target: [target.x, target.y, target.z],
    up: [up.x, up.y, up.z],
    direction: [direction.x, direction.y, direction.z],
    radius_norm: radiusNorm,
    base_radius: 1.0,
    fov: 20,
    near: viewerState.camera.near,
    far: viewerState.camera.far,
    center_offset: viewerState.centerOffset,
  };
}

async function startViewerExport() {
  if (!viewerState.runId || !viewerState.pointCloud) {
    els.viewerExportStatus.textContent = "3D assets are not ready yet.";
    return;
  }
  const camera = getViewerCameraPayload();
  if (!camera) return;

  els.viewerExportBtn.disabled = true;
  els.viewerExportStatus.textContent = "Export queued...";

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(viewerState.runId)}/viewer/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera,
        width: 550,
        height: 802,
        fps: viewerState.fps || 25,
        render_mode: "gaussian",
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    const record = await response.json();
    pollViewerExport(record.export_id);
  } catch (error) {
    els.viewerExportBtn.disabled = false;
    els.viewerExportStatus.textContent = `Export failed to start: ${error.message || error}`;
  }
}

function pollViewerExport(exportId) {
  if (viewerState.exportPollTimer) clearInterval(viewerState.exportPollTimer);
  const tick = async () => {
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(viewerState.runId)}/viewer/export/${encodeURIComponent(exportId)}`);
      if (!response.ok) throw new Error(await response.text());
      const record = await response.json();
      if (record.status === "done" && record.output_video_url) {
        clearInterval(viewerState.exportPollTimer);
        viewerState.exportPollTimer = null;
        els.viewerExportBtn.disabled = false;
        showExportedVideo(record.output_video_url);
        els.viewerExportStatus.innerHTML = `Export done and loaded: <a href="${record.output_video_url}" target="_blank" rel="noreferrer">custom_view.mp4</a>`;
      } else if (record.status === "failed") {
        clearInterval(viewerState.exportPollTimer);
        viewerState.exportPollTimer = null;
        els.viewerExportBtn.disabled = false;
        els.viewerExportStatus.textContent = "Export failed. Check server logs for details.";
      } else {
        els.viewerExportStatus.textContent = `Export ${record.status || "running"}...`;
      }
    } catch (error) {
      clearInterval(viewerState.exportPollTimer);
      viewerState.exportPollTimer = null;
      els.viewerExportBtn.disabled = false;
      els.viewerExportStatus.textContent = `Export status failed: ${error.message || error}`;
    }
  };
  tick();
  viewerState.exportPollTimer = setInterval(tick, 3000);
}

function showExportedVideo(url) {
  viewerState.exportedVideoUrl = url;
  els.viewerAudio.pause();
  els.finalVideo.src = `${url}?t=${Date.now()}`;
  els.finalVideo.load();
  setActiveView("video");
  const playPromise = els.finalVideo.play();
  if (playPromise?.catch) {
    playPromise.catch(() => {
      els.viewerExportStatus.innerHTML = `Export done: <a href="${url}" target="_blank" rel="noreferrer">custom_view.mp4</a>`;
    });
  }
  addDownloadLink("custom_view.mp4", url);
}

function addDownloadLink(label, url) {
  const existing = Array.from(els.downloads.querySelectorAll("a")).find((link) => link.href === new URL(url, window.location.origin).href);
  if (existing) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  a.textContent = label;
  els.downloads.prepend(a);
}
