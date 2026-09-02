// --- UTILS ---
(function() {
  const originalLog = console.log.bind(console);
  console.log = (...args) => {
    originalLog(...args);
  };
})();

// --- UI ELEMENTS ---
const startBtn = document.getElementById("startBtn");
const orbContainer = document.getElementById("orbContainer");
const liveText = document.getElementById("liveText");
const transcriptLabel = document.getElementById("transcriptLabel");
const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");
const messagesList = document.getElementById("messagesList");
const resetBtn = document.getElementById("resetBtn");
const visualizerCanvas = document.getElementById("visualizerCanvas");
const canvasCtx = visualizerCanvas.getContext("2d");

// Mobile Sidebar
const menuBtn = document.getElementById("menuBtn");
const sidebar = document.getElementById("sidebar");
const closeSidebarBtn = document.getElementById("closeSidebarBtn");

if (menuBtn) {
  menuBtn.onclick = () => { sidebar.classList.add("open"); closeSidebarBtn.style.display="block"; };
}
if (closeSidebarBtn) {
  closeSidebarBtn.onclick = () => { sidebar.classList.remove("open"); closeSidebarBtn.style.display="none"; };
}

// --- STATE ---
let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;
let isTTSPlaying = false;
let isRecording = false;
let ignoreIncomingTTS = false;

// --- VISUALIZER VARIABLES ---
let analyser = null;
let dataArray = null;
let animationId = null;

// --- BATCHING ---
const BATCH_SAMPLES = 2048;
const HEADER_BYTES  = 8;
const MESSAGE_BYTES = HEADER_BYTES + (BATCH_SAMPLES * 2);
const bufferPool = [];
let batchBuffer = null;
let batchView = null;
let batchInt16 = null;
let batchOffset = 0;

// --- INITIALIZATION ---
function resizeCanvas() {
  if (!visualizerCanvas || !orbContainer) return;
  visualizerCanvas.width = orbContainer.offsetWidth * 2;
  visualizerCanvas.height = orbContainer.offsetHeight * 2;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// --- RESET LOGIC ---
resetBtn.onclick = () => {
  messagesList.innerHTML = `<div class="log-entry system"><span class="log-role">SYSTEM</span>Context Cleared.</div>`;
  updateLiveText("System Reset", "SYSTEM");
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'clear_history' }));
  }
};

// --- MAIN TOGGLE ---
startBtn.onclick = async () => {
  if (!isRecording) {
    await startConnection();
  } else {
    stopConnection();
  }
};

// --- CONNECTION LOGIC ---
async function startConnection() {
  if (socket && socket.readyState === WebSocket.OPEN) return;
  
  updateLiveText("Initializing Uplink...", "SYSTEM");
  orbContainer.classList.add("active");
  document.body.classList.add("session-active");
  
  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${wsProto}//${location.host}/ws`);

  socket.onopen = async () => {
    isRecording = true;
    setStatus("ONLINE", "#00f0ff");
    updateLiveText("100x.inc Listening...", "SYSTEM"); // FIXED NAME
    
    await startAudioSystem();
    
    visualizerCanvas.style.opacity = "0.8";
    drawVisualizer();
  };

  socket.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        handleJSONMessage(msg);
      } catch (e) { console.error(e); }
    }
  };

  socket.onclose = () => {
    stopUIState();
    setStatus("OFFLINE", "#666");
  };

  socket.onerror = (err) => {
    console.error(err);
    stopUIState();
    updateLiveText("Connection Failure", "ERROR");
  };
}

function stopConnection() {
  if (socket) socket.close();
  cleanupAudio();
  stopUIState();
}

function stopUIState() {
  isRecording = false;
  orbContainer.classList.remove("active");
  orbContainer.classList.remove("speaking");
  document.body.classList.remove("session-active");
  visualizerCanvas.style.opacity = "0";
  if (animationId) cancelAnimationFrame(animationId);
  setStatus("OFFLINE", "#666");
  updateLiveText("Initialize Sequence", "READY");
}

function setStatus(text, color) {
  statusText.textContent = text;
  statusText.style.color = color;
  statusDot.style.background = color;
  if (text === "ONLINE") statusDot.classList.add("blink");
  else statusDot.classList.remove("blink");
}

// --- AUDIO SYSTEM ---
function initBatch() {
  if (!batchBuffer) {
    batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
    batchView = new DataView(batchBuffer);
    batchInt16 = new Int16Array(batchBuffer, HEADER_BYTES);
    batchOffset = 0;
  }
}

function flushBatch() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const ts = Date.now() & 0xFFFFFFFF;
  batchView.setUint32(0, ts, false);
  const flags = isTTSPlaying ? 1 : 0;
  batchView.setUint32(4, flags, false);
  socket.send(batchBuffer);
  bufferPool.push(batchBuffer);
  batchBuffer = null;
}

async function startAudioSystem() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 24000 },
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
    mediaStream = stream;
    
    if (!audioContext) audioContext = new AudioContext();
    
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    await audioContext.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');
    micWorkletNode.port.onmessage = ({ data }) => {
      const incoming = new Int16Array(data);
      let read = 0;
      while (read < incoming.length) {
        initBatch();
        const toCopy = Math.min(incoming.length - read, BATCH_SAMPLES - batchOffset);
        batchInt16.set(incoming.subarray(read, read + toCopy), batchOffset);
        batchOffset += toCopy;
        read += toCopy;
        if (batchOffset === BATCH_SAMPLES) flushBatch();
      }
    };
    source.connect(micWorkletNode);

    await audioContext.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
    ttsWorkletNode = new AudioWorkletNode(audioContext, 'tts-playback-processor');
    ttsWorkletNode.port.onmessage = (event) => {
      const { type } = event.data;
      if (type === 'ttsPlaybackStarted') {
        if (!isTTSPlaying && socket) {
          isTTSPlaying = true;
          orbContainer.classList.add("speaking");
          socket.send(JSON.stringify({ type: 'tts_start' }));
        }
      } else if (type === 'ttsPlaybackStopped') {
        if (isTTSPlaying) {
          isTTSPlaying = false;
          ignoreIncomingTTS = false;
          orbContainer.classList.remove("speaking");
        }
      }
    };
    ttsWorkletNode.connect(audioContext.destination);

  } catch (err) {
    console.error(err);
    updateLiveText("Access Denied", "ERROR");
  }
}

function cleanupAudio() {
  if (micWorkletNode) { micWorkletNode.disconnect(); micWorkletNode = null; }
  if (ttsWorkletNode) { ttsWorkletNode.disconnect(); ttsWorkletNode = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (mediaStream) { mediaStream.getAudioTracks().forEach(t => t.stop()); mediaStream = null; }
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i);
  return new Int16Array(buf);
}

// --- UI UPDATES ---

function updateLiveText(text, label) {
  transcriptLabel.textContent = label || "SYSTEM";
  transcriptLabel.style.opacity = 1;
  
  liveText.style.opacity = 0;
  liveText.style.transform = "translateY(10px)";
  
  setTimeout(() => {
    liveText.textContent = text;
    liveText.classList.add("visible");
    liveText.style.opacity = 1;
    liveText.style.transform = "translateY(0)";
  }, 100);
}

function addLogEntry(role, content) {
  const div = document.createElement("div");
  div.className = `log-entry ${role}`;
  div.innerHTML = `<span class="log-role">${role.toUpperCase()}</span>${content}`;
  messagesList.appendChild(div);
  messagesList.scrollTop = messagesList.scrollHeight;
}

function handleJSONMessage({ type, content }) {
  if (type === "partial_user_request") {
    updateLiveText(content, "USER INPUT");
  }
  else if (type === "final_user_request") {
    addLogEntry("user", content);
    updateLiveText("Processing...", "100x.inc"); // FIXED NAME
  }
  else if (type === "partial_assistant_answer") {
    updateLiveText(content, "100x.inc"); // FIXED NAME
  }
  else if (type === "final_assistant_answer") {
    addLogEntry("assistant", content);
  }
  else if (type === "tts_chunk") {
    if (ignoreIncomingTTS || !ttsWorkletNode) return;
    const int16 = base64ToInt16Array(content);
    ttsWorkletNode.port.postMessage(int16);
  }
  else if (type === "stop_tts") {
    if (ttsWorkletNode) ttsWorkletNode.port.postMessage({ type: "clear" });
    isTTSPlaying = false;
    ignoreIncomingTTS = true;
    orbContainer.classList.remove("speaking");
    socket.send(JSON.stringify({ type: 'tts_stop' }));
  }
}

// --- SCI-FI VISUALIZER ---
function drawVisualizer() {
  if (!isRecording) return;
  animationId = requestAnimationFrame(drawVisualizer);

  if (!analyser) return;
  analyser.getByteFrequencyData(dataArray);

  const width = visualizerCanvas.width;
  const height = visualizerCanvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = 160;

  canvasCtx.clearRect(0, 0, width, height);
  
  const gradient = canvasCtx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "rgba(0, 240, 255, 0.1)");
  gradient.addColorStop(0.5, "rgba(0, 240, 255, 0.8)");
  gradient.addColorStop(1, "rgba(112, 0, 255, 0.8)");

  canvasCtx.beginPath();
  
  const bars = 60; 
  const step = Math.floor(dataArray.length / bars);

  for (let i = 0; i < bars; i++) {
    const amplitude = dataArray[i * step];
    const angle = (i / bars) * Math.PI * 2;
    const barHeight = (amplitude / 255) * 70; 
    
    const x = centerX + Math.cos(angle) * (radius + barHeight);
    const y = centerY + Math.sin(angle) * (radius + barHeight);
    
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
  }
  
  const firstAmp = dataArray[0];
  const firstHeight = (firstAmp / 255) * 70;
  canvasCtx.lineTo(centerX + Math.cos(0) * (radius + firstHeight), centerY + Math.sin(0) * (radius + firstHeight));

  canvasCtx.strokeStyle = gradient;
  canvasCtx.lineWidth = 3;
  canvasCtx.lineJoin = 'round'; 
  canvasCtx.stroke();
  
  canvasCtx.beginPath();
  canvasCtx.arc(centerX, centerY, radius - 10, 0, Math.PI * 2);
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  canvasCtx.lineWidth = 1;
  canvasCtx.stroke();
}