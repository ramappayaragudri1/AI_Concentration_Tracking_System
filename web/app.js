/**
 * AI Concentration Tracking System - Client-Side Computer Vision Engine
 * Powered by MediaPipe FaceMesh & HTML5 Web Audio
 */

// Landmark Indices corresponding to concen_tracker.py
const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];
const LEFT_IRIS_IDX = 468;
const RIGHT_IRIS_IDX = 473;
const NOSE_IDX = 1;

// DOM Elements
const videoElement = document.getElementById('webcam-video');
const canvasElement = document.getElementById('overlay-canvas');
const canvasCtx = canvasElement.getContext('2d');
const placeholder = document.getElementById('camera-placeholder');

const btnToggleCam = document.getElementById('btn-toggle-cam');
const camBtnIcon = document.getElementById('cam-btn-icon');
const camBtnText = document.getElementById('cam-btn-text');
const btnExportLog = document.getElementById('btn-export-log');

const statusChip = document.getElementById('status-chip');
const statusText = document.getElementById('status-text');
const hudFps = document.getElementById('hud-fps');
const hudState = document.getElementById('hud-state');

const meterValue = document.getElementById('meter-value');
const meterFill = document.getElementById('meter-fill');

const metricGaze = document.getElementById('metric-gaze');
const metricGazePill = document.getElementById('metric-gaze-pill');
const metricHead = document.getElementById('metric-head');
const metricHeadPill = document.getElementById('metric-head-pill');
const metricEar = document.getElementById('metric-ear');
const metricBlinkPill = document.getElementById('metric-blink-pill');
const metricDistractions = document.getElementById('metric-distractions');
const metricDistractPill = document.getElementById('metric-distract-pill');
const metricTotalBlinks = document.getElementById('metric-total-blinks');
const metricTimer = document.getElementById('metric-timer');
const metricAvgScore = document.getElementById('metric-avg-score');

const chkDrawMesh = document.getElementById('chk-draw-mesh');
const chkAudioAlert = document.getElementById('chk-audio-alert');
const chkMirrorMode = document.getElementById('chk-mirror-mode');
const toastAlert = document.getElementById('toast-alert');
const toastMessage = document.getElementById('toast-message');

// Tracking State
let isRunning = false;
let camera = null;
let mediaStream = null;
let animFrameId = null;
let faceMesh = null;

let scoreHistory = [];
const SCORE_HISTORY_MAX = 10;
let distractionFrames = 0;
let totalDistractionEvents = 0;
let totalBlinks = 0;
let wasBlinking = false;
let allScoresRecorded = [];

// Session Log for CSV Export
let sessionLogs = [];

// FPS Calculation
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

// Session Timer
let sessionStartTime = null;
let sessionTimerInterval = null;

// Audio Context for synthesized chime
let audioCtx = null;
let lastAudioAlertTime = 0;

// Initialize Chart.js
const chartCanvas = document.getElementById('concentration-chart');
let concentrationChart = null;

function initChart() {
  const ctx = chartCanvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 180);
  gradient.addColorStop(0, 'rgba(0, 242, 254, 0.4)');
  gradient.addColorStop(1, 'rgba(0, 242, 254, 0.0)');

  const initialLabels = Array.from({ length: 30 }, (_, i) => '');
  const initialData = Array(30).fill(null);

  concentrationChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: initialLabels,
      datasets: [{
        label: 'Concentration %',
        data: initialData,
        borderColor: '#00f2fe',
        borderWidth: 2.5,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        pointHitRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { size: 10 } }
        },
        x: {
          display: false
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateChart(score) {
  if (!concentrationChart) return;
  const dataset = concentrationChart.data.datasets[0].data;
  dataset.push(score);
  if (dataset.length > 30) {
    dataset.shift();
  }
  concentrationChart.update();
}

// Distance helper
function distance2D(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// Eye Aspect Ratio (EAR)
function calculateEAR(landmarks, eyePoints, imgW, imgH) {
  const pts = eyePoints.map(idx => ({
    x: landmarks[idx].x * imgW,
    y: landmarks[idx].y * imgH
  }));

  const A = distance2D(pts[1], pts[5]);
  const B = distance2D(pts[2], pts[4]);
  const C = distance2D(pts[0], pts[3]);

  return (A + B) / (2.0 * C);
}

// Head Pose Attentiveness
function calculateHeadPoseScore(landmarks, imgW, imgH) {
  const nose = landmarks[NOSE_IDX];
  const nx = nose.x * imgW;
  const ny = nose.y * imgH;
  const d = Math.hypot(nx - imgW / 2, ny - imgH / 2);

  // If nose stays within 30% of image center, head is oriented well
  return (d < 0.3 * imgW) ? 1.0 : 0.0;
}

// Gaze Attentiveness (Relative to eye corners, independent of where user sits)
function calculateGazeScore(landmarks) {
  const leftIris = landmarks[LEFT_IRIS_IDX];
  const rightIris = landmarks[RIGHT_IRIS_IDX];
  if (!leftIris || !rightIris) return 1.0;

  const lOuter = landmarks[33];
  const lInner = landmarks[133];
  const rInner = landmarks[362];
  const rOuter = landmarks[263];

  const lWidth = Math.abs(lInner.x - lOuter.x) + 1e-6;
  const rWidth = Math.abs(rOuter.x - rInner.x) + 1e-6;

  const lRatio = (leftIris.x - Math.min(lOuter.x, lInner.x)) / lWidth;
  const rRatio = (rightIris.x - Math.min(rInner.x, rOuter.x)) / rWidth;
  const avgRatio = (lRatio + rRatio) / 2.0;

  // Looking forward at the screen yields ratio between 0.32 and 0.68
  if (avgRatio >= 0.30 && avgRatio <= 0.70) {
    return 1.0;
  } else if (avgRatio >= 0.20 && avgRatio <= 0.80) {
    return 0.75;
  }
  return 0.0;
}

// Sound Synthesis using Web Audio API
function playDistractionChime() {
  if (!chkAudioAlert.checked) return;
  const now = Date.now();
  if (now - lastAudioAlertTime < 4000) return; // Prevent alert spam
  lastAudioAlertTime = now;

  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(520, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(380, audioCtx.currentTime + 0.35);

    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.4);
  } catch (err) {
    console.warn("Audio warning couldn't play:", err);
  }
}

// Show Alert Toast
function showToast(msg) {
  toastMessage.textContent = msg;
  toastAlert.classList.add('show');
  setTimeout(() => {
    toastAlert.classList.remove('show');
  }, 3500);
}

// Render Contours & Landmarks
function drawOverlay(landmarks, imgW, imgH, smoothScore) {
  canvasCtx.clearRect(0, 0, imgW, imgH);
  if (!chkDrawMesh.checked) return;

  canvasCtx.save();
  canvasCtx.lineWidth = 1;

  // Draw eye landmarks
  [...LEFT_EYE_IDX, ...RIGHT_EYE_IDX].forEach(idx => {
    const lm = landmarks[idx];
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * imgW, lm.y * imgH, 2, 0, 2 * Math.PI);
    canvasCtx.fillStyle = '#00f2fe';
    canvasCtx.fill();
  });

  // Draw Irises
  [LEFT_IRIS_IDX, RIGHT_IRIS_IDX].forEach(idx => {
    const lm = landmarks[idx];
    if (lm) {
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * imgW, lm.y * imgH, 3.5, 0, 2 * Math.PI);
      canvasCtx.fillStyle = '#10b981';
      canvasCtx.fill();
    }
  });

  // Draw Nose Orientation Marker
  const nose = landmarks[NOSE_IDX];
  canvasCtx.beginPath();
  canvasCtx.arc(nose.x * imgW, nose.y * imgH, 4, 0, 2 * Math.PI);
  canvasCtx.fillStyle = smoothScore > 40 ? '#00f2fe' : '#f43f5e';
  canvasCtx.fill();

  canvasCtx.restore();
}

// Main Results Processing
function onResults(results) {
  const imgW = canvasElement.width = videoElement.videoWidth || 640;
  const imgH = canvasElement.height = videoElement.videoHeight || 480;

  // FPS Tracking
  frameCount++;
  const now = performance.now();
  if (now - lastFrameTime >= 1000) {
    currentFps = (frameCount * 1000) / (now - lastFrameTime);
    hudFps.textContent = `FPS: ${currentFps.toFixed(1)}`;
    frameCount = 0;
    lastFrameTime = now;
  }

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
    const landmarks = results.multiFaceLandmarks[0];

    // 1. EAR (Eye Aspect Ratio)
    const leftEar = calculateEAR(landmarks, LEFT_EYE_IDX, imgW, imgH);
    const rightEar = calculateEAR(landmarks, RIGHT_EYE_IDX, imgW, imgH);
    const avgEar = (leftEar + rightEar) / 2.0;
    const isBlink = avgEar < 0.20;

    metricEar.textContent = avgEar.toFixed(2);
    if (isBlink) {
      metricBlinkPill.textContent = "Blinking";
      metricBlinkPill.className = "metric-pill pill-warn";
      if (!wasBlinking) {
        totalBlinks++;
        metricTotalBlinks.textContent = totalBlinks;
        wasBlinking = true;
      }
    } else {
      metricBlinkPill.textContent = "Eyes Open";
      metricBlinkPill.className = "metric-pill pill-good";
      wasBlinking = false;
    }

    // 2. Gaze Score
    const gazeScore = calculateGazeScore(landmarks);
    if (gazeScore === 1.0) {
      metricGaze.textContent = "Forward";
      metricGazePill.textContent = "Optimal";
      metricGazePill.className = "metric-pill pill-good";
    } else {
      metricGaze.textContent = "Away";
      metricGazePill.textContent = "Looking Aside";
      metricGazePill.className = "metric-pill pill-warn";
    }

    // 3. Head Pose Score
    const headScore = calculateHeadPoseScore(landmarks, imgW, imgH);
    if (headScore === 1.0) {
      metricHead.textContent = "Centered";
      metricHeadPill.textContent = "Screen Aligned";
      metricHeadPill.className = "metric-pill pill-good";
    } else {
      metricHead.textContent = "Off-Center";
      metricHeadPill.textContent = "Head Turned";
      metricHeadPill.className = "metric-pill pill-warn";
    }

    // 4. Composite Concentration Score (Model matching concen_tracker.py)
    const rawScore = 0.4 * gazeScore + 0.4 * headScore + 0.2 * (isBlink ? 0 : 1);
    const scorePercent = Math.round(rawScore * 100);

    // 10-Frame Moving Average
    scoreHistory.push(scorePercent);
    if (scoreHistory.length > SCORE_HISTORY_MAX) {
      scoreHistory.shift();
    }
    const smoothScore = Math.round(scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length);

    allScoresRecorded.push(smoothScore);
    const avgScoreTotal = Math.round(allScoresRecorded.reduce((a, b) => a + b, 0) / allScoresRecorded.length);
    metricAvgScore.textContent = `Avg: ${avgScoreTotal}%`;

    // Update Meter UI
    meterValue.textContent = `${smoothScore}%`;
    meterFill.style.width = `${smoothScore}%`;

    if (smoothScore >= 70) {
      meterValue.style.color = '#10b981';
      meterFill.style.background = 'linear-gradient(90deg, #10b981, #00f2fe)';
    } else if (smoothScore >= 40) {
      meterValue.style.color = '#f59e0b';
      meterFill.style.background = 'linear-gradient(90deg, #f59e0b, #38bdf8)';
    } else {
      meterValue.style.color = '#f43f5e';
      meterFill.style.background = 'linear-gradient(90deg, #ef4444, #f43f5e)';
    }

    // Distraction Tracking
    if (smoothScore < 40) {
      distractionFrames++;
      if (distractionFrames === 60) { // ~2 seconds of low concentration
        totalDistractionEvents++;
        metricDistractions.textContent = totalDistractionEvents;
        metricDistractPill.textContent = "Distracted";
        metricDistractPill.className = "metric-pill pill-alert";

        hudState.textContent = "DISTRACTED";
        hudState.className = "hud-badge state distracted";
        statusChip.className = "status-chip distracted";
        statusText.textContent = "Distraction Detected";

        playDistractionChime();
        showToast("Low concentration detected! Please look at your workspace.");
      }
    } else {
      if (distractionFrames > 0) {
        distractionFrames = 0;
      }
      hudState.textContent = "ACTIVE";
      hudState.className = "hud-badge state active";
      statusChip.className = "status-chip active";
      statusText.textContent = "Tracking Active";
      metricDistractPill.textContent = "Focused";
      metricDistractPill.className = "metric-pill pill-good";
    }

    // Render Canvas Mesh
    drawOverlay(landmarks, imgW, imgH, smoothScore);

    // Update Timeline Chart
    updateChart(smoothScore);

    // Save session telemetry log
    sessionLogs.push({
      time: new Date().toISOString(),
      score: smoothScore,
      ear: avgEar.toFixed(2),
      gaze: gazeScore,
      headPose: headScore,
      blinking: isBlink ? 1 : 0
    });

  } else {
    // No Face in Camera View
    canvasCtx.clearRect(0, 0, imgW, imgH);
    hudState.textContent = "NO FACE";
    hudState.className = "hud-badge state distracted";
    meterValue.textContent = "0%";
    meterFill.style.width = "0%";
    updateChart(0);
  }
}

// Session Timer
function startSessionTimer() {
  sessionStartTime = Date.now();
  metricTimer.textContent = "00:00";
  sessionTimerInterval = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - sessionStartTime) / 1000);
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
    const secs = String(elapsedSeconds % 60).padStart(2, '0');
    metricTimer.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopSessionTimer() {
  if (sessionTimerInterval) {
    clearInterval(sessionTimerInterval);
    sessionTimerInterval = null;
  }
}

// Toggle Tracking
async function startTracking() {
  btnToggleCam.disabled = true;
  statusText.textContent = "Requesting Camera Access...";

  try {
    // 1. Direct WebRTC Camera Request
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Webcam access is not supported by this browser environment. Please use Google Chrome, Edge, or Firefox.");
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });

    videoElement.srcObject = mediaStream;
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });

    placeholder.style.display = 'none';
    statusText.textContent = "Loading AI FaceMesh Model...";

    // 2. Initialize MediaPipe FaceMesh
    if (!faceMesh) {
      faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      faceMesh.onResults(onResults);
    }

    isRunning = true;
    btnToggleCam.disabled = false;
    btnToggleCam.className = 'btn btn-danger';
    camBtnIcon.textContent = '⏹';
    camBtnText.textContent = 'Stop Tracking';

    statusChip.className = 'status-chip active';
    statusText.textContent = 'Tracking Active';
    hudState.textContent = 'ACTIVE';
    hudState.className = 'hud-badge state active';

    startSessionTimer();

    // 3. Continuous Video Frame Processing Loop
    let isProcessing = false;
    async function processLoop() {
      if (!isRunning) return;
      if (!isProcessing && videoElement.readyState >= 2) {
        isProcessing = true;
        try {
          await faceMesh.send({ image: videoElement });
        } catch (e) {
          console.warn("FaceMesh frame processing error:", e);
        }
        isProcessing = false;
      }
      animFrameId = requestAnimationFrame(processLoop);
    }

    animFrameId = requestAnimationFrame(processLoop);

  } catch (err) {
    console.error("Camera initialization error:", err);
    btnToggleCam.disabled = false;
    placeholder.style.display = 'flex';
    statusChip.className = 'status-chip distracted';
    statusText.textContent = "Camera Error";

    if (err.name === 'NotReadableError' || err.name === 'TrackStartError' || err.message?.includes('starting video input')) {
      alert("⚠️ CAMERA IS IN USE BY ANOTHER PROGRAM:\n\nYour webcam is currently locked by another application (like the Python OpenCV window running in your terminal, Zoom, or Teams).\n\nPlease stop the Python script in your terminal (press 'q' or Ctrl+C) to release your webcam, then click Start Tracking again.");
    } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      alert("⚠️ PERMISSION DENIED:\n\nPlease allow camera access in your browser's address bar settings (look for the camera icon next to the URL) and refresh the page.");
    } else {
      alert(`⚠️ Could not start camera: ${err.message}\n\nPlease check that your webcam is connected and enabled.`);
    }
  }
}

function stopTracking() {
  isRunning = false;
  
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => {
      track.stop();
    });
    mediaStream = null;
  }
  videoElement.srcObject = null;

  stopSessionTimer();

  btnToggleCam.className = 'btn btn-primary';
  camBtnIcon.textContent = '▶';
  camBtnText.textContent = 'Start Tracking';

  statusChip.className = 'status-chip';
  statusText.textContent = 'Session Stopped';
  hudState.textContent = 'OFFLINE';
  hudState.className = 'hud-badge state';

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  placeholder.style.display = 'flex';
}

// Mirror toggle
chkMirrorMode.addEventListener('change', () => {
  const transformVal = chkMirrorMode.checked ? 'scaleX(-1)' : 'scaleX(1)';
  videoElement.style.transform = transformVal;
  canvasElement.style.transform = transformVal;
});

// Start/Stop Button Click
btnToggleCam.addEventListener('click', () => {
  if (isRunning) {
    stopTracking();
  } else {
    startTracking();
  }
});

// Export Session Log as CSV
btnExportLog.addEventListener('click', () => {
  if (sessionLogs.length === 0) {
    alert("No session tracking data recorded yet. Start tracking to collect data first.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,Timestamp,ConcentrationScore,EAR,GazeScore,HeadPoseScore,IsBlink\n";
  sessionLogs.forEach(row => {
    csvContent += `${row.time},${row.score},${row.ear},${row.gaze},${row.headPose},${row.blinking}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `concentration_session_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// Initialize Chart on load
window.addEventListener('DOMContentLoaded', () => {
  initChart();
});
