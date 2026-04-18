const POLL_INTERVAL_MS = 250;

const canvas = document.getElementById("map-canvas");
const ctx = canvas.getContext("2d");
const overlayMessage = document.getElementById("overlay-message");
const statusEl = document.getElementById("status");
const poseEl = document.getElementById("pose");
const goalEl = document.getElementById("goal");
const pathEl = document.getElementById("path");
const cursorEl = document.getElementById("cursor");

const uiState = {
  map: null,
  mapVersion: 0,
  mapImage: null,
  pose: null,
  goal: null,
  path: [],
};

let pollInFlight = false;

function formatMeters(value) {
  return `${value.toFixed(2)} m`;
}

function formatPose(pose) {
  if (!pose) {
    return "-";
  }
  return `x=${pose.x.toFixed(2)}, y=${pose.y.toFixed(2)}, yaw=${(
    (pose.yaw * 180) /
    Math.PI
  ).toFixed(1)} deg`;
}

function decodeBase64Bytes(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function buildMapImage(mapData) {
  const imageCanvas = document.createElement("canvas");
  imageCanvas.width = mapData.width;
  imageCanvas.height = mapData.height;
  const imageCtx = imageCanvas.getContext("2d");
  const imageData = imageCtx.createImageData(mapData.width, mapData.height);
  const cells = decodeBase64Bytes(mapData.data);

  for (let gridY = 0; gridY < mapData.height; gridY += 1) {
    const sourceRow = gridY * mapData.width;
    const canvasY = mapData.height - 1 - gridY;
    const targetRow = canvasY * mapData.width;
    for (let gridX = 0; gridX < mapData.width; gridX += 1) {
      const value = cells[sourceRow + gridX];
      const offset = (targetRow + gridX) * 4;
      let shade = 255;
      if (value === 255) {
        shade = 166;
      } else {
        shade = 255 - Math.round((value / 100) * 255);
      }
      imageData.data[offset + 0] = shade;
      imageData.data[offset + 1] = shade;
      imageData.data[offset + 2] = shade;
      imageData.data[offset + 3] = 255;
    }
  }

  imageCtx.putImageData(imageData, 0, 0);
  return imageCanvas;
}

function worldToCanvasPoint(x, y) {
  if (!uiState.map) {
    return null;
  }

  const { origin, resolution, height } = uiState.map;
  const dx = x - origin.x;
  const dy = y - origin.y;
  const cosYaw = Math.cos(-origin.yaw);
  const sinYaw = Math.sin(-origin.yaw);
  const localX = dx * cosYaw - dy * sinYaw;
  const localY = dx * sinYaw + dy * cosYaw;
  return {
    x: localX / resolution,
    y: height - localY / resolution,
  };
}

function canvasToWorldPoint(canvasX, canvasY) {
  const { origin, resolution, height } = uiState.map;
  const localX = canvasX * resolution;
  const localY = (height - canvasY) * resolution;
  const cosYaw = Math.cos(origin.yaw);
  const sinYaw = Math.sin(origin.yaw);
  return {
    x: origin.x + localX * cosYaw - localY * sinYaw,
    y: origin.y + localX * sinYaw + localY * cosYaw,
  };
}

function poseHeadingTip(pose, lengthMeters = 0.6) {
  return worldToCanvasPoint(
    pose.x + Math.cos(pose.yaw) * lengthMeters,
    pose.y + Math.sin(pose.yaw) * lengthMeters
  );
}

function drawPoseMarker(pose, color) {
  if (!pose) {
    return;
  }

  const center = worldToCanvasPoint(pose.x, pose.y);
  const tip = poseHeadingTip(pose);
  if (!center || !tip) {
    return;
  }

  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.arc(center.x, center.y, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(tip.x, tip.y);
  ctx.stroke();
}

function drawPath(path) {
  if (!path || path.length === 0) {
    return;
  }

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.beginPath();
  path.forEach(([x, y], index) => {
    const point = worldToCanvasPoint(x, y);
    if (!point) {
      return;
    }
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.stroke();
}

function redraw() {
  if (!uiState.map || !uiState.mapImage) {
    return;
  }

  canvas.width = uiState.map.width;
  canvas.height = uiState.map.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(uiState.mapImage, 0, 0);
  drawPath(uiState.path);
  drawPoseMarker(uiState.goal, "#f59e0b");
  drawPoseMarker(uiState.pose, "#22c55e");
}

function updateStatus() {
  const hasMap = Boolean(uiState.map);
  const hasPose = Boolean(uiState.pose);
  const hasPath = uiState.path.length > 0;

  if (!hasMap) {
    statusEl.textContent = "Waiting for /map";
    overlayMessage.hidden = false;
    overlayMessage.innerHTML = "Waiting for <code>/map</code>...";
  } else if (!hasPose) {
    statusEl.textContent = "Map ready, waiting for /localization/pose";
    overlayMessage.hidden = false;
    overlayMessage.innerHTML = "Map ready. Waiting for <code>/localization/pose</code>...";
  } else {
    statusEl.textContent = hasPath
      ? "Ready to click goals"
      : "Pose ready, waiting for a goal";
    overlayMessage.hidden = true;
  }

  poseEl.textContent = formatPose(uiState.pose);
  goalEl.textContent = formatPose(uiState.goal);
  pathEl.textContent = hasPath ? `${uiState.path.length} waypoints` : "No planned path";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const errorPayload = await response.json();
      if (errorPayload.message) {
        message = errorPayload.message;
      }
    } catch (error) {
      // Ignore JSON parse errors for non-JSON responses.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }
  return response.json();
}

async function ensureMap(mapVersion) {
  if (uiState.map && uiState.mapVersion === mapVersion) {
    return;
  }

  const mapData = await fetchJson("/api/map");
  uiState.map = mapData;
  uiState.mapVersion = mapData.version;
  uiState.mapImage = buildMapImage(mapData);
}

async function pollState() {
  if (pollInFlight) {
    return;
  }
  pollInFlight = true;

  try {
    const snapshot = await fetchJson("/api/state");
    if (snapshot.ready && snapshot.ready.map && snapshot.mapVersion > 0) {
      await ensureMap(snapshot.mapVersion);
    }
    uiState.pose = snapshot.pose;
    uiState.goal = snapshot.goal;
    uiState.path = snapshot.path || [];
    updateStatus();
    redraw();
  } catch (error) {
    statusEl.textContent = `Connection error: ${error.message}`;
    overlayMessage.hidden = false;
    overlayMessage.textContent = "Waiting for planner web server...";
  } finally {
    pollInFlight = false;
  }
}

function eventToCanvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

async function handleCanvasClick(event) {
  if (!uiState.map) {
    return;
  }

  const canvasPoint = eventToCanvasCoordinates(event);
  const goalPoint = canvasToWorldPoint(canvasPoint.x, canvasPoint.y);
  const goalYaw = uiState.pose
    ? Math.atan2(goalPoint.y - uiState.pose.y, goalPoint.x - uiState.pose.x)
    : 0.0;

  try {
    const result = await fetchJson("/api/goal", {
      method: "POST",
      body: JSON.stringify({
        x: goalPoint.x,
        y: goalPoint.y,
        yaw: goalYaw,
      }),
    });
    uiState.goal = result.goal;
    updateStatus();
    redraw();
  } catch (error) {
    statusEl.textContent = `Goal publish failed: ${error.message}`;
  }
}

function handleCanvasMove(event) {
  if (!uiState.map) {
    cursorEl.textContent = "-";
    return;
  }

  const canvasPoint = eventToCanvasCoordinates(event);
  const worldPoint = canvasToWorldPoint(canvasPoint.x, canvasPoint.y);
  cursorEl.textContent = `x=${worldPoint.x.toFixed(2)}, y=${worldPoint.y.toFixed(2)}`;
}

canvas.addEventListener("click", handleCanvasClick);
canvas.addEventListener("mousemove", handleCanvasMove);

pollState();
window.setInterval(pollState, POLL_INTERVAL_MS);
