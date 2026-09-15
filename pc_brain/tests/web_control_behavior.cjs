const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const matches = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
assert.ok(matches.length, "inline script is required");
const source = matches.map((match) => match[1]).join("\n").replace(/\binitialize\(\);\s*$/, "");

class FakeClassList {
  constructor() { this.values = new Set(); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  toggle(value, force) {
    const enabled = force === undefined ? !this.values.has(value) : Boolean(force);
    enabled ? this.values.add(value) : this.values.delete(value);
    return enabled;
  }
  contains(value) { return this.values.has(value); }
}

class FakeElement {
  constructor(id = "") {
    this.id = id;
    this.textContent = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.dataset = {};
    this.style = {};
    this.classList = new FakeClassList();
    this.children = [];
    this.listeners = {};
    this.parentElement = {clientWidth: 640, clientHeight: 360};
    this.clientWidth = 640;
    this.clientHeight = 360;
    this.naturalWidth = 640;
    this.naturalHeight = 360;
    this.scrollHeight = 0;
    this.scrollTop = 0;
    this._innerHTML = "";
  }
  set innerHTML(value) {
    this._innerHTML = String(value);
    const span = this._innerHTML.match(/<span[^>]*>([\s\S]*?)<\/span>/i);
    if (span) {
      this._span = new FakeElement();
      this._span.textContent = span[1].replace(/<[^>]+>/g, "");
    }
  }
  get innerHTML() { return this._innerHTML; }
  get firstElementChild() { return this.children[0] || null; }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  removeEventListener(type, listener) {
    this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== listener);
  }
  emit(type, event = {}) { for (const listener of this.listeners[type] || []) listener(event); }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  remove() {
    if (this.parentElement?.children) this.parentElement.children = this.parentElement.children.filter((item) => item !== this);
  }
  querySelector(selector) {
    if (selector === "span") return this._span || null;
    if (selector === ".mode-empty") return null;
    return null;
  }
  querySelectorAll() { return []; }
  setAttribute(name, value) { this[name] = String(value); }
  getAttribute(name) { return this[name] ?? null; }
  insertAdjacentHTML(_position, value) {
    const child = new FakeElement();
    child.innerHTML = value;
    this.appendChild(child);
  }
  animate() {}
  getContext() {
    return {
      clearRect() {}, setTransform() {}, setLineDash() {}, strokeRect() {}, fillText() {},
      strokeStyle: "", fillStyle: "", lineWidth: 1, font: "",
    };
  }
}

const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) elements.set(id, new FakeElement(id));
  return elements.get(id);
};
element("clock").textContent = "12:00:00";
element("command").value = "";

const documentListeners = {};
const document = {
  hidden: false,
  visibilityState: "visible",
  getElementById: element,
  createElement: () => new FakeElement(),
  querySelector: (selector) => selector.startsWith('[data-panel="') ? element(selector) : null,
  querySelectorAll: () => [],
  addEventListener(type, listener) { (documentListeners[type] ||= []).push(listener); },
};

let nextTimer = 1;
const intervals = new Map();
const timeouts = new Map();
const setIntervalFake = (callback, delay) => {
  const id = nextTimer++;
  intervals.set(id, {callback, delay});
  return id;
};
const clearIntervalFake = (id) => intervals.delete(id);
const setTimeoutFake = (callback, delay) => {
  const id = nextTimer++;
  timeouts.set(id, {callback, delay});
  return id;
};
const clearTimeoutFake = (id) => timeouts.delete(id);

class FakeWebSocket {
  static OPEN = 1;
  constructor(url) {
    if (FakeWebSocket.constructorError) {
      const error = FakeWebSocket.constructorError;
      FakeWebSocket.constructorError = null;
      throw error;
    }
    this.url = url;
    this.readyState = 0;
    this.listeners = {};
    FakeWebSocket.instances.push(this);
  }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  emit(type, event = {}) { for (const listener of this.listeners[type] || []) listener(event); }
  send() {}
  close() { this.readyState = 3; }
}
FakeWebSocket.instances = [];
FakeWebSocket.constructorError = null;

class FakeAudioNode {
  constructor() {
    this.port = {};
    this.gain = {value: 1};
  }
  connect() {}
  disconnect() {}
}

class FakeAudioContext {
  constructor() {
    this.currentTime = 0;
    this.destination = new FakeAudioNode();
    this.audioWorklet = {addModule: async () => {}};
  }
  async resume() {}
  async close() {}
  createMediaStreamSource() { return new FakeAudioNode(); }
  createGain() { return new FakeAudioNode(); }
}

class FakeAudioWorkletNode extends FakeAudioNode {}

const windowListeners = {};
const sandbox = {
  console,
  document,
  window: {
    addEventListener(type, listener) { (windowListeners[type] ||= []).push(listener); },
    AudioContext: FakeAudioContext,
  },
  navigator: {mediaDevices: {getUserMedia: async () => ({getTracks: () => []})}},
  Intl,
  Date,
  Math,
  Number,
  String,
  Array,
  Object,
  Promise,
  Error,
  TypeError,
  JSON,
  URL: {createObjectURL: () => "blob:test", revokeObjectURL() {}},
  Blob,
  AbortController,
  Uint8Array,
  Int16Array,
  Float32Array,
  DataView,
  performance: {now: () => 10},
  devicePixelRatio: 1,
  btoa: (value) => Buffer.from(value, "binary").toString("base64"),
  atob: (value) => Buffer.from(value, "base64").toString("binary"),
  setInterval: setIntervalFake,
  clearInterval: clearIntervalFake,
  setTimeout: setTimeoutFake,
  clearTimeout: clearTimeoutFake,
  WebSocket: FakeWebSocket,
  AudioWorkletNode: FakeAudioWorkletNode,
  fetch: async () => { throw new Error("unexpected fetch"); },
};
sandbox.window.document = document;
sandbox.window.setInterval = setIntervalFake;
sandbox.window.clearInterval = clearIntervalFake;
sandbox.window.setTimeout = setTimeoutFake;
sandbox.window.clearTimeout = clearTimeoutFake;

vm.createContext(sandbox);
vm.runInContext(`${source}\n;globalThis.__api={state,applyTelemetry,refreshTelemetry,startTelemetryPolling,stopTelemetryPolling,handleVisibilityChange,fetchCameraFrame,stopCameraLoop,showCameraError,applyTrackingStatus,renderTrackingOverlay,refreshDiagnostics,sendText,connectRealtime,disconnectRealtime,setMode,overrideRealtimeAudio:value=>setupRealtimeAudio=value,overrideCameraDecoder:value=>loadCameraBlob=value};`, sandbox, {filename: "index.html"});
const api = sandbox.__api;
const flushMicrotasks = async () => { await Promise.resolve(); await Promise.resolve(); };

function telemetrySample(index, rtt = 18.5) {
  return {
    sampled_at: `2026-09-13T00:00:${String(index).padStart(2, "0")}+00:00`,
    sequence: index,
    host: {
      cpu_percent: 12.5,
      memory: {used_bytes: 4 * 1024 ** 3, total_bytes: 8 * 1024 ** 3, percent: 50},
      process: {cpu_percent: 3.5, rss_bytes: 256 * 1024 ** 2},
      event_loop_lag_ms: 2.25,
      network: {rx_bytes: 1000, tx_bytes: 2000, rx_bytes_per_second: index, tx_bytes_per_second: index * 2},
      gpu: {available: true, name: "RTX Test", utilization_percent: 25, memory_used_bytes: 2 * 1024 ** 3, memory_total_bytes: 4 * 1024 ** 3, reason: null},
    },
    llm: {
      status: "ready", provider: "local", model: "test-model",
      last_probe_at: "2026-09-13T00:00:00+00:00", last_probe_latency_ms: 8.5, last_probe_error: null,
      last_inference_at: "2026-09-13T00:00:01+00:00", last_inference_latency_ms: 22.5,
      last_inference_error: null, last_inference_success: true,
    },
    robot: {
      ok: true, status: "online", connected: true, ready: true, source: "tcp", sample_age_ms: 25,
      movement: "stop", speed: 0,
      head: {actual: {pan: 91, tilt: 89}, target: {pan: 90, tilt: 90}},
      eyes: {actual: "neutral", requested: "content", override_reason: null},
      firmware: {
        version: "1.2.3", protocol: 1, uptime_ms: 123456, ip: "192.0.2.2", hostname: "robit",
        wifi: {mode: "sta", rssi: -55},
        memory: {heap_free_bytes: 100, heap_min_free_bytes: 80, heap_total_bytes: 200, psram_free_bytes: 300, psram_total_bytes: 400},
        camera_enabled: true,
      },
      watchdog: {armed: true, fault: false, control_last_receive_age_ms: 4},
      control: {round_trip_ms: rtt, reconnect_count: 2, error: null},
    },
    tracking: {available: true, active: true, latest_frame_age_seconds: 0.5, latest_result_age_seconds: 0.25, error: null},
    faults: [],
  };
}

api.applyTelemetry(telemetrySample(1));
assert.equal(element("hostCpu").textContent, "12.5%");
assert.match(element("gpuVramValue").textContent, /2\.00 GIB \/ 4\.00 GIB.*50\.0%/i);
assert.equal(element("llmStatus").textContent, "READY");
assert.equal(element("robotStatus").textContent, "ONLINE");
assert.equal(element("robotRtt").textContent, "18.5 MS");
assert.equal(element("trackingTelemetryStatus").textContent, "ACTIVE");
assert.equal(element("networkChartState").textContent, "WAITING / NEED 2 SAMPLES");

api.applyTelemetry(telemetrySample(2));
assert.equal(element("networkChartState").textContent, "LIVE / 2 SAMPLES");
assert.notEqual(element("networkRxPath").getAttribute?.("d") || element("networkRxPath").d || "", "");

for (let index = 3; index <= 65; index += 1) api.applyTelemetry(telemetrySample(index, index % 4 === 0 ? null : index));
assert.equal(api.state.telemetry.networkHistory.length, 60);
assert.equal(api.state.telemetry.rttHistory.length, 60);
assert.ok(api.state.telemetry.rttHistory.some((sample) => sample.value === null), "missing RTT samples must remain gaps");

const unavailable = telemetrySample(66, null);
unavailable.host.gpu = {available: false, name: null, utilization_percent: null, memory_used_bytes: null, memory_total_bytes: null, reason: "not installed"};
unavailable.host.event_loop_lag_ms = null;
unavailable.llm.status = "unavailable";
unavailable.llm.last_probe_latency_ms = null;
unavailable.robot.status = "offline";
unavailable.robot.head.actual.pan = null;
unavailable.tracking.latest_result_age_seconds = null;
api.applyTelemetry(unavailable);
assert.equal(element("gpuVramValue").textContent, "N/A");
assert.equal(element("hostLoopLag").textContent, "N/A");
assert.equal(element("llmStatus").textContent, "UNAVAILABLE");
assert.equal(element("robotHeadActual").textContent, "N/A / 89°");
assert.match(element("trackingTelemetryStatus").textContent, /ACTIVE/);

(async () => {
  api.stopTelemetryPolling();
  intervals.clear();
  const fetches = [];
  let resolveFetch;
  sandbox.fetch = (path) => {
    fetches.push(path);
    return new Promise((resolve) => { resolveFetch = resolve; });
  };
  const first = api.startTelemetryPolling();
  const overlap = api.refreshTelemetry();
  assert.deepEqual(fetches, ["/system/telemetry"]);
  const oneSecondIntervals = [...intervals.values()].filter((timer) => timer.delay === 1000);
  assert.equal(oneSecondIntervals.length, 1);
  resolveFetch({ok: true, json: async () => telemetrySample(70)});
  await Promise.all([first, overlap]);

  document.hidden = true;
  document.visibilityState = "hidden";
  api.handleVisibilityChange();
  assert.equal(api.state.telemetry.timer, null);
  assert.equal([...intervals.values()].filter((timer) => timer.delay === 1000).length, 0);

  document.hidden = false;
  document.visibilityState = "visible";
  sandbox.fetch = async (path) => {
    fetches.push(path);
    return {ok: true, json: async () => telemetrySample(71)};
  };
  await api.handleVisibilityChange();
  assert.deepEqual(fetches.slice(-1), ["/system/telemetry"]);
  assert.equal([...intervals.values()].filter((timer) => timer.delay === 1000).length, 1);

  api.stopTelemetryPolling();
  intervals.clear();
  const pendingTelemetry = [];
  sandbox.fetch = (path, options = {}) => new Promise((resolve) => {
    pendingTelemetry.push({path, signal: options.signal, resolve});
  });
  const staleTelemetry = api.refreshTelemetry();
  assert.equal(pendingTelemetry.length, 1);
  document.hidden = true;
  document.visibilityState = "hidden";
  api.handleVisibilityChange();
  assert.equal(pendingTelemetry[0].signal?.aborted, true, "hiding must abort the unresolved telemetry request");

  document.hidden = false;
  document.visibilityState = "visible";
  const freshTelemetry = api.handleVisibilityChange();
  assert.equal(pendingTelemetry.length, 2, "visibility regain must launch a fresh telemetry request immediately");
  pendingTelemetry[1].resolve({ok: true, json: async () => telemetrySample(73)});
  await freshTelemetry;
  assert.equal(api.state.telemetry.latest.sequence, 73);
  pendingTelemetry[0].resolve({ok: true, json: async () => telemetrySample(72)});
  await staleTelemetry;
  assert.equal(api.state.telemetry.latest.sequence, 73, "a late hidden-generation response must not overwrite visible data");

  api.stopTelemetryPolling();
  intervals.clear();
  const realDateNow = Date.now;
  const sampleTimeMs = Date.parse("2026-09-13T12:00:00.000Z");
  Date.now = () => sampleTimeMs + 3000;
  const freshnessSample = telemetrySample(74);
  freshnessSample.sampled_at = new Date(sampleTimeMs).toISOString();
  api.applyTelemetry(freshnessSample);
  assert.equal(element("telemetryStatus").textContent, "ONLINE");
  let stalledRequestAborted = false;
  sandbox.fetch = (_path, options = {}) => new Promise((_resolve, reject) => {
    options.signal?.addEventListener("abort", () => {
      stalledRequestAborted = true;
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    });
  });
  const stalledTelemetry = api.refreshTelemetry();
  await flushMicrotasks();
  assert.equal(api.state.telemetry.inFlight, true);

  Date.now = () => sampleTimeMs + 3001;
  await api.refreshTelemetry();
  assert.equal(element("telemetryStatus").textContent, "STALE", "a pending request must not freeze freshness at ONLINE");
  Date.now = () => sampleTimeMs + 15000;
  await api.refreshTelemetry();
  assert.equal(element("telemetryStatus").textContent, "STALE");
  Date.now = () => sampleTimeMs + 15001;
  await api.refreshTelemetry();
  assert.equal(element("telemetryStatus").textContent, "OFFLINE");

  const requestTimeoutId = api.state.telemetry.requestTimeout;
  assert.ok(requestTimeoutId && timeouts.has(requestTimeoutId), "visible telemetry fetches must have a timeout");
  timeouts.get(requestTimeoutId).callback();
  await stalledTelemetry;
  assert.equal(stalledRequestAborted, true);
  assert.equal(api.state.telemetry.inFlight, false);
  Date.now = realDateNow;

  let rejectChat;
  sandbox.fetch = () => new Promise((_resolve, reject) => { rejectChat = reject; });
  const failed = api.sendText("move left", false);
  assert.equal(element("chatSubmit").disabled, true);
  assert.match(element("chatStatus").textContent, /^SENDING \/ 0\.0 S$/);
  const duplicate = await api.sendText("move left", false);
  assert.equal(duplicate, undefined);
  rejectChat(new Error("unsafe <tag>"));
  await failed;
  assert.equal(element("chatSubmit").disabled, false);
  assert.equal(element("command").value, "move left");
  assert.match(element("chatStatus").textContent, /^FAULT \/ RETRY READY$/);

  element("command").value = "move left";
  sandbox.fetch = async () => ({ok: true, json: async () => ({response: "stopped", parse_error: null, action_result: null, vision: null})});
  await api.sendText("move left", false);
  assert.equal(element("chatSubmit").disabled, false);
  assert.equal(element("chatStatus").textContent, "READY");
  assert.equal(element("command").value, "");

  element("command").value = "move left";
  sandbox.fetch = async () => ({ok: true, json: async () => ({
    response: "moving now",
    parse_error: null,
    action_result: {ok: false, error: "control acknowledgement rejected"},
    vision: null,
  })});
  await api.sendText("move left", false);
  assert.equal(element("chatStatus").textContent, "FAULT / RETRY READY");
  assert.equal(element("command").value, "move left", "failed actuation must retain the prompt");
  const failedActionMessage = element("transcript").children.at(-1).querySelector("span").textContent;
  assert.match(failedActionMessage, /^FAULT \/ /);
  assert.doesNotMatch(failedActionMessage, /moving now/i);

  element("command").value = "look left";
  sandbox.fetch = async () => ({ok: true, json: async () => ({
    response: "done",
    parse_error: null,
    execution_error: "robot action failed; retry",
    action_result: null,
    vision: null,
  })});
  await api.sendText("look left", false);
  assert.equal(element("chatStatus").textContent, "FAULT / RETRY READY");
  assert.equal(element("command").value, "look left", "explicit execution errors must retain the prompt");

  api.state.realtime.url = "ws://localhost/v1/realtime";
  api.state.mode = "voice";
  FakeWebSocket.instances.length = 0;
  const microphoneRequests = [];
  sandbox.navigator.mediaDevices.getUserMedia = () => new Promise((resolve) => microphoneRequests.push(resolve));
  let disconnectedPendingTrackStops = 0;
  const disconnectedPending = api.connectRealtime();
  await flushMicrotasks();
  assert.equal(microphoneRequests.length, 1);
  api.disconnectRealtime();
  microphoneRequests[0]({getTracks: () => [{stop() { disconnectedPendingTrackStops += 1; }}]});
  await disconnectedPending;
  assert.equal(disconnectedPendingTrackStops, 1, "disconnect must retire and dispose the pending microphone attempt");
  assert.equal(FakeWebSocket.instances.length, 0, "a retired connection attempt must not construct a socket");
  assert.equal(api.state.realtime.ws, null);
  assert.equal(api.state.realtime.connected, false);

  let leftVoiceTrackStops = 0;
  api.state.mode = "voice";
  const leftVoicePending = api.connectRealtime();
  await flushMicrotasks();
  assert.equal(microphoneRequests.length, 2);
  api.setMode("telemetry");
  microphoneRequests[1]({getTracks: () => [{stop() { leftVoiceTrackStops += 1; }}]});
  await leftVoicePending;
  assert.equal(leftVoiceTrackStops, 1, "leaving Voice must dispose its pending microphone attempt");
  assert.equal(FakeWebSocket.instances.length, 0);

  let retiredTrackStops = 0;
  let currentTrackStops = 0;
  api.state.mode = "voice";
  const retiredConnect = api.connectRealtime();
  await flushMicrotasks();
  assert.equal(microphoneRequests.length, 3);
  api.disconnectRealtime();
  const currentConnect = api.connectRealtime();
  await flushMicrotasks();
  assert.equal(microphoneRequests.length, 4);
  microphoneRequests[3]({getTracks: () => [{stop() { currentTrackStops += 1; }}]});
  await currentConnect;
  const currentSocket = FakeWebSocket.instances.at(-1);
  assert.ok(currentSocket, "the replacement attempt must own the current socket");
  microphoneRequests[2]({getTracks: () => [{stop() { retiredTrackStops += 1; }}]});
  await retiredConnect;
  assert.equal(retiredTrackStops, 1, "the superseded attempt must dispose only its own stream");
  assert.equal(currentTrackStops, 0, "the superseded attempt must not stop the replacement stream");
  assert.equal(api.state.realtime.ws, currentSocket);
  api.disconnectRealtime();
  assert.equal(currentTrackStops, 1);

  api.state.realtime.url = "ws://localhost/v1/realtime";
  api.state.mode = "voice";
  let setupTrackStops = 0;
  let setupCaptureDisconnects = 0;
  let setupSourceDisconnects = 0;
  let setupSinkDisconnects = 0;
  api.overrideRealtimeAudio(async () => {
    api.state.realtime.audioContext = {currentTime: 0};
    api.state.realtime.micStream = {getTracks: () => [{stop() { setupTrackStops += 1; }}]};
    api.state.realtime.captureNode = {disconnect() { setupCaptureDisconnects += 1; }};
    api.state.realtime.micSource = {disconnect() { setupSourceDisconnects += 1; }};
    api.state.realtime.captureSink = {disconnect() { setupSinkDisconnects += 1; }};
  });
  FakeWebSocket.constructorError = new Error("websocket construction failed");
  await assert.rejects(api.connectRealtime(), /websocket construction failed/);
  assert.equal(setupTrackStops, 1, "connect failure after microphone acquisition must stop its track");
  assert.equal(setupCaptureDisconnects, 1, "connect failure must disconnect the capture node");
  assert.equal(setupSourceDisconnects, 1, "connect failure must disconnect the microphone source");
  assert.equal(setupSinkDisconnects, 1, "connect failure must disconnect the capture sink");
  assert.equal(api.state.realtime.micStream, null);
  assert.equal(api.state.realtime.captureNode, null);
  assert.equal(api.state.realtime.micSource, null);
  assert.equal(api.state.realtime.captureSink, null);
  assert.equal(api.state.realtime.ws, null);
  assert.equal(api.state.realtime.connecting, false);
  assert.equal(api.state.realtime.connected, false);
  assert.equal(element("voiceConnect").textContent, "CONNECT VOICE");
  assert.equal(element("voiceStatus").textContent, "WEBSOCKET CONSTRUCTION FAILED");

  api.overrideRealtimeAudio(async () => { api.state.realtime.audioContext = {currentTime: 0}; });
  let trackStops = 0;
  let captureDisconnects = 0;
  let sourceDisconnects = 0;
  let sinkDisconnects = 0;
  api.state.realtime.micStream = {getTracks: () => [{stop() { trackStops += 1; }}]};
  api.state.realtime.captureNode = {disconnect() { captureDisconnects += 1; }};
  api.state.realtime.micSource = {disconnect() { sourceDisconnects += 1; }};
  api.state.realtime.captureSink = {disconnect() { sinkDisconnects += 1; }};
  await api.connectRealtime();
  const socket = FakeWebSocket.instances.at(-1);
  socket.emit("error", {});
  assert.equal(api.state.realtime.connecting, false);
  assert.equal(api.state.realtime.connected, false);
  assert.equal(api.state.realtime.reconnectTimer, null);
  assert.match(element("voiceStatus").textContent, /CONNECTION ERROR/);
  assert.equal(trackStops, 1, "terminal socket error must stop microphone tracks");
  assert.equal(captureDisconnects, 1, "terminal socket error must disconnect capture node");
  assert.equal(sourceDisconnects, 1, "terminal socket error must disconnect microphone source");
  assert.equal(sinkDisconnects, 1, "terminal socket error must disconnect capture sink");
  assert.equal(api.state.realtime.micStream, null);
  assert.equal(api.state.realtime.captureNode, null);
  assert.equal(api.state.realtime.micSource, null);
  assert.equal(api.state.realtime.captureSink, null);
  socket.emit("close", {code: 1006});
  assert.equal(api.state.realtime.connecting, false);
  assert.equal(api.state.realtime.reconnectTimer, null);
  assert.equal(element("voiceConnect").textContent, "CONNECT VOICE");
  assert.equal(trackStops, 1, "error followed by close must not stop tracks twice");
  assert.equal(captureDisconnects, 1, "error followed by close must not disconnect capture twice");
  assert.equal(sourceDisconnects, 1, "error followed by close must not disconnect source twice");
  assert.equal(sinkDisconnects, 1, "error followed by close must not disconnect sink twice");

  const revokedCameraUrls = [];
  let nextCameraUrl = 1;
  sandbox.URL.createObjectURL = () => `blob:camera-${nextCameraUrl++}`;
  sandbox.URL.revokeObjectURL = (url) => revokedCameraUrls.push(url);
  const decodeRequests = [];
  api.overrideCameraDecoder((url) => new Promise((resolve, reject) => {
    decodeRequests.push({url, resolve, reject});
  }));
  const cameraResponses = [];
  sandbox.fetch = async () => cameraResponses.shift();
  const cameraResponse = (frameId) => ({
    ok: true,
    headers: {get(name) {
      if (name === "X-Robit-Frame-Id") return frameId;
      if (name === "X-Robit-Frame-Interval") return "5";
      return null;
    }},
    blob: async () => ({frameId}),
  });
  const settleCameraAwaitPoints = async () => {
    for (let index = 0; index < 6; index += 1) await Promise.resolve();
  };

  document.hidden = false;
  document.visibilityState = "visible";
  api.state.camera = {capture_url: "/robot/camera/capture"};
  api.state.cameraGeneration += 1;
  const retiredCameraGeneration = api.state.cameraGeneration;
  cameraResponses.push(cameraResponse("frame-retired-success"));
  const retiredCameraFetch = api.fetchCameraFrame(retiredCameraGeneration);
  await settleCameraAwaitPoints();
  assert.equal(decodeRequests.length, 1);

  api.stopCameraLoop();
  const currentCameraGeneration = api.state.cameraGeneration;
  cameraResponses.push(cameraResponse("frame-current"));
  const currentCameraFetch = api.fetchCameraFrame(currentCameraGeneration);
  await settleCameraAwaitPoints();
  assert.equal(decodeRequests.length, 2);
  const currentCameraUrl = decodeRequests[1].url;
  decodeRequests[1].resolve();
  await currentCameraFetch;
  assert.equal(api.state.cameraFrameId, "frame-current");
  assert.equal(element("cameraStream").src, currentCameraUrl, "decoded pixels and frame identity must publish together");

  const retiredCameraUrl = decodeRequests[0].url;
  decodeRequests[0].resolve();
  await retiredCameraFetch;
  assert.equal(api.state.cameraFrameId, "frame-current", "a stale decoded frame must not replace the new frame identity");
  assert.equal(element("cameraStream").src, currentCameraUrl, "a stale decoded frame must not replace the visible pixels");
  assert.ok(revokedCameraUrls.includes(retiredCameraUrl), "a retired decoded object URL must be revoked");

  api.state.cameraGeneration += 1;
  cameraResponses.push(cameraResponse("frame-retired-error"));
  const retiredErrorFetch = api.fetchCameraFrame(api.state.cameraGeneration);
  await settleCameraAwaitPoints();
  const retiredErrorUrl = decodeRequests[2].url;
  document.hidden = true;
  document.visibilityState = "hidden";
  api.handleVisibilityChange();
  decodeRequests[2].reject(new Error("retired decode failed"));
  await retiredErrorFetch;
  assert.equal(api.state.cameraFrameId, "frame-current", "a hidden-generation error must not erase the current frame identity");
  assert.equal(element("cameraStream").hidden, false, "a hidden-generation error must not erase the current pixels");
  assert.ok(revokedCameraUrls.includes(retiredErrorUrl), "a retired failed-decode URL must be revoked");
  document.hidden = false;
  document.visibilityState = "visible";

  api.state.camera = {capture_url: "/robot/camera/capture"};
  sandbox.fetch = async (path) => {
    const payloads = {
      "/health": {
        ready: false,
        components: {llm: {status: "ready"}, camera: {status: "degraded", reason: "capture failed"}},
        event_loop: {lag_ms: 1}, journal: {status: "ok"}, tracking: {backend: "cpu"},
        voice_health: {status: "ok", degraded_reasons: []},
      },
      "/robot/status": {status: "online", source: "tcp", movement: "stop", head: {actual: {pan: 90, tilt: 90}}},
      "/tracking/status": {available: true, enabled: false, state: "off", backend: "cpu"},
    };
    return {ok: true, json: async () => payloads[path]};
  };
  await api.refreshDiagnostics();
  assert.match(element("visionReadout").innerHTML, /CAMERA[\s\S]*DEGRADED/);
  assert.doesNotMatch(element("visionReadout").innerHTML, /CAMERA[\s\S]*READY/);

  const trackingStatus = {
    available: true,
    enabled: true,
    state: "tracking",
    target: {frame_id: "frame-7", bounding_box: [0.1, 0.2, 0.5, 0.7], track_id: 7, confidence: 0.9},
  };
  api.state.cameraFrameId = "frame-7";
  element("cameraStream").hidden = false;
  api.applyTrackingStatus(trackingStatus);
  assert.equal(element("trackingOverlay").hidden, false);
  api.showCameraError("camera offline");
  assert.equal(element("trackingOverlay").hidden, true);
  api.applyTrackingStatus(trackingStatus);
  assert.equal(element("trackingOverlay").hidden, true, "tracking status must not redraw an overlay after camera failure");
  assert.equal(api.state.cameraFrameId, null, "camera failure must invalidate decoded-frame ownership");

  console.log("web control behavior: ok");
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
