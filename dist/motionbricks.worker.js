const G1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint"
];
const ROOT_SIZE = 7;
const POSE_SIZE = ROOT_SIZE + G1_JOINT_NAMES.length;
const SAMPLE_RATE = 30;
const POSES_PER_TOKEN = 4;
const BOUNDARY_FRAMES = POSES_PER_TOKEN - 1;
const MIN_TOKENS$1 = 6;
const MAX_TOKENS$1 = 16;
const MAX_RAW_FRAMES = MAX_TOKENS$1 * POSES_PER_TOKEN;
const ROOT_POSITION_FILTER = [-0.0465, 0.0639, 0.2827, 0.3998, 0.2827, 0.0639, -0.0465];
const JOINT_ANGLE_FADE_SECONDS = 0.3;
const DEFAULT_TOKEN_COUNTS = [8, 9, 10, 11, 12, 13, 14];
function qmul$1(a, b) {
  const [ax, ay, az, aw] = [a[0], a[1], a[2], a[3]];
  const [bx, by, bz, bw] = [b[0], b[1], b[2], b[3]];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz
  ];
}
function qinv$1(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}
function qnorm$1(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qrot$1(q, v) {
  const [x, y, z, w] = [q[0], q[1], q[2], q[3]];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + y * tz - z * ty,
    v[1] + w * ty + z * tx - x * tz,
    v[2] + w * tz + x * ty - y * tx
  ];
}
function qz(angle) {
  return [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];
}
function qpow(q, t) {
  let [x, y, z, w] = qnorm$1(q);
  if (w < 0) [x, y, z, w] = [-x, -y, -z, -w];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0, 0, 1];
  const angle = 2 * Math.atan2(s, w) * t;
  const k = Math.sin(angle / 2) / s;
  return [x * k, y * k, z * k, Math.cos(angle / 2)];
}
function heading(q) {
  const f = qrot$1(qnorm$1(q), [1, 0, 0]);
  return Math.atan2(f[1], f[0]);
}
function fadeWeight(v) {
  const t = Math.min(Math.max(v, 0), 1);
  const r = 1 - t;
  return t < 0.5 ? 1 - 2 * t * t : 2 * r * r;
}
async function setupOrtWeb(ort, wasm) {
  ort.env.logLevel = "error";
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  if (!wasm?.length) return;
  const parts = await Promise.all(wasm.map(async (u) => fetchBytes(u)));
  const binary = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((offset, p) => (binary.set(p, offset), offset + p.length), 0);
  ort.env.wasm.wasmBinary = binary.buffer;
}
async function fetchBytes(url, onBytes, target, offset = 0, cache) {
  const hit = cache ? await cache.match(url) : void 0;
  const res = hit ?? await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download ${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const parts = [];
  let size = 0;
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    if (target) target.set(value, offset + size);
    else parts.push(value);
    size += value.length;
    onBytes?.(value.length);
  }
  let out;
  if (target) out = target.subarray(offset, offset + size);
  else {
    out = new Uint8Array(size);
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
  }
  if (cache && !hit) cache.store(url, out);
  return out;
}
class PackCache {
  constructor(_cache, _base) {
    this._cache = _cache;
    this._base = _base;
    this._writes = Promise.resolve();
  }
  static async Open(name, base, manifestText) {
    try {
      if (typeof caches === "undefined") return null;
      const cache = await caches.open(name);
      const stored = await cache.match(base + "pack.json");
      if (stored && await stored.text() !== manifestText) {
        for (const key of await cache.keys()) if (key.url.startsWith(base)) await cache.delete(key);
      }
      return new PackCache(cache, base);
    } catch {
      return null;
    }
  }
  async match(url) {
    return this._cache.match(url).catch(() => void 0);
  }
  /**
   * Store a downloaded file from the buffer it was read into, one at a time, rather than teeing the download:
   * a tee holds whatever the disk has not taken yet, which on a phone can be the whole pack. A full disk only
   * means it downloads again next time.
   */
  store(url, bytes) {
    this._writes = this._writes.then(async () => this._cache.put(url, new Response(bytes)).catch(() => void 0));
  }
  /** Wait for the stores, before the weights are handed to the session and can be collected. */
  async flush() {
    return this._writes;
  }
  async complete(manifestText) {
    await this._writes;
    await this._cache.put(this._base + "pack.json", new Response(manifestText, { headers: [["content-type", "application/json"]] })).catch(() => void 0);
  }
}
async function fetchManifest(base, cacheName) {
  try {
    const res = await fetch(base + "pack.json", { cache: "no-cache" });
    if (res.ok) return await res.text();
    throw new Error(`Could not download ${base}pack.json: HTTP ${res.status}`);
  } catch (e) {
    const stored = cacheName && typeof caches !== "undefined" ? await caches.open(cacheName).then(async (c) => c.match(base + "pack.json")).catch(() => void 0) : void 0;
    if (stored) return stored.text();
    throw e;
  }
}
const MOTIONBRICKS_CACHE = "three-motionbricks";
function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1;
  const e = h >> 10 & 31;
  const m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function dequantizePack(manifest, q8) {
  const out = new Float32Array(manifest.f32Length);
  const int8 = new Int8Array(q8.buffer, q8.byteOffset, q8.byteLength);
  const view = new DataView(q8.buffer, q8.byteOffset, q8.byteLength);
  const block = manifest.block;
  for (const t of manifest.tensors) {
    for (let b = 0; b < t.blocks; b++) {
      const scale = halfToFloat(view.getUint16(t.s + b * 2, true));
      const start = b * block;
      const end = Math.min(start + block, t.n);
      for (let i = start; i < end; i++) out[t.f + i] = int8[t.q + i] * scale;
    }
  }
  return out;
}
class MotionBricksG1 {
  constructor(ort, session, backend = "default") {
    this._busy = false;
    this.ort = ort;
    this.session = session;
    this.backend = backend;
  }
  /**
   * Load a model packed by tools/pack_model.py: a graph, a manifest, and int8 weight shards.
   * @param baseUrl folder that holds pack.json, ending with a slash
   * @param cacheName Cache Storage to keep the pack in, so it downloads once per browser; false to always download
   */
  static async LoadPacked(ort, baseUrl, sessionOptions = {}, onProgress, cacheName = MOTIONBRICKS_CACHE) {
    const manifestText = await fetchManifest(baseUrl, cacheName);
    const manifest = JSON.parse(manifestText);
    const cache = cacheName ? await PackCache.Open(cacheName, baseUrl, manifestText) : null;
    const engine2 = manifest.format === "motionbricks-external" ? await this._loadLite(ort, baseUrl, manifest, sessionOptions, onProgress, cache) : await this._loadQ8(ort, baseUrl, manifest, sessionOptions, onProgress, cache);
    await cache?.complete(manifestText);
    return engine2;
  }
  static async _loadQ8(ort, baseUrl, manifest, sessionOptions, onProgress, cache) {
    if (manifest.format !== "motionbricks-q8") throw new Error("Unknown model pack format");
    let loaded = 0;
    const total = manifest.q8Length;
    const count = (n) => {
      loaded += n;
      onProgress?.(loaded, total);
    };
    const q8 = new Uint8Array(manifest.q8Length);
    const [graph] = await Promise.all([
      fetchBytes(baseUrl + (manifest.graph ?? "graph.onnx"), void 0, void 0, 0, cache),
      ...manifest.shards.map(async (s, i) => fetchBytes(baseUrl + s, count, q8, i * manifest.shardBytes, cache))
    ]);
    await cache?.flush();
    const weights = dequantizePack(manifest, q8);
    return this._create(ort, graph, {
      ...sessionOptions,
      externalData: [{ path: manifest.weights, data: new Uint8Array(weights.buffer) }]
    });
  }
  /** A pack from tools/pack_lite.py: the weights go to the session as they are, with no fp32 copy. */
  static async _loadLite(ort, baseUrl, manifest, sessionOptions, onProgress, cache) {
    let loaded = 0;
    const count = (n) => onProgress?.(loaded += n, manifest.bytes);
    const data = new Uint8Array(manifest.bytes);
    const [graph] = await Promise.all([
      fetchBytes(baseUrl + manifest.graph, void 0, void 0, 0, cache),
      ...manifest.shards.map(async (s, i) => fetchBytes(baseUrl + s, count, data, i * manifest.shardBytes, cache))
    ]);
    await cache?.flush();
    return this._create(ort, graph, {
      enableCpuMemArena: false,
      enableMemPattern: false,
      ...sessionOptions,
      externalData: [{ path: manifest.weights, data }]
    });
  }
  /**
   * Load a pack folder from tools/pack_model.py or tools/pack_lite.py (a URL ending in /), or an original .onnx
   * file. Packs are kept in Cache Storage under `cacheName` (false: always download); .onnx files are not.
   */
  static async Load(ort, model, sessionOptions = {}, onProgress, cacheName = MOTIONBRICKS_CACHE) {
    return model.endsWith("/") ? this.LoadPacked(ort, model, sessionOptions, onProgress, cacheName) : this.LoadOnnx(ort, model, sessionOptions, onProgress);
  }
  /** Load an original MotionBricks G1 .onnx file, such as mmm_v6_g1_fp32.onnx. */
  static async LoadOnnx(ort, url, sessionOptions = {}, onProgress) {
    let loaded = 0;
    const res = await fetch(url, { method: "HEAD" }).catch(() => null);
    const total = Number(res?.headers.get("content-length")) || 0;
    const bytes = await fetchBytes(url, (n) => onProgress?.(loaded += n, total));
    return this._create(ort, bytes, sessionOptions);
  }
  /**
   * Create the session with one execution provider at a time, in the given order, so the backend
   * that runs is known and a WebGPU failure falls back to WebAssembly.
   */
  static async _create(ort, model, sessionOptions) {
    const providers = sessionOptions.executionProviders ?? ["wasm"];
    for (let i = 0; ; i++) {
      try {
        const session = await ort.InferenceSession.create(model, { ...sessionOptions, executionProviders: [providers[i]] });
        return new this(ort, session, providers[i]);
      } catch (e) {
        if (i >= providers.length - 1) throw e;
        console.warn(`MotionBricks: ${providers[i]} failed, trying ${providers[i + 1]}`, e);
      }
    }
  }
  get busy() {
    return this._busy;
  }
  /**
   * Generate motion from four recent poses toward one to four target poses, all in world space.
   * The model picks the duration from the allowed token counts.
   */
  async generate(context, target, options = {}) {
    if (context.length !== POSES_PER_TOKEN) throw new Error("Context needs exactly 4 poses");
    if (target.length < 1 || target.length > POSES_PER_TOKEN) throw new Error("Target needs 1 to 4 poses");
    if (this._busy) throw new Error("A generation is already running");
    const tokens = options.tokens ?? DEFAULT_TOKEN_COUNTS;
    const numTargets = target.length;
    const targets = [0, 1, 2, 3].map((i) => target[Math.min(i, numTargets - 1)]);
    const last = context[POSES_PER_TOKEN - 1];
    const ap = [last[0], last[1], 0];
    const aq = qz(heading([last[3], last[4], last[5], last[6]]));
    const aqi = qinv$1(aq);
    const encode = (poses2) => {
      const out = new Float32Array(POSES_PER_TOKEN * POSE_SIZE);
      poses2.forEach((p, i) => {
        const o = i * POSE_SIZE;
        for (let k = 0; k < POSE_SIZE; k++) out[o + k] = p[k];
        const pos = qrot$1(aqi, [p[0] - ap[0], p[1] - ap[1], p[2] - ap[2]]);
        out[o] = pos[0];
        out[o + 1] = pos[1];
        out.set(qnorm$1(qmul$1(aqi, qnorm$1([p[3], p[4], p[5], p[6]]))), o + 3);
      });
      return out;
    };
    const has = new Int32Array(8);
    const hasLocalRoot = new Int32Array(8);
    for (let i = 0; i < 8; i++) {
      has[i] = i < POSES_PER_TOKEN + numTargets ? 1 : 0;
      hasLocalRoot[i] = i === POSES_PER_TOKEN - 1 || i === POSES_PER_TOKEN + numTargets - 1 ? 0 : has[i];
    }
    const allowed = new Int32Array(MAX_TOKENS$1 - MIN_TOKENS$1 + 1);
    for (const t of tokens) allowed[t - MIN_TOKENS$1] = 1;
    const T = this.ort.Tensor;
    const feeds = {
      context_qpos: new T("float32", encode(context), [1, 4, POSE_SIZE]),
      target_qpos: new T("float32", encode(targets), [1, 4, POSE_SIZE]),
      has_global_root_values: new T("int32", has, [1, 8]),
      has_local_root_values: new T("int32", hasLocalRoot, [1, 8]),
      has_local_poses: new T("int32", has.slice(), [1, 8]),
      allowed_pred_num_tokens: new T("int32", allowed, [1, allowed.length])
    };
    this._busy = true;
    const t0 = performance.now();
    let results;
    try {
      results = await this.session.run(feeds);
    } finally {
      this._busy = false;
    }
    const inferenceMs = performance.now() - t0;
    const predQpos = results.pred_qpos.data;
    const predContacts = results.pred_foot_contacts.data;
    const tokenCount = Number(results.pred_num_tokens.data[0]);
    const rawCount = Math.min(Math.max(tokenCount * POSES_PER_TOKEN, 0), MAX_RAW_FRAMES);
    const frameCount = Math.max(0, rawCount - 2 * BOUNDARY_FRAMES);
    const poses = [];
    const footContacts = [];
    const radius = ROOT_POSITION_FILTER.length >> 1;
    for (let f = 0; f < frameCount; f++) {
      const rf = f + BOUNDARY_FRAMES;
      const pos = [0, 0, 0];
      ROOT_POSITION_FILTER.forEach((w, tap) => {
        const src = Math.min(Math.max(rf + tap - radius, 0), rawCount - 1) * POSE_SIZE;
        for (let k = 0; k < 3; k++) pos[k] += w * predQpos[src + k];
      });
      const o = rf * POSE_SIZE;
      const wpos = qrot$1(aq, pos);
      const pose = new Float32Array(POSE_SIZE);
      pose[0] = wpos[0] + ap[0];
      pose[1] = wpos[1] + ap[1];
      pose[2] = pos[2];
      pose.set(qnorm$1(qmul$1(aq, qnorm$1([predQpos[o + 3], predQpos[o + 4], predQpos[o + 5], predQpos[o + 6]]))), 3);
      pose.set(predQpos.subarray(o + ROOT_SIZE, o + POSE_SIZE), ROOT_SIZE);
      poses.push(pose);
      footContacts.push(Float32Array.from(predContacts.subarray(rf * 4, rf * 4 + 4), (c) => Math.min(Math.max(c, 0), 1)));
    }
    if (options.correct !== false && frameCount >= 2) correctEndpoints$1(poses, last, targets[0]);
    return { poses, footContacts, tokenCount, inferenceMs };
  }
}
function correctEndpoints$1(poses, start, end) {
  const n = poses.length;
  const delta = (p, guide) => ({
    p: [guide[0] - p[0], guide[1] - p[1], guide[2] - p[2]],
    q: qnorm$1(qmul$1(qinv$1(qnorm$1([p[3], p[4], p[5], p[6]])), qnorm$1([guide[3], guide[4], guide[5], guide[6]]))),
    a: Float32Array.from({ length: POSE_SIZE - ROOT_SIZE }, (_, i) => guide[ROOT_SIZE + i] - p[ROOT_SIZE + i])
  });
  const d0 = delta(poses[0], start);
  const d1 = delta(poses[n - 1], end);
  const fadeFrames = Math.min(JOINT_ANGLE_FADE_SECONDS * SAMPLE_RATE, 0.5 * (n - 1));
  for (let f = 0; f < n; f++) {
    const p = poses[f];
    const w = f / (n - 1);
    const sw = 1 - w;
    p.set(qnorm$1(qmul$1(qmul$1([p[3], p[4], p[5], p[6]], qpow(d0.q, sw)), qpow(d1.q, w))), 3);
    for (let k = 0; k < 3; k++) p[k] += d0.p[k] * sw + d1.p[k] * w;
    const ws = fadeWeight(f / fadeFrames);
    const we = fadeWeight((n - 1 - f) / fadeFrames);
    for (let i = ROOT_SIZE; i < POSE_SIZE; i++) p[i] += d0.a[i - ROOT_SIZE] * ws + d1.a[i - ROOT_SIZE] * we;
  }
}
const SOMA_SKELETON = {
  names: ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd", "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd"],
  parents: [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75],
  reference: [
    [0, 1.012054, 0, 0.496573, 0.503404, 0.496573, 0.503404],
    [0.051311, 593e-6, 0, 0, 0, -4111e-6, 0.999992],
    [0.07206, 0, 0, 0, 0, -0.029529, 0.999564],
    [0.07262, -666e-6, 0, 0, 0, 0.038697, 0.999251],
    [0.263508, 0, 0, 0, 0, 0.150518, 0.988607],
    [0.078721, 0, 0, 0, 0, 0, 1],
    [0.066593, 0, 0, 0, 0, -0.211553, 0.977367],
    [0.161575, 1599e-6, 0, 0, 0, 0, 1],
    [1112e-6, 0.030214, 0, 0, 0, 497e-5, 0.999988],
    [0.042952, 0.082239, 0.031058, 0.033097, 1e-3, 0.097265, 0.994708],
    [0.042952, 0.082239, -0.031058, -0.033097, -1e-3, 0.097265, 0.994708],
    [0.233884, 0.050382, 0.016136, 0.125499, -0.695881, -0.123036, 0.69632],
    [0.15743, 0, 0, 0, 0, 0.175741, 0.984436],
    [0.28916, 0, 0, 0, 0, 0, 1],
    [0.270262, 0, 0, 0.084666, 114e-6, 298e-6, 0.996409],
    [0.020852, 0.03313, 8397e-6, 0.470529, -0.042939, 0.267821, 0.839661],
    [0.047045, 0, 0, 0.181556, -0.140146, -0.232387, 0.945195],
    [0.028564, 0, 0, 0, 0, 0, 1],
    [0.031806, 0, 0, 0, 0, 0, 1],
    [0.03171, 0.022789, 1213e-6, -0.012302, 3106e-6, 0.013654, 0.999826],
    [0.063209, 0, 0, -0.072394, -2049e-6, -0.014168, 0.997273],
    [0.036355, 0, 0, 0, 0, 0, 1],
    [0.023208, 0, 0, 0, 0, 0, 1],
    [0.027612, 0, 0, 0, 0, 0, 1],
    [0.031009, 9073e-6, -4079e-6, -0.029877, -0.031692, -0.076527, 0.996116],
    [0.060545, 0, 0, -0.054568, 0.024994, 0.078635, 0.995095],
    [0.045741, 0, 0, 0, 0, 0, 1],
    [0.029249, 0, 0, 0, 0, 0, 1],
    [0.023414, 0, 0, 0, 0, 0, 1],
    [0.029041, -3612e-6, 799e-6, -0.039477, -0.055117, -0.108545, 0.991777],
    [0.058845, 0, 0, -0.044639, 0.045627, 0.112522, 0.991597],
    [0.043686, 0, 0, 0, 0, 0, 1],
    [0.026752, 0, 0, 0, 0, 0, 1],
    [0.019416, 0, 0, 0, 0, 0, 1],
    [0.028617, -0.0163, 5647e-6, -0.07089, -0.13775, -0.15043, 0.976407],
    [0.054877, 0, 0, -0.012057, 0.124429, 0.161253, 0.978963],
    [0.030999, 0, 0, 0, 0, 0, 1],
    [0.01585, 0, 0, 0, 0, 0, 1],
    [0.018839, 0, 0, 0, 0, 0, 1],
    [0.233885, 0.050382, -0.016167, 0.695881, 0.125499, 0.69632, 0.123036],
    [-0.157431, 0, 0, 0, 0, 0.175741, 0.984436],
    [-0.289151, 0, 0, 0, 0, 0, 1],
    [-0.270262, 0, 0, 0.084666, 114e-6, 298e-6, 0.996409],
    [-0.020851, -0.03313, -8396e-6, 0.470542, -0.042946, 0.267809, 0.839657],
    [-0.047047, 0, 0, 0.181544, -0.140133, -0.232382, 0.9452],
    [-0.028564, 0, 0, 0, 0, 0, 1],
    [-0.031806, 0, 0, 0, 0, 0, 1],
    [-0.031714, -0.022787, -1217e-6, -0.01228, 3145e-6, 0.013672, 0.999826],
    [-0.063207, 0, 0, -0.072416, -2087e-6, -0.014189, 0.997271],
    [-0.036355, 0, 0, 0, 0, 0, 1],
    [-0.023208, 0, 0, 0, 0, 0, 1],
    [-0.027612, 0, 0, 0, 0, 0, 1],
    [-0.03101, -9073e-6, 4079e-6, -0.029878, -0.03169, -0.076524, 0.996116],
    [-0.060546, 0, 0, -0.054567, 0.024992, 0.078632, 0.995095],
    [-0.045741, 0, 0, 0, 0, 0, 1],
    [-0.029248, 0, 0, 0, 0, 0, 1],
    [-0.023415, 0, 0, 0, 0, 0, 1],
    [-0.029041, 3612e-6, -798e-6, -0.039477, -0.055113, -0.108544, 0.991777],
    [-0.058847, 0, 0, -0.044639, 0.045623, 0.11252, 0.991597],
    [-0.043685, 0, 0, 0, 0, 0, 1],
    [-0.026752, 0, 0, 0, 0, 0, 1],
    [-0.019415, 0, 0, 0, 0, 0, 1],
    [-0.028619, 0.016301, -5648e-6, -0.070891, -0.137748, -0.150424, 0.976408],
    [-0.054877, 0, 0, -0.012056, 0.124427, 0.161247, 0.978965],
    [-0.030998, 0, 0, 0, 0, 0, 1],
    [-0.01585, 0, 0, 0, 0, 0, 1],
    [-0.018839, 0, 0, 0, 0, 0, 1],
    [-0.08308, 0.021259, 0.100069, 0, 0, 0.999977, -683e-5],
    [0.433003, 0, 0, 0, 0, 0, 1],
    [0.424241, 0, 0, 0, 0, -0.566943, 0.823757],
    [0.141328, 0, 0, 235e-6, 699e-6, -0.04126, 0.999148],
    [0.067743, -1182e-6, 0, 0, 0, -8726e-6, 0.999962],
    [-0.083079, 0.021259, -0.100015, 0, 0, 683e-5, 0.999977],
    [-0.433004, 0, 0, 0, 0, 0, 1],
    [-0.42424, 0, 0, 0, 0, -0.566943, 0.823757],
    [-0.141328, 0, 0, 235e-6, 699e-6, -0.041258, 0.999148],
    [-0.067743, 1182e-6, 0, 0, 0, -8726e-6, 0.999962]
  ]
};
const SOMA_MODEL_JOINTS = [
  "Hips",
  "Spine1",
  "Spine2",
  "Chest",
  "Neck1",
  "Neck2",
  "Head",
  "Jaw",
  "LeftEye",
  "RightEye",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "LeftHandThumbEnd",
  "LeftHandMiddleEnd",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "RightHandThumbEnd",
  "RightHandMiddleEnd",
  "LeftLeg",
  "LeftShin",
  "LeftFoot",
  "LeftToeBase",
  "RightLeg",
  "RightShin",
  "RightFoot",
  "RightToeBase"
];
const FPS = 30, PPT = 4, BOUNDARY = 3, MIN_TOKENS = 6, MAX_TOKENS = 16;
const MODEL_J = SOMA_MODEL_JOINTS.length;
const DEFAULT_TOKENS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
function qmul(a, b, ao = 0, bo = 0) {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3], bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz];
}
function qrot(q, v, qo = 0) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
function qnorm(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
const qinv = (q, o = 0) => [-q[o], -q[o + 1], -q[o + 2], q[o + 3]];
function tmulInto(a, ao, b, bo, out, oo) {
  const p = qrot(a, [b[bo], b[bo + 1], b[bo + 2]], ao + 3);
  const q = qnorm(qmul(a, b, ao + 3, bo + 3));
  out[oo] = p[0] + a[ao];
  out[oo + 1] = p[1] + a[ao + 1];
  out[oo + 2] = p[2] + a[ao + 2];
  out[oo + 3] = q[0];
  out[oo + 4] = q[1];
  out[oo + 5] = q[2];
  out[oo + 6] = q[3];
}
function tinv(t, o = 0) {
  const q = qinv(t, o + 3), p = qrot(q, [t[o], t[o + 1], t[o + 2]]);
  return [-p[0], -p[1], -p[2], ...q];
}
function slerpIdentity(q, t, o = 0) {
  let x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  if (w < 0) [x, y, z, w] = [-x, -y, -z, -w];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0, 0, 1];
  const angle = 2 * Math.atan2(s, w) * t, k = Math.sin(angle / 2) / s;
  return [x * k, y * k, z * k, Math.cos(angle / 2)];
}
function rotationFrom6d(f, o) {
  let xl = Math.hypot(f[o], f[o + 1], f[o + 2]) || 1;
  const x = [f[o] / xl, f[o + 1] / xl, f[o + 2] / xl], yr = [f[o + 3], f[o + 4], f[o + 5]];
  let z = [x[1] * yr[2] - x[2] * yr[1], x[2] * yr[0] - x[0] * yr[2], x[0] * yr[1] - x[1] * yr[0]];
  xl = Math.hypot(...z) || 1;
  z = [z[0] / xl, z[1] / xl, z[2] / xl];
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const m00 = x[0], m10 = x[1], m20 = x[2], m01 = y[0], m11 = y[1], m21 = y[2], m02 = z[0], m12 = z[1], m22 = z[2];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    q = [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  return qnorm(q);
}
function headingFromHips(right, left, ro = 0, lo = 0) {
  let ax = right[ro] - left[lo], az = right[ro + 2] - left[lo + 2];
  const l = Math.hypot(ax, az);
  if (l <= 1e-6) return 0;
  ax /= l;
  az /= l;
  return Math.atan2(az, -ax);
}
class SOMASkeletonInfo {
  constructor() {
    this.names = SOMA_SKELETON.names;
    this.parents = SOMA_SKELETON.parents;
    this.J = SOMA_SKELETON.names.length;
    this.reference = Float32Array.from(SOMA_SKELETON.reference.flat());
    this.modelToPose = SOMA_MODEL_JOINTS.map((n) => this.names.indexOf(n));
    this.poseToModel = new Int32Array(this.J).fill(-1);
    this.modelToPose.forEach((p, m) => {
      const parent = this.parents[p];
      if (parent === -1 || this.poseToModel[parent] >= 0) this.poseToModel[p] = m;
    });
    this.rightLeg = this.names.indexOf("RightLeg");
    this.leftLeg = this.names.indexOf("LeftLeg");
  }
  index(name) {
    return this.names.indexOf(name);
  }
  /** A pose in the reference (bind) pose. */
  referencePose() {
    return Float32Array.from(this.reference);
  }
  /** World transforms of a pose, J * 7. */
  world(pose, out = new Float32Array(this.J * 7)) {
    for (let j = 0; j < this.J; j++) {
      const p = this.parents[j];
      if (p < 0) for (let k = 0; k < 7; k++) out[j * 7 + k] = pose[j * 7 + k];
      else tmulInto(out, p * 7, pose, j * 7, out, j * 7);
    }
    return out;
  }
  /** The pose's heading from its hips: an angle around Y, 0 facing +Z. */
  heading(pose) {
    const w = this.world(pose);
    return headingFromHips(w, w, this.rightLeg * 7, this.leftLeg * 7);
  }
}
const SOMA_SKELETON_INFO = new SOMASkeletonInfo();
class MotionBricksSOMA extends MotionBricksG1 {
  constructor() {
    super(...arguments);
    this.skeleton = SOMA_SKELETON_INFO;
  }
  static async Load(ort, model, sessionOptions = {}, onProgress, cacheName) {
    return super.Load(ort, model, sessionOptions, onProgress, cacheName);
  }
  async generate(context, target, options = {}) {
    if (context.length !== PPT) throw new Error("Context needs exactly 4 poses");
    if (target.length < 1 || target.length > PPT) throw new Error("Target needs 1 to 4 poses");
    if (this._busy) throw new Error("A generation is already running");
    const k = this.skeleton, J = k.J;
    const nt = target.length;
    const guide = [...context, ...[0, 1, 2, 3].map((i) => target[Math.min(i, nt - 1)])];
    const W = guide.map((p) => k.world(p));
    const last = W[PPT - 1];
    const current = headingFromHips(last, last, k.rightLeg * 7, k.leftLeg * 7);
    const desired = Math.atan2(1, 0);
    const half = (current - desired) / 2;
    const align = [last[0], 0, last[2], 0, Math.sin(half), 0, Math.cos(half)];
    const alignInv = tinv(align);
    const A = W.map((w) => {
      const out = new Float32Array(J * 7);
      for (let j = 0; j < J; j++) tmulInto(alignInv, 0, w, j * 7, out, j * 7);
      return out;
    });
    const T = this.ort.Tensor;
    const g = new Float32Array(8 * 5), l = new Float32Array(8 * 4), lp = new Float32Array(8 * 267);
    for (let f = 0; f < 8; f++) {
      const nf = f === PPT - 1 ? f : Math.min(f + 1, 7);
      const a = A[f], b = A[nf];
      const hc = headingFromHips(a, a, k.rightLeg * 7, k.leftLeg * 7), hn = headingFromHips(b, b, k.rightLeg * 7, k.leftLeg * 7);
      const dh = Math.atan2(Math.sin(hn - hc), Math.cos(hn - hc));
      g.set([a[0], a[1], a[2], Math.cos(hc), Math.sin(hc)], f * 5);
      l.set([dh * FPS, (b[0] - a[0]) * FPS, (b[2] - a[2]) * FPS, a[1]], f * 4);
      const o = f * 267;
      for (let m = 0; m < MODEL_J; m++) {
        const p = k.modelToPose[m] * 7;
        if (m > 0) lp.set([a[p] - a[0], a[p + 1], a[p + 2] - a[2]], o + (m - 1) * 3);
        const x = qrot(a, [1, 0, 0], p + 3), y = qrot(a, [0, 1, 0], p + 3);
        lp.set([...x, ...y], o + (MODEL_J - 1) * 3 + m * 6);
      }
    }
    const has = new Int32Array(8), hasLocalRoot = new Int32Array(8);
    for (let i = 0; i < 8; i++) {
      has[i] = i < PPT + nt ? 1 : 0;
      hasLocalRoot[i] = i === PPT - 1 || i === PPT + nt - 1 ? 0 : has[i];
    }
    const allowed = new Int32Array(MAX_TOKENS - MIN_TOKENS + 1);
    for (const t of options.tokens ?? DEFAULT_TOKENS) allowed[t - MIN_TOKENS] = 1;
    const feeds = {
      global_root_values: new T("float32", g, [1, 8, 5]),
      has_global_root_values: new T("int32", has, [1, 8]),
      local_root_values: new T("float32", l, [1, 8, 4]),
      has_local_root_values: new T("int32", hasLocalRoot, [1, 8]),
      local_poses: new T("float32", lp, [1, 8, 267]),
      has_local_poses: new T("int32", has.slice(), [1, 8]),
      allowed_pred_num_tokens: new T("int32", allowed, [1, allowed.length])
    };
    this._busy = true;
    const t0 = performance.now();
    let results;
    try {
      results = await this.session.run(feeds);
    } finally {
      this._busy = false;
    }
    const inferenceMs = performance.now() - t0;
    const pred = results.pred_global_motions.data;
    const F = results.pred_global_motions.dims[2];
    const tokenCount = Number(results.pred_num_tokens.data[0]);
    const raw = Math.min(Math.max(tokenCount * PPT, 0), results.pred_global_motions.dims[1]);
    const n = Math.max(0, raw - 2 * BOUNDARY);
    const ro = 5 + (MODEL_J - 1) * 3;
    const poses = [], footContacts = [];
    const G = new Array(MODEL_J);
    for (let f = 0; f < n; f++) {
      const base = (BOUNDARY + f) * F;
      footContacts.push(Float32Array.from(pred.subarray(base + F - 4, base + F), (c) => Math.min(Math.max(c, 0), 1)));
      for (let m = 0; m < MODEL_J; m++) G[m] = rotationFrom6d(pred, base + ro + m * 6);
      const pose = new Float32Array(J * 7);
      tmulInto(align, 0, [pred[base], pred[base + 1], pred[base + 2], ...G[0]], 0, pose, 0);
      for (let j = 1; j < J; j++) {
        const m = k.poseToModel[j];
        if (m < 0) {
          pose.set(k.reference.subarray(j * 7, j * 7 + 7), j * 7);
          continue;
        }
        const pm = k.poseToModel[k.parents[j]];
        pose.set(k.reference.subarray(j * 7, j * 7 + 3), j * 7);
        pose.set(qnorm(qmul(qinv(G[pm]), G[m])), j * 7 + 3);
      }
      poses.push(pose);
    }
    if (options.correct !== false && n >= 2) correctEndpoints(poses, context[PPT - 1], target[0], J);
    return { poses, footContacts, tokenCount, inferenceMs };
  }
}
function correctEndpoints(poses, from, to, jointCount) {
  const J = jointCount;
  const n = poses.length, lastFrame = poses[n - 1];
  const start = new Float32Array(J * 7), end = new Float32Array(J * 7);
  for (const [delta, gen, guide] of [[start, poses[0], from], [end, lastFrame, to]]) {
    const r = qnorm(qmul(qinv(qnorm([gen[3], gen[4], gen[5], gen[6]])), qnorm([guide[3], guide[4], guide[5], guide[6]])));
    delta.set([guide[0] - gen[0], guide[1] - gen[1], guide[2] - gen[2], ...r], 0);
    for (let j = 1; j < J; j++) tmulInto(tinv(gen, j * 7), 0, guide, j * 7, delta, j * 7);
  }
  const tmp = new Float32Array(7), w = new Float32Array(7);
  for (let f = 0; f < n; f++) {
    const we = f / (n - 1), ws = 1 - we, pose = poses[f];
    const q = qnorm(qmul(qmul(pose, slerpIdentity(start, ws, 3), 3, 0), slerpIdentity(end, we, 3)));
    pose.set([pose[0] + start[0] * ws + end[0] * we, pose[1] + start[1] * ws + end[1] * we, pose[2] + start[2] * ws + end[2] * we, ...q], 0);
    for (let j = 1; j < J; j++) {
      for (const [delta, weight] of [[start, ws], [end, we]]) {
        const o = j * 7;
        w.set([delta[o] * weight, delta[o + 1] * weight, delta[o + 2] * weight, ...slerpIdentity(delta, weight, o + 3)]);
        tmulInto(pose, o, w, 0, tmp, 0);
        pose.set(tmp, o);
      }
    }
  }
}
const scope = self;
let engine = null;
async function load(id, o) {
  const ort = await import(
    /* @vite-ignore */
    o.ort
  );
  await setupOrtWeb(ort, o.wasm);
  let last = 0;
  const progress = (loaded, total) => {
    const now = performance.now();
    if (now - last < 100 && loaded < total) return;
    last = now;
    scope.postMessage({ id, type: "progress", loaded, total });
  };
  engine = await (o.rig === "soma" ? MotionBricksSOMA : MotionBricksG1).Load(ort, o.model, o.sessionOptions, progress, o.cache);
  return { backend: engine.backend };
}
scope.onmessage = async (e) => {
  const r = e.data;
  try {
    if (r.type === "load") {
      scope.postMessage({ id: r.id, type: "done", result: await load(r.id, r.options) });
    } else {
      if (!engine) throw new Error("The model is not loaded");
      const motion = await engine.generate(r.context, r.target, r.options);
      const transfer = [...motion.poses, ...motion.footContacts].map((a) => a.buffer);
      scope.postMessage({ id: r.id, type: "done", result: motion }, transfer);
    }
  } catch (err) {
    scope.postMessage({ id: r.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
//# sourceMappingURL=motionbricks.worker.js.map
