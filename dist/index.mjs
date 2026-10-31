import { Quaternion, Vector3, Group, Euler, Object3D, MeshStandardMaterial, Mesh, CapsuleGeometry, SphereGeometry, BoxGeometry } from "three";
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
function qmul$2(a, b) {
  const [ax, ay, az, aw] = [a[0], a[1], a[2], a[3]];
  const [bx, by, bz, bw] = [b[0], b[1], b[2], b[3]];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz
  ];
}
function qinv$2(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}
function qnorm$2(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qrot$2(q, v) {
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
  let [x, y, z, w] = qnorm$2(q);
  if (w < 0) [x, y, z, w] = [-x, -y, -z, -w];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0, 0, 1];
  const angle = 2 * Math.atan2(s, w) * t;
  const k = Math.sin(angle / 2) / s;
  return [x * k, y * k, z * k, Math.cos(angle / 2)];
}
function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bs = d < 0 ? -1 : 1;
  d *= bs;
  if (d > 0.9995) return qnorm$2([0, 1, 2, 3].map((i) => a[i] + (b[i] * bs - a[i]) * t));
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s * bs;
  return [0, 1, 2, 3].map((i) => a[i] * wa + b[i] * wb);
}
function heading(q) {
  const f = qrot$2(qnorm$2(q), [1, 0, 0]);
  return Math.atan2(f[1], f[0]);
}
function fadeWeight(v) {
  const t = Math.min(Math.max(v, 0), 1);
  const r = 1 - t;
  return t < 0.5 ? 1 - 2 * t * t : 2 * r * r;
}
class SharedEngine {
  constructor(engine) {
    this._tail = Promise.resolve();
    this.waiting = 0;
    this.engine = engine;
  }
  get backend() {
    return this.engine.backend;
  }
  async generate(context, target, options) {
    const start = performance.now();
    this.waiting++;
    const run = this._tail.then(async () => this.engine.generate(context, target, options));
    this._tail = run.catch(() => void 0);
    try {
      const motion = await run;
      return { ...motion, inferenceMs: performance.now() - start };
    } finally {
      this.waiting--;
    }
  }
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
    const engine = manifest.format === "motionbricks-external" ? await this._loadLite(ort, baseUrl, manifest, sessionOptions, onProgress, cache) : await this._loadQ8(ort, baseUrl, manifest, sessionOptions, onProgress, cache);
    await cache?.complete(manifestText);
    return engine;
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
    const aqi = qinv$2(aq);
    const encode = (poses2) => {
      const out = new Float32Array(POSES_PER_TOKEN * POSE_SIZE);
      poses2.forEach((p, i) => {
        const o = i * POSE_SIZE;
        for (let k = 0; k < POSE_SIZE; k++) out[o + k] = p[k];
        const pos = qrot$2(aqi, [p[0] - ap[0], p[1] - ap[1], p[2] - ap[2]]);
        out[o] = pos[0];
        out[o + 1] = pos[1];
        out.set(qnorm$2(qmul$2(aqi, qnorm$2([p[3], p[4], p[5], p[6]]))), o + 3);
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
      const wpos = qrot$2(aq, pos);
      const pose = new Float32Array(POSE_SIZE);
      pose[0] = wpos[0] + ap[0];
      pose[1] = wpos[1] + ap[1];
      pose[2] = pos[2];
      pose.set(qnorm$2(qmul$2(aq, qnorm$2([predQpos[o + 3], predQpos[o + 4], predQpos[o + 5], predQpos[o + 6]]))), 3);
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
    q: qnorm$2(qmul$2(qinv$2(qnorm$2([p[3], p[4], p[5], p[6]])), qnorm$2([guide[3], guide[4], guide[5], guide[6]]))),
    a: Float32Array.from({ length: POSE_SIZE - ROOT_SIZE }, (_, i) => guide[ROOT_SIZE + i] - p[ROOT_SIZE + i])
  });
  const d0 = delta(poses[0], start);
  const d1 = delta(poses[n - 1], end);
  const fadeFrames = Math.min(JOINT_ANGLE_FADE_SECONDS * SAMPLE_RATE, 0.5 * (n - 1));
  for (let f = 0; f < n; f++) {
    const p = poses[f];
    const w = f / (n - 1);
    const sw = 1 - w;
    p.set(qnorm$2(qmul$2(qmul$2([p[3], p[4], p[5], p[6]], qpow(d0.q, sw)), qpow(d1.q, w))), 3);
    for (let k = 0; k < 3; k++) p[k] += d0.p[k] * sw + d1.p[k] * w;
    const ws = fadeWeight(f / fadeFrames);
    const we = fadeWeight((n - 1 - f) / fadeFrames);
    for (let i = ROOT_SIZE; i < POSE_SIZE; i++) p[i] += d0.a[i - ROOT_SIZE] * ws + d1.a[i - ROOT_SIZE] * we;
  }
}
function lerpPose(a, b, t, out = new Float32Array(POSE_SIZE)) {
  for (let i = 0; i < POSE_SIZE; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  out.set(qslerp([a[3], a[4], a[5], a[6]], [b[3], b[4], b[5], b[6]], t), 3);
  return out;
}
function placePoses(poses, x, y, angle) {
  const q = qz(angle);
  return poses.map((src) => {
    const p = Float32Array.from(src);
    const pos = qrot$2(q, [src[0], src[1], 0]);
    p[0] = pos[0] + x;
    p[1] = pos[1] + y;
    p.set(qnorm$2(qmul$2(q, [src[3], src[4], src[5], src[6]])), 3);
    return p;
  });
}
function withSpeed(poses, speed) {
  const first = poses[0], last = poses[poses.length - 1];
  const current = Math.hypot(last[0] - first[0], last[1] - first[1]) * SAMPLE_RATE / (poses.length - 1);
  const s = current > 1e-3 ? speed / current : 1;
  return poses.map((src) => {
    const p = Float32Array.from(src);
    p[0] = first[0] + (src[0] - first[0]) * s;
    p[1] = first[1] + (src[1] - first[1]) * s;
    return p;
  });
}
const G1_JOINT_BODIES = [
  ["left_hip_pitch_link", [0, 1, 0]],
  ["left_hip_roll_link", [1, 0, 0]],
  ["left_hip_yaw_link", [0, 0, 1]],
  ["left_knee_link", [0, 1, 0]],
  ["left_ankle_pitch_link", [0, 1, 0]],
  ["left_ankle_roll_link", [1, 0, 0]],
  ["right_hip_pitch_link", [0, 1, 0]],
  ["right_hip_roll_link", [1, 0, 0]],
  ["right_hip_yaw_link", [0, 0, 1]],
  ["right_knee_link", [0, 1, 0]],
  ["right_ankle_pitch_link", [0, 1, 0]],
  ["right_ankle_roll_link", [1, 0, 0]],
  ["waist_yaw_link", [0, 0, 1]],
  ["waist_roll_link", [1, 0, 0]],
  ["torso_link", [0, 1, 0]],
  ["left_shoulder_pitch_link", [0, 1, 0]],
  ["left_shoulder_roll_link", [1, 0, 0]],
  ["left_shoulder_yaw_link", [0, 0, 1]],
  ["left_elbow_link", [0, 1, 0]],
  ["left_wrist_roll_link", [1, 0, 0]],
  ["left_wrist_pitch_link", [0, 1, 0]],
  ["left_wrist_yaw_link", [0, 0, 1]],
  ["right_shoulder_pitch_link", [0, 1, 0]],
  ["right_shoulder_roll_link", [1, 0, 0]],
  ["right_shoulder_yaw_link", [0, 0, 1]],
  ["right_elbow_link", [0, 1, 0]],
  ["right_wrist_roll_link", [1, 0, 0]],
  ["right_wrist_pitch_link", [0, 1, 0]],
  ["right_wrist_yaw_link", [0, 0, 1]]
];
const tempQuat = new Quaternion();
class G1Robot {
  constructor(object, pelvis, joints, materials) {
    this.object = object;
    this.pelvis = pelvis;
    this.joints = joints;
    this.materials = materials;
  }
  /**
   * Wrap a loaded G1 model. Its meshes cast and receive shadows unless `shadows` is false.
   */
  static FromObject(root, { shadows = true } = {}) {
    const find = (name) => {
      const node = root.getObjectByName(name);
      if (!node) throw new Error(`The robot model has no node named ${name}`);
      return node;
    };
    const joints = G1_JOINT_BODIES.map(([body, axis]) => {
      const node = find(body);
      return { node, rest: node.quaternion.clone(), axis: new Vector3(...axis) };
    });
    const materials = /* @__PURE__ */ new Set();
    root.traverse((o) => {
      const mesh = o;
      if (!mesh.isMesh) return;
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(m);
    });
    return new G1Robot(root, find("pelvis"), joints, [...materials]);
  }
  /** Show a G1 pose: root XYZ, root XYZW quaternion, then the 29 joint angles. */
  setPose(pose) {
    this.pelvis.position.set(pose[0], pose[1], pose[2]);
    this.pelvis.quaternion.set(pose[3], pose[4], pose[5], pose[6]).normalize();
    this.joints.forEach((j, i) => {
      j.node.quaternion.copy(j.rest).multiply(tempQuat.setFromAxisAngle(j.axis, pose[ROOT_SIZE + i]));
    });
  }
  /** Convert a ground point from three.js space (Y up) to G1 space (Z up). */
  static ToG1(p) {
    return [p.x, -p.z];
  }
  /** Convert a G1 space point to three.js space. */
  static FromG1(x, y, z = 0, target = new Vector3()) {
    return target.set(x, z, -y);
  }
}
const G1_BODY_TREE = [
  ["pelvis", "", [0, 0, 0.793], [0, 0, 0, 1]],
  ["left_hip_pitch_link", "pelvis", [0, 0.064452, -0.1027], [0, 0, 0, 1]],
  ["left_hip_roll_link", "left_hip_pitch_link", [0, 0.052, -0.030465], [0, -0.0873386, 0, 0.996179]],
  ["left_hip_yaw_link", "left_hip_roll_link", [0.025001, 0, -0.12412], [0, 0, 0, 1]],
  ["left_knee_link", "left_hip_yaw_link", [-0.078273, 21489e-7, -0.17734], [0, 0.0873386, 0, 0.996179]],
  ["left_ankle_pitch_link", "left_knee_link", [0, -94445e-9, -0.30001], [0, 0, 0, 1]],
  ["left_ankle_roll_link", "left_ankle_pitch_link", [0, 0, -0.017558], [0, 0, 0, 1]],
  ["right_hip_pitch_link", "pelvis", [0, -0.064452, -0.1027], [0, 0, 0, 1]],
  ["right_hip_roll_link", "right_hip_pitch_link", [0, -0.052, -0.030465], [0, -0.0873386, 0, 0.996179]],
  ["right_hip_yaw_link", "right_hip_roll_link", [0.025001, 0, -0.12412], [0, 0, 0, 1]],
  ["right_knee_link", "right_hip_yaw_link", [-0.078273, -21489e-7, -0.17734], [0, 0.0873386, 0, 0.996179]],
  ["right_ankle_pitch_link", "right_knee_link", [0, 94445e-9, -0.30001], [0, 0, 0, 1]],
  ["right_ankle_roll_link", "right_ankle_pitch_link", [0, 0, -0.017558], [0, 0, 0, 1]],
  ["waist_yaw_link", "pelvis", [0, 0, 0], [0, 0, 0, 1]],
  ["waist_roll_link", "waist_yaw_link", [-39635e-7, 0, 0.044], [0, 0, 0, 1]],
  ["torso_link", "waist_roll_link", [0, 0, 0], [0, 0, 0, 1]],
  ["left_shoulder_pitch_link", "torso_link", [39563e-7, 0.10022, 0.24778], [0.139201, 138722e-10, -986868e-10, 0.990264]],
  ["left_shoulder_roll_link", "left_shoulder_pitch_link", [0, 0.038, -0.013831], [-0.139172, 0, 0, 0.990268]],
  ["left_shoulder_yaw_link", "left_shoulder_roll_link", [0, 624e-5, -0.1032], [0, 0, 0, 1]],
  ["left_elbow_link", "left_shoulder_yaw_link", [0.015783, 0, -0.080518], [0, 0, 0, 1]],
  ["left_wrist_roll_link", "left_elbow_link", [0.1, 188791e-8, -0.01], [0, 0, 0, 1]],
  ["left_wrist_pitch_link", "left_wrist_roll_link", [0.038, 0, 0], [0, 0, 0, 1]],
  ["left_wrist_yaw_link", "left_wrist_pitch_link", [0.046, 0, 0], [0, 0, 0, 1]],
  ["right_shoulder_pitch_link", "torso_link", [39563e-7, -0.10021, 0.24778], [-0.139201, 138722e-10, 986868e-10, 0.990264]],
  ["right_shoulder_roll_link", "right_shoulder_pitch_link", [0, -0.038, -0.013831], [0.139172, 0, 0, 0.990268]],
  ["right_shoulder_yaw_link", "right_shoulder_roll_link", [0, -624e-5, -0.1032], [0, 0, 0, 1]],
  ["right_elbow_link", "right_shoulder_yaw_link", [0.015783, 0, -0.080518], [0, 0, 0, 1]],
  ["right_wrist_roll_link", "right_elbow_link", [0.1, -188791e-8, -0.01], [0, 0, 0, 1]],
  ["right_wrist_pitch_link", "right_wrist_roll_link", [0.038, 0, 0], [0, 0, 0, 1]],
  ["right_wrist_yaw_link", "right_wrist_pitch_link", [0.046, 0, 0], [0, 0, 0, 1]]
];
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
function qmul$1(a, b, ao = 0, bo = 0) {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3], bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz];
}
function qrot$1(q, v, qo = 0) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
function qnorm$1(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
const qinv$1 = (q, o = 0) => [-q[o], -q[o + 1], -q[o + 2], q[o + 3]];
function tmulInto(a, ao, b, bo, out, oo) {
  const p = qrot$1(a, [b[bo], b[bo + 1], b[bo + 2]], ao + 3);
  const q = qnorm$1(qmul$1(a, b, ao + 3, bo + 3));
  out[oo] = p[0] + a[ao];
  out[oo + 1] = p[1] + a[ao + 1];
  out[oo + 2] = p[2] + a[ao + 2];
  out[oo + 3] = q[0];
  out[oo + 4] = q[1];
  out[oo + 5] = q[2];
  out[oo + 6] = q[3];
}
function tinv$1(t, o = 0) {
  const q = qinv$1(t, o + 3), p = qrot$1(q, [t[o], t[o + 1], t[o + 2]]);
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
  return qnorm$1(q);
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
    const alignInv = tinv$1(align);
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
        const x = qrot$1(a, [1, 0, 0], p + 3), y = qrot$1(a, [0, 1, 0], p + 3);
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
        pose.set(qnorm$1(qmul$1(qinv$1(G[pm]), G[m])), j * 7 + 3);
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
    const r = qnorm$1(qmul$1(qinv$1(qnorm$1([gen[3], gen[4], gen[5], gen[6]])), qnorm$1([guide[3], guide[4], guide[5], guide[6]])));
    delta.set([guide[0] - gen[0], guide[1] - gen[1], guide[2] - gen[2], ...r], 0);
    for (let j = 1; j < J; j++) tmulInto(tinv$1(gen, j * 7), 0, guide, j * 7, delta, j * 7);
  }
  const tmp = new Float32Array(7), w = new Float32Array(7);
  for (let f = 0; f < n; f++) {
    const we = f / (n - 1), ws = 1 - we, pose = poses[f];
    const q = qnorm$1(qmul$1(qmul$1(pose, slerpIdentity(start, ws, 3), 3, 0), slerpIdentity(end, we, 3)));
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
function parseSOMABvh(text, skeleton = SOMA_SKELETON_INFO, rate = FPS) {
  const [head, motion] = text.split(/\bMOTION\b/);
  const names = [], channels = [], offsets = [], stack = [];
  let inEnd = false;
  for (const line of head.split("\n")) {
    const t = line.trim().split(/\s+/);
    if (t[0] === "ROOT" || t[0] === "JOINT") {
      names.push(t[1].split(":").pop());
      channels.push([]);
      offsets.push([0, 0, 0]);
      stack.push(names.length - 1);
    } else if (t[0] === "End" && t[1] === "Site") inEnd = true;
    else if (t[0] === "OFFSET" && !inEnd) offsets[stack[stack.length - 1]] = [+t[1] * 0.01, +t[2] * 0.01, +t[3] * 0.01];
    else if (t[0] === "CHANNELS") channels[stack[stack.length - 1]] = t.slice(2);
    else if (t[0] === "}") {
      if (inEnd) inEnd = false;
      else stack.pop();
    }
  }
  const lines = motion.trim().split("\n");
  const frameTime = parseFloat(lines[1].split(":")[1]);
  const step = Math.max(1, Math.round(1 / frameTime / rate));
  const rows = lines.slice(2);
  const out = [];
  const d = Math.PI / 180;
  for (let r = 0; r < rows.length; r += step) {
    const v = rows[r].trim().split(/\s+/).map(Number);
    const pose = skeleton.referencePose();
    let col = 0;
    names.forEach((name, j) => {
      const pi = skeleton.index(name);
      let pos = null;
      let q = [0, 0, 0, 1];
      for (const c of channels[j]) {
        const lc = c.toLowerCase(), axis = "xyz".indexOf(lc[0]);
        if (lc.endsWith("position")) {
          pos ??= [0, 0, 0];
          pos[axis] = v[col] * 0.01;
        } else {
          const h = v[col] * d / 2, aq = [0, 0, 0, Math.cos(h)];
          aq[axis] = Math.sin(h);
          q = qmul$1(q, aq);
        }
        col++;
      }
      if (pi < 0) return;
      pose.set(pos ?? offsets[j], pi * 7);
      pose.set(qnorm$1(q), pi * 7 + 3);
    });
    out.push(pose);
  }
  return out;
}
const G1_PARTS = {
  pelvis: "pelvis",
  torso: "torso_link",
  leftThigh: "left_hip_yaw_link",
  leftShin: "left_knee_link",
  leftFoot: "left_ankle_roll_link",
  rightThigh: "right_hip_yaw_link",
  rightShin: "right_knee_link",
  rightFoot: "right_ankle_roll_link",
  leftUpperArm: "left_shoulder_yaw_link",
  leftForearm: "left_elbow_link",
  leftHand: "left_wrist_yaw_link",
  rightUpperArm: "right_shoulder_yaw_link",
  rightForearm: "right_elbow_link",
  rightHand: "right_wrist_yaw_link"
};
const SOMA_PARTS = {
  pelvis: "Hips",
  torso: "Chest",
  leftThigh: "LeftLeg",
  leftShin: "LeftShin",
  leftFoot: "LeftFoot",
  rightThigh: "RightLeg",
  rightShin: "RightShin",
  rightFoot: "RightFoot",
  leftUpperArm: "LeftArm",
  leftForearm: "LeftForeArm",
  leftHand: "LeftHand",
  rightUpperArm: "RightArm",
  rightForearm: "RightForeArm",
  rightHand: "RightHand"
};
const LIMB_SEGMENTS = [
  ["leftThigh", "leftShin"],
  ["leftShin", "leftFoot"],
  ["rightThigh", "rightShin"],
  ["rightShin", "rightFoot"],
  ["leftUpperArm", "leftForearm"],
  ["leftForearm", "leftHand"],
  ["rightUpperArm", "rightForearm"],
  ["rightForearm", "rightHand"]
];
const MIXAMO = (prefix) => ({
  pelvis: `${prefix}Hips`,
  torso: `${prefix}Spine1`,
  leftThigh: `${prefix}LeftUpLeg`,
  leftShin: `${prefix}LeftLeg`,
  leftFoot: `${prefix}LeftFoot`,
  rightThigh: `${prefix}RightUpLeg`,
  rightShin: `${prefix}RightLeg`,
  rightFoot: `${prefix}RightFoot`,
  leftUpperArm: `${prefix}LeftArm`,
  leftForearm: `${prefix}LeftForeArm`,
  leftHand: `${prefix}LeftHand`,
  rightUpperArm: `${prefix}RightArm`,
  rightForearm: `${prefix}RightForeArm`,
  rightHand: `${prefix}RightHand`
});
const CHARACTER_RIGS = {
  /** Mixamo skeletons in a T-pose facing -Z, such as Polygonal Mind's 100 Avatars (VRM 0). */
  mixamo: {
    label: "Mixamo skeleton",
    bones: MIXAMO("mixamorig:"),
    forward: "-z"
  },
  /** The same, facing +Z, as Mixamo exports to glTF. */
  mixamoPlusZ: {
    label: "Mixamo skeleton (+Z)",
    bones: MIXAMO("mixamorig"),
    forward: "+z"
  },
  /** Tomás Laulhé's RobotExpressive (Quaternius, CC0), from the three.js examples. */
  robotExpressive: {
    label: "RobotExpressive",
    bones: {
      pelvis: "Hips",
      torso: "Abdomen",
      leftThigh: "UpperLeg.L",
      leftShin: "LowerLeg.L",
      leftFoot: "Foot.L",
      rightThigh: "UpperLeg.R",
      rightShin: "LowerLeg.R",
      rightFoot: "Foot.R",
      leftUpperArm: "UpperArm.L",
      leftForearm: "LowerArm.L",
      rightUpperArm: "UpperArm.R",
      rightForearm: "LowerArm.R"
    },
    root: "Body",
    forward: "+z",
    // Its feet are IK targets under the root: they follow the shins' ends, which also mark the ankles.
    // eslint-disable-next-line @typescript-eslint/naming-convention
    attach: { "Foot.L": "LowerLeg.L_end", "Foot.R": "LowerLeg.R_end" },
    ends: { leftShin: "LowerLeg.L_end", rightShin: "LowerLeg.R_end", leftForearm: "Palm2.L", rightForearm: "Palm2.R" }
  },
  unitreeH1: {
    label: "Unitree H1",
    bones: {
      pelvis: "pelvis",
      torso: "torso_link",
      leftThigh: "left_hip_pitch_link",
      leftShin: "left_knee_link",
      leftFoot: "left_ankle_link",
      rightThigh: "right_hip_pitch_link",
      rightShin: "right_knee_link",
      rightFoot: "right_ankle_link",
      leftUpperArm: "left_shoulder_yaw_link",
      leftForearm: "left_elbow_link",
      rightUpperArm: "right_shoulder_yaw_link",
      rightForearm: "right_elbow_link"
    },
    forward: "+x",
    // Like the G1, its zero pose has the forearms forward; it has no hand bodies to point them by.
    rest: { left_elbow_link: [0, 0, -Math.PI / 2], right_elbow_link: [0, 0, -Math.PI / 2] }
  },
  boosterT1: {
    label: "Booster T1",
    bones: {
      pelvis: "Waist",
      torso: "Trunk",
      leftThigh: "Hip_Yaw_Left",
      leftShin: "Shank_Left",
      leftFoot: "left_foot_link",
      rightThigh: "Hip_Yaw_Right",
      rightShin: "Shank_Right",
      rightFoot: "right_foot_link",
      leftUpperArm: "AL2",
      leftForearm: "AL3",
      leftHand: "left_hand_link",
      rightUpperArm: "AR2",
      rightForearm: "AR3",
      rightHand: "right_hand_link"
    },
    root: "Trunk",
    forward: "+x"
  },
  robotisOP3: {
    label: "Robotis OP3",
    bones: {
      pelvis: "body_link",
      leftThigh: "l_hip_pitch_link",
      leftShin: "l_knee_link",
      leftFoot: "l_ank_roll_link",
      rightThigh: "r_hip_pitch_link",
      rightShin: "r_knee_link",
      rightFoot: "r_ank_roll_link",
      leftUpperArm: "l_sho_roll_link",
      leftForearm: "l_el_link",
      rightUpperArm: "r_sho_roll_link",
      rightForearm: "r_el_link"
    },
    forward: "+x"
  }
};
function referencePose() {
  const p = new Float32Array(POSE_SIZE);
  p[2] = G1_BODY_TREE[0][2][2];
  p[6] = 1;
  p[7 + G1_JOINT_NAMES.indexOf("left_elbow_joint")] = Math.PI / 2;
  p[7 + G1_JOINT_NAMES.indexOf("right_elbow_joint")] = Math.PI / 2;
  return p;
}
function g1Skeleton() {
  const root = new Group();
  root.quaternion.setFromAxisAngle(new Vector3(1, 0, 0), -Math.PI / 2);
  const nodes = {};
  for (const [name, parent, position, quaternion] of G1_BODY_TREE) {
    const node = new Object3D();
    node.name = name;
    node.position.set(...position);
    node.quaternion.set(...quaternion);
    (parent ? nodes[parent] : root).add(node);
    nodes[name] = node;
  }
  return { root, robot: G1Robot.FromObject(root, { shadows: false }) };
}
function g1Source() {
  const g1 = g1Skeleton();
  return {
    root: g1.root,
    setPose: (pose) => g1.robot.setPose(pose),
    part: (part) => g1.root.getObjectByName(G1_PARTS[part]),
    reference: referencePose(),
    facing: [1, 0]
  };
}
function somaSource() {
  const k = SOMA_SKELETON_INFO, root = new Group();
  const joints = k.names.map((name) => Object.assign(new Object3D(), { name }));
  k.parents.forEach((p, j) => (p < 0 ? root : joints[p]).add(joints[j]));
  const setPose = (pose) => joints.forEach((o, j) => {
    o.position.set(pose[j * 7], pose[j * 7 + 1], pose[j * 7 + 2]);
    o.quaternion.set(pose[j * 7 + 3], pose[j * 7 + 4], pose[j * 7 + 5], pose[j * 7 + 6]);
  });
  const reference = k.referencePose();
  const heading2 = k.heading(reference);
  return { root, setPose, part: (part) => joints[k.index(SOMA_PARTS[part])], reference, facing: [Math.sin(heading2), Math.cos(heading2)] };
}
function sanitize(name) {
  return name.replace(/\s/g, "_").replace(/[[\].:/]/g, "");
}
const tq = new Quaternion(), tq2 = new Quaternion(), tv = new Vector3(), tv2 = new Vector3();
function depth(o) {
  let n = 0;
  for (let p = o.parent; p; p = p.parent) n++;
  return n;
}
class RetargetedCharacter {
  /**
   * @param model the loaded model, such as `gltf.scene`
   * @param rig one of CHARACTER_RIGS, or your own mapping
   * @param matchSize scale the model so its legs match the source's (default true); otherwise its steps slide
   * @param source the skeleton the motion comes from: 'g1' (default) or 'soma'
   */
  constructor(model, rig, { shadows = true, matchSize = true, source = "g1" } = {}) {
    this.object = new Group();
    this._parts = [];
    this._pelvisRest = new Vector3();
    this._g1PelvisRest = new Vector3();
    this._attach = [];
    this.model = model;
    this.rig = rig;
    this._src = source === "soma" ? somaSource() : g1Source();
    const lookup = (name) => {
      let node = model.getObjectByName(name);
      const wanted = sanitize(name), bare = sanitize(name.replace(/^[^:]*:/, ""));
      if (!node) model.traverse((o) => {
        if (!node && sanitize(o.name) === wanted) node = o;
      });
      if (!node) model.traverse((o) => {
        if (!node && sanitize(o.name) === bare) node = o;
      });
      return node;
    };
    const find = (name) => {
      const node = lookup(name);
      if (!node) throw new Error(`The ${rig.label} model has no node named ${name}`);
      return node;
    };
    const holder = new Group();
    holder.add(model);
    this.object.add(holder);
    model.updateMatrixWorld(true);
    for (const [name, euler] of Object.entries(rig.rest ?? {})) {
      const node = find(name);
      const turn = tq.setFromEuler(new Euler(...euler));
      const world = node.getWorldQuaternion(tq2).premultiply(turn);
      node.quaternion.copy(node.parent.getWorldQuaternion(new Quaternion()).invert().multiply(world));
      node.updateMatrixWorld(true);
    }
    const authored = { "+x": [1, 0], "-x": [-1, 0], "+z": [0, 1], "-z": [0, -1] }[rig.forward];
    const [fx, fz] = this._src.facing;
    const yaw = Math.atan2(fx, fz) - Math.atan2(authored[0], authored[1]);
    holder.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), yaw);
    this.object.updateMatrixWorld(true);
    const src = this._src;
    src.setPose(src.reference);
    src.root.updateMatrixWorld(true);
    const g1Part = (part) => src.part(part);
    const bones = rig.bones;
    const nodes = {};
    for (const part of Object.keys(G1_PARTS)) {
      const name = bones[part];
      if (!name) continue;
      const node = lookup(name);
      if (node) nodes[part] = node;
      else if (/pelvis|Thigh|Shin/.test(part)) throw new Error(`The ${rig.label} model has no node named ${name}`);
    }
    if (rig.autoRest !== false) {
      for (const [part, child] of LIMB_SEGMENTS) {
        const node = nodes[part], endName = rig.ends?.[part];
        const end = endName ? find(endName) : nodes[child];
        if (!node || !end) continue;
        const want = g1Part(child).getWorldPosition(tv).sub(g1Part(part).getWorldPosition(tv2)).normalize();
        const have = this._localPosition(end, new Vector3()).sub(this._localPosition(node, new Vector3())).normalize();
        const turn = new Quaternion().setFromUnitVectors(have, want);
        const desired = this._localQuaternion(node, new Quaternion()).premultiply(turn);
        const parentRotation = this._localQuaternion(node.parent, new Quaternion());
        node.quaternion.copy(parentRotation.invert().multiply(desired));
        node.updateMatrixWorld(true);
      }
    }
    const legLength = (a, b) => a.getWorldPosition(tv).distanceTo(b.getWorldPosition(tv2));
    let scale = 1;
    const ankle = rig.ends?.leftShin ? find(rig.ends.leftShin) : nodes.leftFoot;
    if (matchSize && nodes.leftThigh && ankle) {
      const g1Leg = legLength(g1Part("leftThigh"), g1Part("leftFoot"));
      scale = g1Leg / Math.max(1e-6, legLength(nodes.leftThigh, ankle));
      holder.scale.setScalar(scale);
      this.object.updateMatrixWorld(true);
    }
    this.scale = scale;
    for (const part of Object.keys(nodes)) {
      const node = nodes[part];
      const g1Node = g1Part(part);
      this._parts.push({ node, g1: g1Node, g1Rest: g1Node.getWorldQuaternion(new Quaternion()), rest: this._localQuaternion(node, new Quaternion()) });
    }
    this._parts.sort((a, b) => depth(a.node) - depth(b.node));
    this._pelvis = nodes.pelvis;
    this._root = rig.root ? find(rig.root) : this._pelvis;
    this._g1Pelvis = src.part("pelvis");
    this._localPosition(this._pelvis, this._pelvisRest);
    this._g1Pelvis.getWorldPosition(this._g1PelvisRest);
    for (const [name, target] of Object.entries(rig.attach ?? {})) this._attach.push([find(name), find(target)]);
    const materials = /* @__PURE__ */ new Set();
    model.traverse((o) => {
      const mesh = o;
      if (!mesh.isMesh) return;
      mesh.castShadow = mesh.receiveShadow = shadows;
      mesh.frustumCulled = false;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(m);
    });
    this.materials = [...materials];
  }
  /** A node's rotation in the space of `object`. */
  _localQuaternion(node, target) {
    return node.getWorldQuaternion(target).premultiply(this.object.getWorldQuaternion(tq2).invert());
  }
  _localPosition(node, target) {
    return this.object.worldToLocal(node.getWorldPosition(target));
  }
  /** Show a G1 pose: root XYZ, root XYZW quaternion, then the 29 joint angles. */
  setPose(pose) {
    this._src.setPose(pose);
    this._src.root.updateMatrixWorld(true);
    const objectRotation = this.object.getWorldQuaternion(new Quaternion());
    for (const part of this._parts) {
      const turn = part.g1.getWorldQuaternion(tq).multiply(tq2.copy(part.g1Rest).invert());
      const desired2 = turn.multiply(part.rest);
      const parentRotation = part.node.parent.getWorldQuaternion(tq2).premultiply(objectRotation.clone().invert());
      part.node.quaternion.copy(parentRotation.invert().multiply(desired2));
      part.node.updateMatrixWorld(true);
    }
    const g1Pelvis = this._g1Pelvis.getWorldPosition(tv);
    const desired = tv2.set(g1Pelvis.x, this._pelvisRest.y + (g1Pelvis.y - this._g1PelvisRest.y), g1Pelvis.z);
    const current = this._localPosition(this._pelvis, new Vector3());
    const shift = desired.sub(current);
    const root = this._root;
    const parent = root.parent;
    const a = this.object.localToWorld(new Vector3()), b = this.object.localToWorld(shift.clone());
    const local = parent.worldToLocal(b).sub(parent.worldToLocal(a));
    root.position.add(local);
    root.updateMatrixWorld(true);
    for (const [node, target] of this._attach) {
      node.position.copy(node.parent.worldToLocal(target.getWorldPosition(tv)));
      node.updateMatrixWorld(true);
    }
  }
}
const G1_RIG = {
  name: "g1",
  poseSize: POSE_SIZE,
  ground: (p) => [p[0], p[1]],
  heading: (p) => heading([p[3], p[4], p[5], p[6]]),
  place: placePoses,
  withSpeed,
  translate(p, dx, dy) {
    p[0] += dx;
    p[1] += dy;
  },
  lerp: (a, b, t, out) => lerpPose(a, b, t, out),
  distance(a, b) {
    let e = 0;
    for (let k = POSE_SIZE - 29; k < POSE_SIZE; k++) e += (a[k] - b[k]) ** 2;
    return e;
  }
};
const S = SOMA_SKELETON_INFO;
const SOMA_SIZE = S.J * 7;
function rotateY(q, o, angle, out, oo) {
  const s = Math.sin(angle / 2), c = Math.cos(angle / 2);
  const x = q[o], y = q[o + 1], z = q[o + 2], w = q[o + 3];
  out[oo] = c * x + s * z;
  out[oo + 1] = c * y + s * w;
  out[oo + 2] = c * z - s * x;
  out[oo + 3] = c * w - s * y;
}
const SOMA_RIG = {
  name: "soma",
  poseSize: SOMA_SIZE,
  ground: (p) => [p[0], -p[2]],
  // The model's hip heading is 0 facing +Z; controller angle 0 faces +X, which is a SOMA heading of 90 degrees.
  heading: (p) => S.heading(p) - Math.PI / 2,
  place(poses, x, y, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    return poses.map((src) => {
      const p = Float32Array.from(src);
      p[0] = src[0] * c + src[2] * s + x;
      p[2] = -src[0] * s + src[2] * c - y;
      rotateY(src, 3, angle, p, 3);
      return p;
    });
  },
  withSpeed(poses, speed) {
    const first = poses[0], last = poses[poses.length - 1];
    const current = Math.hypot(last[0] - first[0], last[2] - first[2]) * 30 / (poses.length - 1);
    const k = current > 1e-3 ? speed / current : 1;
    return poses.map((src) => {
      const p = Float32Array.from(src);
      p[0] = first[0] + (src[0] - first[0]) * k;
      p[2] = first[2] + (src[2] - first[2]) * k;
      return p;
    });
  },
  translate(p, dx, dy) {
    p[0] += dx;
    p[2] -= dy;
  },
  lerp(a, b, t, out) {
    for (let j = 0; j < S.J; j++) {
      const o = j * 7;
      for (let k = 0; k < 3; k++) out[o + k] = a[o + k] + (b[o + k] - a[o + k]) * t;
      const d = a[o + 3] * b[o + 3] + a[o + 4] * b[o + 4] + a[o + 5] * b[o + 5] + a[o + 6] * b[o + 6] < 0 ? -1 : 1;
      let l = 0;
      for (let k = 3; k < 7; k++) {
        out[o + k] = a[o + k] + (b[o + k] * d - a[o + k]) * t;
        l += out[o + k] * out[o + k];
      }
      l = Math.sqrt(l) || 1;
      for (let k = 3; k < 7; k++) out[o + k] /= l;
    }
    return out;
  },
  distance(a, b) {
    let e = 0;
    for (const j of S.modelToPose) {
      if (j === 0) continue;
      const o = j * 7 + 3;
      const d = a[o] * b[o] + a[o + 1] * b[o + 1] + a[o + 2] * b[o + 2] + a[o + 3] * b[o + 3];
      e += 1 - d * d;
    }
    return e;
  }
};
const NVIDIA_REPO = "https://raw.githubusercontent.com/NVlabs/motionbricks/main/assets/";
const NVIDIA_LFS = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/assets/";
const G1_PRIMITIVE_SOURCES = {
  documents: NVIDIA_REPO + "smart_primitives/unitree_g1/",
  /** Clips are in Git LFS, so they come from GitHub's media host; documents name them relative to this folder. */
  clips: NVIDIA_LFS + "smart_primitives/unitree_g1/"
};
const SOMA_PRIMITIVE_SOURCES = {
  documents: NVIDIA_REPO + "smart_primitives/soma/",
  clips: NVIDIA_LFS + "smart_primitives/soma/"
};
function qmul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}
function qnorm(q) {
  const l = Math.hypot(...q) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qrot(q, v) {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
const qinv = (q) => [-q[0], -q[1], -q[2], q[3]];
const qaxis = (axis, angle) => [axis[0] * Math.sin(angle / 2), axis[1] * Math.sin(angle / 2), axis[2] * Math.sin(angle / 2), Math.cos(angle / 2)];
const tmul = (a, b) => {
  const r = qrot(a.q, b.p);
  return { p: [r[0] + a.p[0], r[1] + a.p[1], r[2] + a.p[2]], q: qnorm(qmul(a.q, b.q)) };
};
const tinv = (a) => {
  const q = qinv(a.q), p = qrot(q, a.p);
  return { p: [-p[0], -p[1], -p[2]], q };
};
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const addScaled = (a, b, s) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
function placePose(t, pose) {
  const out = Float32Array.from(pose);
  const root = tmul(t, { p: [pose[0], pose[1], pose[2]], q: [pose[3], pose[4], pose[5], pose[6]] });
  out.set(root.p, 0);
  out.set(root.q, 3);
  return out;
}
const clipCache = /* @__PURE__ */ new Map();
async function loadClip(url, rig = "g1") {
  let clip = clipCache.get(url);
  if (!clip) {
    clip = fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`Could not download ${url}: HTTP ${r.status}`);
      if (rig === "soma") return parseSOMABvh(await r.text(), SOMA_SKELETON_INFO, 120);
      const rows = (await r.text()).trim().split("\n").slice(1);
      return rows.map((row) => {
        const v = row.split(",").map(Number);
        const p = new Float32Array(POSE_SIZE);
        p[0] = v[1] * 0.01;
        p[1] = v[2] * 0.01;
        p[2] = v[3] * 0.01;
        const d = Math.PI / 180;
        p.set(qnorm(qmul(qaxis([0, 0, 1], v[6] * d), qmul(qaxis([0, 1, 0], v[5] * d), qaxis([1, 0, 0], v[4] * d)))), 3);
        for (let j = 7; j < POSE_SIZE; j++) p[j] = v[j] * d;
        return p;
      });
    });
    clipCache.set(url, clip);
  }
  return clip;
}
function sampleClip(frames, seconds, rig = "g1") {
  const f = Math.min(Math.max(seconds * 120, 0), frames.length - 1);
  const i = Math.min(Math.floor(f), frames.length - 2), t = f - i;
  if (rig === "soma") return SOMA_RIG.lerp(frames[i], frames[i + 1], t, new Float32Array(frames[i].length));
  const a = frames[i], b = frames[i + 1], out = new Float32Array(POSE_SIZE);
  for (let k = 0; k < POSE_SIZE; k++) out[k] = a[k] * (1 - t) + b[k] * t;
  let qa = [a[3], a[4], a[5], a[6]], qb = [b[3], b[4], b[5], b[6]];
  if (dot([qa[0], qa[1], qa[2]], [qb[0], qb[1], qb[2]]) + qa[3] * qb[3] < 0) qb = qb.map((v) => -v);
  qa = qnorm([0, 1, 2, 3].map((k) => qa[k] * (1 - t) + qb[k] * t));
  out.set(qa, 3);
  return out;
}
const Y_TO_Z = { p: [0, 0, 0], q: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] };
const Z_TO_Y = { p: [0, 0, 0], q: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] };
async function loadG1Primitive(name, sources = G1_PRIMITIVE_SOURCES) {
  return loadPrimitive(name, { rig: "g1", sources });
}
async function loadPrimitive(name, { rig = "g1", sources } = {}) {
  sources ??= rig === "soma" ? SOMA_PRIMITIVE_SOURCES : G1_PRIMITIVE_SOURCES;
  const res = await fetch(sources.documents + name + ".smart_primitive.json");
  if (!res.ok) throw new Error(`Could not download the ${name} Smart Primitive: HTTP ${res.status}`);
  const doc = await res.json();
  const nodes = /* @__PURE__ */ new Map();
  const walk = (node, parentWorld, parent) => {
    const world = tmul(parentWorld, { p: node.local_transform.translation, q: node.local_transform.rotation_xyzw });
    nodes.set(node.id, { node, world, parent });
    for (const child of node.children ?? []) walk(child, world, node.id);
  };
  walk(doc.root, { p: [0, 0, 0], q: [0, 0, 0, 1] }, -1);
  let box = null;
  for (const { node, world } of nodes.values()) {
    const c = node.components.find((x) => x.type === "interaction_box");
    if (c && !box) box = { center: world.p, size: [c.width, c.depth, c.height] };
  }
  if (!box) throw new Error(`The ${name} Smart Primitive has no interaction box`);
  const pivotAbove = (id) => {
    for (let n = nodes.get(id); n; n = nodes.get(n.parent)) if (n.node.components.some((x) => x.type === "pivot")) return n.node.id;
    return -1;
  };
  const sequences = await Promise.all(doc.sequences.map(async (s) => {
    const own = nodes.has(s.pivot_node_id) ? s.pivot_node_id : -1;
    const pivotId = own >= 0 ? own : pivotAbove(s.node_ids[0]);
    const pivotNode = nodes.get(pivotId);
    const pc = pivotNode?.node.components.find((x) => x.type === "pivot");
    const pivot = pc && pivotNode ? {
      transform: pivotNode.world,
      motion: pc.motion_direction,
      translationAxis: pc.translation_constraint === "line_projection" ? AXES[pc.translation_axis] ?? null : null,
      lineMin: pc.line_min_offset,
      lineMax: pc.line_max_offset,
      rotationAxis: AXES[pc.rotation_axis] ?? null,
      coneHalfAngle: pc.rotation_constraint === "cone" ? pc.cone_half_angle_degrees * Math.PI / 180 : Infinity,
      coneCenters: pc.cone_side === "both" ? [0, Math.PI] : pc.cone_side === "mirrored" ? [Math.PI] : [0],
      rotationFirst: pc.operation_order === "rotation_then_translation"
    } : {
      transform: { p: [0, 0, 0], q: [0, 0, 0, 1] },
      motion: [0, 1, 0],
      translationAxis: null,
      lineMin: 0,
      lineMax: 0,
      rotationAxis: null,
      coneHalfAngle: Infinity,
      coneCenters: [0],
      rotationFirst: true
    };
    const toPivot = tinv(pivot.transform);
    const steps = await Promise.all(s.node_ids.map(async (id) => {
      const { node, world } = nodes.get(id);
      const c = node.components.find((x) => x.type === "target_pose");
      const frames = await loadClip(new URL(c.clip, sources.clips).href, rig);
      const start = c.playback_mode === "frames" ? c.frame_index / 120 : c.sample_time_seconds;
      const count = Math.min(Math.max(c.num_poses_to_use ?? POSES_PER_TOKEN, 1), POSES_PER_TOKEN);
      const toLocal = tmul(toPivot, world);
      const place = rig === "soma" ? tmul(toLocal, Y_TO_Z) : toLocal;
      const window = Array.from({ length: count }, (_, k) => placePose(place, sampleClip(frames, start + k / SAMPLE_RATE, rig)));
      const tokens = [];
      for (let t = c.min_tokens ?? 6; t <= (c.max_tokens ?? 14); t++) tokens.push(t);
      return { name: node.name, window, transitionEarly: c.transition_early_by_frames ?? 8, tokens };
    }));
    const w = steps[0].window;
    const startSpeed = w.length > 1 ? Math.hypot(w[1][0] - w[0][0], w[1][1] - w[0][1]) * SAMPLE_RATE : 0;
    return { name: s.name, startSpeed, steps, pivot, pivotId, aligns: own >= 0 };
  }));
  const names = sequences.map((q) => q.name);
  return { name, rig, kind: names.includes("Sit") && names.includes("Stand") ? "sit" : "pass", box, sequences };
}
const PROBE = { near: 0.15, minDepth: 0.75, lookahead: 1, extraMinSpeed: 2.3, extraTime: 0.25, width: 1 };
function probeSegment(p, d, speed) {
  const reach = speed * PROBE.lookahead + Math.max(0, speed - PROBE.extraMinSpeed) * PROBE.extraTime;
  return [addScaled(p, d, PROBE.near), addScaled(p, d, Math.max(PROBE.near + PROBE.minDepth, reach))];
}
function obstacleFrame(o) {
  const c = o.primitive.box.center;
  return tmul({ p: [o.x, o.y, 0], q: qaxis([0, 0, 1], o.yaw) }, { p: [-c[0], -c[1], 0], q: [0, 0, 0, 1] });
}
function closestPointOnSegment(a, b, p) {
  const s = sub(b, a), t = dot(sub(p, a), s) / (dot(s, s) + 1e-6);
  return addScaled(a, s, Math.min(Math.max(t, 0), 1));
}
function closestPointToSegment(a, b, c, d) {
  const ab = sub(b, a), cd = sub(d, c);
  const h = Math.hypot(...ab) / 2, oh = Math.hypot(...cd) / 2;
  const u = h > 0 ? [ab[0] / (2 * h), ab[1] / (2 * h), ab[2] / (2 * h)] : ab;
  const v = oh > 0 ? [cd[0] / (2 * oh), cd[1] / (2 * oh), cd[2] / (2 * oh)] : cd;
  const center = addScaled(a, u, h), otherCenter = addScaled(c, v, oh);
  const offset = sub(center, otherCenter), cosine = dot(u, v), along = dot(u, offset), otherAlong = dot(v, offset);
  const t = (cosine * otherAlong - along) / (1 - cosine * cosine + 1e-6);
  const point = addScaled(center, u, Math.min(Math.max(t, -h), h));
  const otherPoint = addScaled(otherCenter, v, Math.min(Math.max(otherAlong + cosine * t, -oh), oh));
  const projected = closestPointOnSegment(a, b, otherPoint), otherProjected = closestPointOnSegment(c, d, point);
  const d1 = sub(otherPoint, projected), d2 = sub(point, otherProjected);
  return dot(d1, d1) < dot(d2, d2) ? projected : point;
}
function alignedPivot(pivot, authored, p, d, speed) {
  let limits = authored;
  let position = authored.p, rotation = authored.q;
  const rotate = () => {
    if (!pivot.rotationAxis) return rotation;
    const axis = qrot(limits.q, pivot.rotationAxis);
    let motion = qrot(limits.q, pivot.motion);
    motion = addScaled(motion, axis, -dot(motion, axis));
    const desired = addScaled(d, axis, -dot(d, axis));
    if (Math.hypot(...motion) < 1e-6 || Math.hypot(...desired) < 1e-6) return rotation;
    const wanted = Math.atan2(dot(axis, cross(motion, desired)), dot(motion, desired));
    let angle = wanted;
    if (Number.isFinite(pivot.coneHalfAngle)) {
      let best = Infinity;
      for (const c of pivot.coneCenters) {
        const delta = wrap(wanted - c), clamped = Math.max(-pivot.coneHalfAngle, Math.min(pivot.coneHalfAngle, delta));
        if (Math.abs(delta - clamped) < best) {
          best = Math.abs(delta - clamped);
          angle = c + clamped;
        }
      }
    }
    return qnorm(qmul(qaxis(axis, angle), limits.q));
  };
  const slide = () => {
    if (!pivot.translationAxis) return position;
    const axis = qrot(limits.q, pivot.translationAxis);
    const [near, far] = probeSegment(p, d, speed);
    return closestPointToSegment(addScaled(limits.p, axis, pivot.lineMin), addScaled(limits.p, axis, pivot.lineMax), near, far);
  };
  if (pivot.rotationFirst) {
    rotation = rotate();
    if (pivot.rotationAxis) limits = { p: limits.p, q: rotation };
    position = slide();
  } else {
    position = slide();
    rotation = rotate();
  }
  return { p: position, q: rotation };
}
function placeSequence(o, sequence, x, y, angle, speed) {
  const authored = tmul(obstacleFrame(o), sequence.pivot.transform);
  let pivot = o.pivots?.get(sequence.pivotId) ?? authored;
  if (sequence.aligns) {
    pivot = alignedPivot(sequence.pivot, authored, [x, y, 0], [Math.cos(angle), Math.sin(angle), 0], speed);
    (o.pivots ??= /* @__PURE__ */ new Map()).set(sequence.pivotId, pivot);
  }
  const place = o.primitive.rig === "soma" ? tmul(Z_TO_Y, pivot) : pivot;
  return sequence.steps.map((step) => ({ ...step, window: step.window.map((pose) => placePose(place, pose)) }));
}
function probeHits(o, x, y, angle, speed, extra = 0) {
  const d = [Math.cos(angle), Math.sin(angle), 0];
  const [near, far0] = probeSegment([x, y, 0], d, speed);
  const far = addScaled(far0, d, extra);
  const frame = obstacleFrame(o), box = o.primitive.box;
  const centre = tmul(frame, { p: box.center, q: [0, 0, 0, 1] }).p;
  const ax = qrot(frame.q, [1, 0, 0]), ay = qrot(frame.q, [0, 1, 0]);
  const pc = [(near[0] + far[0]) / 2, (near[1] + far[1]) / 2, 0];
  const pu = [-d[1], d[0], 0];
  const ph = [PROBE.width / 2, Math.hypot(far[0] - near[0], far[1] - near[1]) / 2];
  const bh = [box.size[0] / 2, box.size[1] / 2];
  const between = sub(centre, pc);
  for (const axis of [pu, d, ax, ay]) {
    const r1 = ph[0] * Math.abs(dot(pu, axis)) + ph[1] * Math.abs(dot(d, axis));
    const r2 = bh[0] * Math.abs(dot(ax, axis)) + bh[1] * Math.abs(dot(ay, axis));
    if (Math.abs(dot(between, axis)) > r1 + r2) return false;
  }
  return true;
}
const FADE_FRAMES = 8;
const GAIT_SPEEDS = { walk: 1.13, run: 3.43 };
class G1Controller {
  constructor(engine, bricks, x = 0, y = 0, angle = 0, rig = G1_RIG) {
    this.goal = null;
    this.gait = "walk";
    this.obstacles = [];
    this.seat = null;
    this._action = null;
    this.lead = 1;
    this.stats = { generations: 0, lastInferenceMs: 0, averageInferenceMs: 0, lastTokenCount: 0 };
    this._segment = null;
    this._frame = 0;
    this._request = null;
    this._requestId = 0;
    this._dirty = false;
    this._pending = null;
    this._retryIn = 0;
    this._fade = null;
    this.engine = engine;
    this.bricks = bricks;
    this.rig = rig;
    this._segmentPose = new Float32Array(rig.poseSize);
    this._fadePose = new Float32Array(rig.poseSize);
    this.pose = rig.place(bricks.rest, x, y, angle)[0];
  }
  /** Walk or run speed in m/s. */
  speedOf(gait) {
    return (this.bricks.speeds ?? GAIT_SPEEDS)[gait];
  }
  get moving() {
    return this._segment !== null;
  }
  get generating() {
    return this._request !== null;
  }
  /**
   * The sequence being played (a jump, a vault, sitting down, or one given to perform), or null: its name, primitive,
   * the step whose motion is playing, and that step's index.
   */
  get action() {
    const seq = this._segment?.sequence;
    return seq ? { name: seq.name ?? "", primitive: seq.obstacle?.primitive ?? null, step: seq.steps[seq.next - 1]?.name ?? "", index: seq.next - 1 } : null;
  }
  /** Whether a sequence is asked for, generating or playing, so that new goals wait for it. */
  get busy() {
    const seq = this._pending?.request.sequence ?? this._request?.sequence ?? this._segment?.sequence;
    return !!this._action || !!seq && (seq.next < seq.steps.length || !!seq.hold && this._segment?.sequence === seq && this._frame < this._segment.poses.length - 1);
  }
  /**
   * Play target steps in controller space, one generation each: poses authored at the origin facing +X and placed
   * with rig.place, such as the human's kick or reach (SOMAActions). With `hold`, the motion ends on the last step
   * and stays there; otherwise the robot goes on to its goal, or stops, after it. The steps start from the next
   * generation, so the motion already planned plays into them.
   */
  perform(steps, { name = "", hold = true } = {}) {
    this.goal = null;
    this._action = { name, steps, next: 0, hold };
    const playing = this._segment?.sequence;
    if (this._request || this._pending) this._dirty = true;
    else if (!playing || playing.next >= playing.steps.length) this._generate(true);
  }
  /** The segment being played: its length and transition point in frames, and the playhead. */
  get playback() {
    const seg = this._segment;
    if (!seg) return null;
    const frames = seg.poses.length;
    return { frames, frame: Math.min(this._frame, frames - 1), transition: frames - 1 - seg.transitionEarly, stopping: seg.stopping };
  }
  /** Walk or run to a ground point in G1 space, then stop there. */
  moveTo(x, y, gait = this.gait) {
    this.goal = [x, y];
    this.gait = gait;
    this._regenerate();
  }
  /**
   * Sit on a 'sit' obstacle: walk the last metre to it, turn and sit, then stay seated. Start within about two
   * metres of it. Moving again stands up first.
   */
  sit(o) {
    const sequence = o.primitive.sequences.find((q) => q.name === "Sit");
    if (!sequence || this.seat) return;
    const [px, py] = this.rig.ground(this.pose);
    const angle = Math.atan2(o.y - py, o.x - px);
    const steps = placeSequence(o, sequence, px, py, angle, this.speedOf("walk"));
    this.goal = null;
    this._action = { steps, next: 0, hold: true, seat: o, obstacle: o };
    if (!this._request && !this._pending) this._generate(true);
  }
  /** Stand up from the seat and stay there. */
  standUp() {
    const o = this.seat;
    const sequence = o?.primitive.sequences.find((q) => q.name === "Stand");
    if (!o || !sequence) return;
    this.goal = null;
    const [px, py] = this.rig.ground(this.pose);
    this._action = { steps: placeSequence(o, sequence, px, py, 0, 0), next: 0, hold: true };
    if (!this._request && !this._pending) this._generate(true);
  }
  /** Come to a stop a short way ahead. */
  stop() {
    this.goal = null;
    if (this._segment || this._request || this._pending) this._regenerate();
  }
  /**
   * Generate for the new goal now, or once the motion in flight is in place: a request's start frame
   * belongs to the segment it was made from, so none is made while another waits to take over.
   */
  _regenerate() {
    const seq = this._pending?.request.sequence ?? this._request?.sequence ?? this._segment?.sequence;
    if (seq && seq.next < seq.steps.length || this._action) return;
    if (this._request || this._pending) this._dirty = true;
    else this._generate(true);
  }
  /** Advance playback by dt seconds and update pose. */
  update(dt) {
    this._retryIn = Math.max(0, this._retryIn - dt);
    const seg = this._segment;
    if (this._fade) {
      this._fade.frame += dt * SAMPLE_RATE;
      this._fade.t += dt * SAMPLE_RATE;
    }
    if (seg) {
      this._frame = Math.min(this._frame + dt * SAMPLE_RATE, seg.poses.length - 1);
      if (this._pending && this._frame >= this._pending.request.startFrame) this._switch();
    } else if (this._pending) this._switch();
    if (this._dirty && !this._request && !this._pending) this._generate(true);
    const current = this._segment;
    if (!current) return;
    const last = current.poses.length - 1;
    samplePoses(this.rig, current.poses, this._frame, this._segmentPose);
    const fade = this._fade;
    if (fade) {
      const w = Math.min(fade.t / FADE_FRAMES, 1);
      samplePoses(this.rig, fade.segment.poses, fade.frame, this._fadePose);
      this.rig.lerp(this._fadePose, this._segmentPose, w * w * (3 - 2 * w), this.pose);
      if (w >= 1) this._fade = null;
    } else this.pose.set(this._segmentPose);
    if (current.stopping && this._frame >= last && !this._pending && !this._request) {
      this._segment = null;
      this._fade = null;
      if (this.goal && this._distanceToGoal() < 0.35) this.goal = null;
      else if (this.goal) this._generate(true);
      return;
    }
    if (!this._request && !this._pending && !current.stopping && this._retryIn <= 0) {
      const transition = last - current.transitionEarly;
      if (this._frame + this._latencyFrames() * 1.5 >= transition) this._generate(false);
    }
  }
  _latencyFrames() {
    const s = this.stats;
    return Math.ceil(Math.max(s.averageInferenceMs || 400, s.lastInferenceMs) / 1e3 * SAMPLE_RATE) + 2;
  }
  /**
   * Frames before a gait segment's end where the next one takes over. With stride cycles, a run hands over 6
   * frames early and a walk 2, so the hand-overs fall on alternate feet rather than always on the same step.
   * Without them, a run keeps its cadence with 10. Slow inference gets 2: more time to generate.
   */
  _transitionEarly(gait) {
    if (gait !== "run" || Math.max(this.stats.averageInferenceMs, this.stats.lastInferenceMs) >= 600) return 2;
    return this.bricks.cycles?.run ? 6 : 10;
  }
  /**
   * A jump to start now: the nearest obstacle the probe meets whose first target is no more than `lead` seconds of
   * travel ahead, with its steps placed for this approach. When the robot is already past the first target, it
   * starts at the next one still ahead.
   */
  _findJump(from, angle, speed, distance) {
    let best, bestAhead = Infinity;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const [fx, fy] = this.rig.ground(from);
    const ahead = (p) => {
      const [x, y] = this.rig.ground(p);
      return (x - fx) * cos + (y - fy) * sin;
    };
    for (const o of this.obstacles) {
      if (o.primitive.kind !== "pass" || !probeHits(o, fx, fy, angle, speed, speed * this.lead + 3)) continue;
      const sequence = o.primitive.sequences.reduce((a, b) => Math.abs(b.startSpeed - speed) < Math.abs(a.startSpeed - speed) ? b : a);
      const steps = placeSequence(o, sequence, fx, fy, angle, speed);
      const entry = ahead(steps[0].window[0]);
      if (entry > speed * this.lead + 0.3 || entry >= bestAhead) continue;
      const first = steps.findIndex((s) => ahead(s.window[0]) > 0.3);
      if (first < 0 || ahead(steps[steps.length - 1].window[0]) > distance + 0.5) continue;
      best = { steps, next: first, obstacle: o };
      bestAhead = entry;
    }
    return best;
  }
  _distanceToGoal() {
    if (!this.goal) return 0;
    const [x, y] = this.rig.ground(this.pose);
    return Math.hypot(this.goal[0] - x, this.goal[1] - y);
  }
  /** The frame of the current segment where new motion can start, and the four context poses ending there. */
  _contextAt(now) {
    const seg = this._segment;
    if (!seg) {
      const p = Float32Array.from(this.pose);
      return { startFrame: -1, context: [p, p, p, p] };
    }
    const last = seg.poses.length - 1;
    const soonest = Math.ceil(this._frame) + this._latencyFrames();
    const startFrame = Math.min(last, now ? soonest : Math.max(soonest, last - seg.transitionEarly));
    const frames = [...seg.prefix, ...seg.poses];
    const end = startFrame + seg.prefix.length;
    return { startFrame, context: frames.slice(end - 3, end + 1) };
  }
  _generate(now) {
    const { startFrame, context } = this._contextAt(now);
    const from = context[3];
    const speed = this.speedOf(this.gait);
    const [fx, fy] = this.rig.ground(from);
    const facing = this.rig.heading(from);
    const stopAhead = this._segment ? 0.5 * speed : 0;
    const goal = this.goal ?? [fx + Math.cos(facing) * stopAhead, fy + Math.sin(facing) * stopAhead];
    const dx = goal[0] - fx, dy = goal[1] - fy;
    const distance = Math.hypot(dx, dy);
    const angle = distance > 0.3 ? Math.atan2(dy, dx) : facing;
    let targets;
    let stopping;
    let options;
    let sequence;
    const lead = this.lead;
    const cycle2 = this.bricks.cycles?.[this.gait];
    const playing = this._segment?.sequence;
    let jump = playing && playing.next < playing.steps.length ? playing : void 0;
    if (!jump && this._action) {
      jump = this._action;
      this._action = null;
    }
    if (!jump && this.seat && this.goal) {
      const stand = this.seat.primitive.sequences.find((q) => q.name === "Stand");
      if (stand) jump = { steps: placeSequence(this.seat, stand, fx, fy, 0, 0), next: 0 };
    }
    if (!jump && this._segment && !this._segment.stopping && distance > 1) jump = this._findJump(from, angle, speed, distance);
    if (jump) {
      const step = jump.steps[jump.next];
      targets = step.window;
      options = { tokens: step.tokens };
      sequence = { ...jump, next: jump.next + 1 };
      stopping = !!jump.hold && jump.next === jump.steps.length - 1;
    } else if (distance > speed * lead + 0.8 && cycle2 && this._segment && !this._segment.stopping) {
      const room = this._latencyFrames() + this._transitionEarly(this.gait) + 2;
      const tokens = Math.min(DEFAULT_TOKEN_COUNTS[DEFAULT_TOKEN_COUNTS.length - 1], Math.max(
        Math.round((lead * SAMPLE_RATE + POSES_PER_TOKEN + BOUNDARY_FRAMES) / POSES_PER_TOKEN),
        Math.ceil((room + POSES_PER_TOKEN + BOUNDARY_FRAMES) / POSES_PER_TOKEN)
      ));
      const frames = Math.round(segmentSeconds(tokens) * SAMPLE_RATE);
      const ahead = speed * frames / SAMPLE_RATE;
      const window = cycleWindow(this.rig, cycle2, cyclePhase(this.rig, cycle2, context) + frames);
      targets = this.rig.place(this.rig.withSpeed(window, speed), fx + Math.cos(angle) * ahead, fy + Math.sin(angle) * ahead, angle);
      options = { tokens: [tokens] };
      stopping = false;
    } else if (distance > speed * lead + 0.8) {
      const ahead = speed * lead;
      targets = this.rig.place(this.rig.withSpeed(this.bricks[this.gait], speed), fx + Math.cos(angle) * ahead, fy + Math.sin(angle) * ahead, angle);
      stopping = false;
    } else {
      if (!this._segment && distance < 0.35) {
        this.goal = null;
        this._dirty = false;
        return;
      }
      targets = this.rig.place(this.bricks.rest, goal[0], goal[1], angle);
      stopping = true;
      const moving = this._segment && !this._segment.stopping;
      const need = moving ? Math.min(2 * distance / speed, segmentSeconds(DEFAULT_TOKEN_COUNTS[DEFAULT_TOKEN_COUNTS.length - 1])) * 0.8 : 0;
      const tokens = DEFAULT_TOKEN_COUNTS.filter((t) => segmentSeconds(t) >= need);
      if (tokens.length < DEFAULT_TOKEN_COUNTS.length) options = { tokens };
    }
    const request = {
      id: ++this._requestId,
      segment: this._segment,
      startFrame,
      stopping,
      gait: this.gait,
      sequence,
      transitionEarly: sequence ? sequence.steps[sequence.next - 1].transitionEarly : void 0
    };
    this._request = request;
    this._dirty = false;
    this.engine.generate(context, targets, options).then((motion) => {
      const s = this.stats;
      s.generations++;
      s.lastInferenceMs = motion.inferenceMs;
      s.lastTokenCount = motion.tokenCount;
      s.averageInferenceMs = s.generations <= 1 ? motion.inferenceMs : s.averageInferenceMs * 0.7 + motion.inferenceMs * 0.3;
      this._request = null;
      if (this._dirty) {
        this._generate(true);
        return;
      }
      this._pending = { request, poses: motion.poses };
      this.onGenerated?.(motion.poses, targets);
    }).catch((e) => {
      this._request = null;
      this._retryIn = 1;
      this.onError?.(e);
    });
  }
  _switch() {
    const { request, poses } = this._pending;
    this._pending = null;
    const seg = this._segment;
    if (request.segment !== seg) {
      this._dirty = true;
      return;
    }
    if (request.sequence && request.sequence.next === 1) this.seat = request.sequence.seat ?? null;
    let prefix;
    if (seg && request.startFrame >= 0) {
      const frames = [...seg.prefix, ...seg.poses];
      const end = request.startFrame + seg.prefix.length;
      prefix = frames.slice(end - 3, end);
      this._fade = { segment: seg, frame: this._frame, t: 0 };
      this._frame = Math.max(0, this._frame - request.startFrame);
    } else {
      const p = Float32Array.from(poses[0]);
      prefix = [p, p, p];
      this._frame = 0;
      this._fade = null;
    }
    this._segment = {
      poses,
      prefix,
      transitionEarly: request.stopping ? 0 : request.transitionEarly ?? this._transitionEarly(request.gait),
      stopping: request.stopping,
      sequence: request.sequence
    };
  }
}
function segmentSeconds(tokens) {
  return (tokens * POSES_PER_TOKEN - POSES_PER_TOKEN - BOUNDARY_FRAMES) / SAMPLE_RATE;
}
function cyclePhase(rig, cycle2, context) {
  const frames = cycle2.frames, n = frames.length;
  let best = 0, bestError = Infinity;
  for (let i = 0; i < n; i++) {
    let error = 0;
    for (let c = 0; c < 2; c++) error += rig.distance(context[2 + c], frames[(i - 1 + c + n) % n]);
    if (error < bestError) [best, bestError] = [i, error];
  }
  return best;
}
function cycleWindow(rig, cycle2, start) {
  const n = cycle2.frames.length;
  const window = [0, 1, 2, 3].map((f) => {
    const i = start + f, stride = Math.floor(i / n);
    const p = Float32Array.from(cycle2.frames[(i % n + n) % n]);
    rig.translate(p, stride * cycle2.travel[0], stride * cycle2.travel[1]);
    return p;
  });
  const [x, y] = rig.ground(window[0]);
  for (const p of window) rig.translate(p, -x, -y);
  return window;
}
function samplePoses(rig, poses, frame, out) {
  const last = poses.length - 1;
  const f = Math.min(Math.max(frame, 0), last);
  const i = Math.floor(f);
  return rig.lerp(poses[i], poses[Math.min(i + 1, last)], f - i, out);
}
const G1_BRICKS = {
  rest: [
    [0, 0, 0.7884, -276e-5, 489e-5, 1e-5, 0.99998, -0.0874, 0.01094, -0.0211, 0.18969, -0.14284, 0.01954, 0.02128, -0.07084, -0.39641, 0.11388, -0.1404, 0.01166, -0.023, 0.04438, 0.0147, -0.08917, 0.37069, -0.74259, 1.0747, 0.30954, 0.01407, -0.14119, 0.11575, -0.35386, 0.39509, 1.1243, -0.48645, -0.07098, 0.02357]
  ],
  walk: [
    [0, 0, 0.77057, 0.01298, -0.06268, 0.05402, 0.99649, 0.25511, -0.11507, 0.21825, 0.12879, -0.23856, 0.11158, -0.25851, -0.12985, -0.19801, 0.7773, -0.20813, -0.05075, 34e-4, 133e-5, 0.26222, -0.14629, 0.43066, -0.54291, 0.95727, 0.14859, 0.03842, 0.12112, 0.06329, -0.40823, 0.63706, 1.1622, -0.09862, 0.06619, -0.11199],
    [0.0293, 84e-5, 0.76821, 0.01185, -0.05596, 0.0623, 0.99642, 0.29217, -0.1172, 0.18608, 0.11767, -0.25987, 0.10153, -0.27461, -0.1004, -0.20898, 0.55595, -0.17863, -0.02236, -0.01612, 0.01001, 0.25665, -0.19288, 0.44718, -0.53701, 0.91354, 0.15732, 0.02806, 0.11062, 0.09766, -0.41923, 0.623, 1.1847, -0.10005, 0.06615, -0.10829],
    [0.05867, 87e-5, 0.76484, 0.01141, -0.05025, 0.07069, 0.99617, 0.32141, -0.1142, 0.15726, 0.12452, -0.28635, 0.09423, -0.27923, -0.06427, -0.20675, 0.34161, -0.13952, 0.01345, -0.03355, 0.01716, 0.25283, -0.22598, 0.46607, -0.52963, 0.85919, 0.1777, 0.01854, 0.11326, 0.12285, -0.42989, 0.6136, 1.2016, -0.10363, 0.07117, -0.09421],
    [0.08791, -0, 0.76068, 985e-5, -0.04617, 0.07763, 0.99586, 0.33902, -0.10313, 0.13317, 0.1636, -0.32359, 0.08806, -0.27529, -0.03751, -0.1956, 0.22078, -0.10744, 0.02224, -0.0435, 0.02556, 0.24985, -0.24419, 0.47906, -0.51888, 0.81263, 0.19963, 0.01877, 0.1291, 0.14538, -0.43533, 0.60899, 1.2147, -0.1083, 0.0791, -0.0706]
  ],
  run: [
    [0, 0, 0.75084, -0.01078, 0.01472, 351e-5, 0.99983, 0.28824, -0.05099, 0.25315, 0.36089, -0.3966, 0.0529, -0.84622, -0.10825, -0.13301, 1.1517, -0.36435, -0.07737, -0.23915, 0.14721, 0.16533, -0.43219, 0.78799, -0.37376, -0.77974, -0.09114, -0.08763, -0.13694, 0.84197, -0.63569, 0.05403, -0.51558, -0.26001, -0.18616, -0.03518],
    [0.12661, 5e-5, 0.76182, -857e-5, 0.02143, 0.0207, 0.99952, 0.4203, -0.04548, 0.23341, 0.32966, -0.08398, 0.01685, -0.797, -0.04224, -0.17504, 0.71178, -0.32534, -308e-5, -0.29255, 0.1601, 0.1092, -0.4969, 0.80401, -0.41278, -0.88569, -0.10467, -0.08707, -0.10015, 1.025, -0.63539, -0.02773, -0.51078, -0.23819, -0.15056, -0.03714],
    [0.25117, -7e-5, 0.76973, 829e-5, 0.01691, 0.03155, 0.99932, 0.43796, -0.04869, 0.23421, 0.46656, 0.11715, -484e-5, -0.7125, -642e-5, -0.1864, 0.36121, -0.30716, 0.04283, -0.31776, 0.1311, 0.08382, -0.47495, 0.78964, -0.44443, -0.96703, -0.09534, -0.11085, -0.07239, 1.1132, -0.6039, -0.0726, -0.51707, -0.19738, -0.13614, -0.04719],
    [0.3706, 0, 0.76923, 0.02229, 566e-5, 0.0366, 0.99907, 0.38521, -0.02289, 0.24802, 0.7865, 0.08637, -0.01123, -0.63147, -0.014, -0.18089, 0.24699, -0.30509, 0.04118, -0.32672, 0.0897, 0.08926, -0.36908, 0.74523, -0.46101, -0.99078, -0.06127, -0.13704, -0.08494, 1.0975, -0.5722, -0.04173, -0.54269, -0.14907, -0.13626, -0.04656]
  ],
  // One stride of each gait, made exactly left/right symmetric by tools/make_cycles.py from the model's own
  // steady output. The controller takes its gait targets from these, at the point of the stride it will reach.
  cycles: {
    walk: {
      travel: [1.22916, 0],
      frames: [
        [0, 0.03031, 0.76907, 0.03199, -0.04743, -0.01124, 0.9983, 0.13924, -0.21304, 0.16351, 0.09255, -0.17222, 0.16854, -0.37563, -0.15701, -0.1766, 0.97396, -0.17262, -0.08102, -0.01876, -0.01667, 0.16711, -0.14064, 0.35274, -0.54883, 0.9609, 0.1375, 0.08183, 0.18197, 0.10723, -0.32806, 0.60142, 1.12967, -0.15293, 0.09778, -0.10878],
        [0.04229, 0.03001, 0.7661, 0.03429, -0.04435, 271e-5, 0.99842, 0.22099, -0.23127, 0.13651, 0.03664, -0.19297, 0.16676, -0.4224, -0.15924, -0.19471, 0.74088, -0.15435, -0.06278, -0.05907, -545e-5, 0.16444, -0.20335, 0.38347, -0.54884, 0.88075, 0.13771, 0.07234, 0.18617, 0.17123, -0.33408, 0.5735, 1.14455, -0.14832, 0.10155, -0.1054],
        [0.07965, 0.02868, 0.76074, 0.03955, -0.04097, 0.01509, 0.99826, 0.27599, -0.24391, 0.11936, 0.0158, -0.22058, 0.16171, -0.4472, -0.16521, -0.20831, 0.50031, -0.12251, -0.03578, -0.09161, -672e-5, 0.16113, -0.25857, 0.41516, -0.55123, 0.81355, 0.13644, 0.06229, 0.1857, 0.22437, -0.33856, 0.55495, 1.14969, -0.14405, 0.10149, -0.10146],
        [0.1159, 0.02645, 0.75444, 0.04425, -0.03793, 0.02549, 0.99797, 0.31912, -0.24791, 0.10712, 0.01653, -0.25225, 0.1517, -0.45699, -0.16401, -0.21641, 0.28032, -0.09041, -47e-4, -0.11789, -0.01129, 0.15606, -0.30526, 0.44546, -0.55751, 0.75971, 0.12957, 0.05486, 0.18283, 0.26669, -0.33732, 0.54705, 1.1585, -0.13867, 0.1003, -0.09656],
        [0.15292, 0.0235, 0.74799, 0.04548, -0.03585, 0.03312, 0.99777, 0.35508, -0.24065, 0.10039, 0.0406, -0.28909, 0.13503, -0.46247, -0.15245, -0.2193, 0.14211, -0.06822, 0.01564, -0.13836, -0.01134, 0.1501, -0.33647, 0.46855, -0.56201, 0.71958, 0.12246, 0.0492, 0.18294, 0.29967, -0.32945, 0.54599, 1.17431, -0.13733, 0.10088, -0.09142],
        [0.19114, 0.01996, 0.74168, 0.04054, -0.0342, 0.03678, 0.99791, 0.38184, -0.21735, 0.1006, 0.09751, -0.33258, 0.11245, -0.47385, -0.12819, -0.22142, 0.12798, -0.05789, 488e-5, -0.15283, -38e-5, 0.14446, -0.34031, 0.48157, -0.55689, 0.69371, 0.11803, 0.04465, 0.18975, 0.32068, -0.314, 0.54543, 1.19563, -0.14428, 0.10296, -0.08279],
        [0.23047, 0.01593, 0.73613, 0.02808, -0.0323, 0.03731, 0.99839, 0.39627, -0.17294, 0.10954, 0.19316, -0.38144, 0.08657, -0.49015, -0.09092, -0.22662, 0.20962, -0.03395, -0.0363, -0.1616, 0.02291, 0.13777, -0.31541, 0.48303, -0.54201, 0.68161, 0.11608, 0.04389, 0.19781, 0.32836, -0.29449, 0.5402, 1.21886, -0.1556, 0.10652, -0.07083],
        [0.26753, 0.01178, 0.73248, 0.01166, -0.03046, 0.03756, 0.99876, 0.3976, -0.1159, 0.12029, 0.30844, -0.42005, 0.06078, -0.50443, -0.05136, -0.23599, 0.31887, 408e-5, -0.08465, -0.16637, 0.05212, 0.13298, -0.28059, 0.47242, -0.52834, 0.68794, 0.11362, 0.04448, 0.20427, 0.32316, -0.28096, 0.53414, 1.23813, -0.16207, 0.11127, -0.06188],
        [0.29967, 774e-5, 0.73163, -322e-5, -0.02958, 0.03872, 0.99881, 0.38316, -0.06234, 0.12301, 0.43038, -0.41563, 0.04019, -0.50894, -0.01947, -0.242, 0.40749, 0.02175, -0.11459, -0.16771, 0.07855, 0.13305, -0.2533, 0.45117, -0.52953, 0.71527, 0.11103, 0.04238, 0.21201, 0.30616, -0.27947, 0.53393, 1.2496, -0.15857, 0.11433, -0.06309],
        [0.3356, 264e-5, 0.73412, -0.01706, -0.03182, 0.03955, 0.99857, 0.33475, -121e-5, 0.12153, 0.61826, -0.31692, 0.02071, -0.48386, 0.01572, -0.2471, 0.46673, 435e-5, -0.13412, -0.16397, 0.10082, 0.13891, -0.22599, 0.42201, -0.54592, 0.76651, 0.11101, 0.03873, 0.22185, 0.27603, -0.28226, 0.53658, 1.25528, -0.14937, 0.11762, -0.0697],
        [0.37921, -436e-5, 0.73989, -0.03158, -0.03678, 0.041, 0.99798, 0.2356, 0.08065, 0.1181, 0.88063, -0.14234, 709e-5, -0.42089, 0.06422, -0.25594, 0.48594, -0.04003, -0.15596, -0.15647, 0.11886, 0.15084, -0.19588, 0.39132, -0.57391, 0.84185, 0.1165, 0.03672, 0.22724, 0.22786, -0.28326, 0.54089, 1.25148, -0.14217, 0.1213, -0.07857],
        [0.42588, -0.0127, 0.74761, -0.04522, -0.04128, 0.04359, 0.99717, 0.09978, 0.16697, 0.11429, 1.12322, -0.04737, 0.01505, -0.33628, 0.11876, -0.26463, 0.46608, -0.09062, -0.17657, -0.14469, 0.12879, 0.16738, -0.16606, 0.366, -0.60849, 0.93405, 0.13101, 0.0381, 0.22189, 0.15875, -0.28391, 0.54985, 1.2332, -0.14376, 0.1198, -0.0952],
        [0.46993, -0.02059, 0.75573, -0.05201, -0.04486, 0.04439, 0.99665, -0.03419, 0.22153, 0.11611, 1.26952, -0.07545, 0.04005, -0.24038, 0.16234, -0.26216, 0.41309, -0.12654, -0.18943, -0.12404, 0.12072, 0.18566, -0.13157, 0.34719, -0.63962, 1.02684, 0.1512, 0.04288, 0.20853, 0.0808, -0.28513, 0.5625, 1.20275, -0.15555, 0.11115, -0.12288],
        [0.50744, -0.02589, 0.76338, -0.04903, -0.04817, 0.04143, 0.99677, -0.13919, 0.22927, 0.12652, 1.29876, -0.12748, 0.06071, -0.13525, 0.18606, -0.24578, 0.33032, -0.13972, -0.19129, -0.09685, 0.09199, 0.19922, -0.08631, 0.33503, -0.65605, 1.09745, 0.16573, 0.05306, 0.19337, 0.01124, -0.28786, 0.57373, 1.16331, -0.17062, 0.09979, -0.15404],
        [0.54289, -0.02872, 0.77005, -0.04125, -0.05032, 0.03441, 0.99729, -0.22055, 0.2093, 0.14647, 1.24049, -0.16085, 0.07147, -0.02483, 0.19614, -0.22103, 0.23309, -0.13875, -0.18526, -0.06482, 0.0544, 0.20347, -0.02716, 0.32756, -0.65566, 1.14249, 0.16865, 0.06578, 0.1764, -0.05021, -0.2964, 0.57839, 1.11282, -0.17973, 0.0885, -0.17952],
        [0.57874, -0.02998, 0.77435, -0.03411, -0.04995, 0.02348, 0.99789, -0.28775, 0.18158, 0.17458, 1.12234, -0.17876, 0.07479, 0.07741, 0.20141, -0.19384, 0.13998, -0.13528, -0.17434, -0.02798, 0.02025, 0.20078, 0.04041, 0.32609, -0.64103, 1.16575, 0.1628, 0.07765, 0.1559, -0.10614, -0.31472, 0.57769, 1.05029, -0.17888, 0.07945, -0.19254],
        [0.61458, -0.03031, 0.76907, -0.03199, -0.04743, 0.01124, 0.9983, -0.37563, 0.15701, 0.1766, 0.97396, -0.17262, 0.08102, 0.13924, 0.21304, -0.16351, 0.09255, -0.17222, -0.16854, 0.01876, 0.01667, 0.16711, 0.10723, 0.32806, -0.60142, 1.12967, 0.15293, 0.09778, 0.10878, -0.14064, -0.35274, 0.54883, 0.9609, -0.1375, 0.08183, -0.18197],
        [0.65687, -0.03001, 0.7661, -0.03429, -0.04435, -271e-5, 0.99842, -0.4224, 0.15924, 0.19471, 0.74088, -0.15435, 0.06278, 0.22099, 0.23127, -0.13651, 0.03664, -0.19297, -0.16676, 0.05907, 545e-5, 0.16444, 0.17123, 0.33408, -0.5735, 1.14455, 0.14832, 0.10155, 0.1054, -0.20335, -0.38347, 0.54884, 0.88075, -0.13771, 0.07234, -0.18617],
        [0.69423, -0.02868, 0.76074, -0.03955, -0.04097, -0.01509, 0.99826, -0.4472, 0.16521, 0.20831, 0.50031, -0.12251, 0.03578, 0.27599, 0.24391, -0.11936, 0.0158, -0.22058, -0.16171, 0.09161, 672e-5, 0.16113, 0.22437, 0.33856, -0.55495, 1.14969, 0.14405, 0.10149, 0.10146, -0.25857, -0.41516, 0.55123, 0.81355, -0.13644, 0.06229, -0.1857],
        [0.73048, -0.02645, 0.75444, -0.04425, -0.03793, -0.02549, 0.99797, -0.45699, 0.16401, 0.21641, 0.28032, -0.09041, 47e-4, 0.31912, 0.24791, -0.10712, 0.01653, -0.25225, -0.1517, 0.11789, 0.01129, 0.15606, 0.26669, 0.33732, -0.54705, 1.1585, 0.13867, 0.1003, 0.09656, -0.30526, -0.44546, 0.55751, 0.75971, -0.12957, 0.05486, -0.18283],
        [0.7675, -0.0235, 0.74799, -0.04548, -0.03585, -0.03312, 0.99777, -0.46247, 0.15245, 0.2193, 0.14211, -0.06822, -0.01564, 0.35508, 0.24065, -0.10039, 0.0406, -0.28909, -0.13503, 0.13836, 0.01134, 0.1501, 0.29967, 0.32945, -0.54599, 1.17431, 0.13733, 0.10088, 0.09142, -0.33647, -0.46855, 0.56201, 0.71958, -0.12246, 0.0492, -0.18294],
        [0.80572, -0.01996, 0.74168, -0.04054, -0.0342, -0.03678, 0.99791, -0.47385, 0.12819, 0.22142, 0.12798, -0.05789, -488e-5, 0.38184, 0.21735, -0.1006, 0.09751, -0.33258, -0.11245, 0.15283, 38e-5, 0.14446, 0.32068, 0.314, -0.54543, 1.19563, 0.14428, 0.10296, 0.08279, -0.34031, -0.48157, 0.55689, 0.69371, -0.11803, 0.04465, -0.18975],
        [0.84505, -0.01593, 0.73613, -0.02808, -0.0323, -0.03731, 0.99839, -0.49015, 0.09092, 0.22662, 0.20962, -0.03395, 0.0363, 0.39627, 0.17294, -0.10954, 0.19316, -0.38144, -0.08657, 0.1616, -0.02291, 0.13777, 0.32836, 0.29449, -0.5402, 1.21886, 0.1556, 0.10652, 0.07083, -0.31541, -0.48303, 0.54201, 0.68161, -0.11608, 0.04389, -0.19781],
        [0.88211, -0.01178, 0.73248, -0.01166, -0.03046, -0.03756, 0.99876, -0.50443, 0.05136, 0.23599, 0.31887, 408e-5, 0.08465, 0.3976, 0.1159, -0.12029, 0.30844, -0.42005, -0.06078, 0.16637, -0.05212, 0.13298, 0.32316, 0.28096, -0.53414, 1.23813, 0.16207, 0.11127, 0.06188, -0.28059, -0.47242, 0.52834, 0.68794, -0.11362, 0.04448, -0.20427],
        [0.91425, -774e-5, 0.73163, 322e-5, -0.02958, -0.03872, 0.99881, -0.50894, 0.01947, 0.242, 0.40749, 0.02175, 0.11459, 0.38316, 0.06234, -0.12301, 0.43038, -0.41563, -0.04019, 0.16771, -0.07855, 0.13305, 0.30616, 0.27947, -0.53393, 1.2496, 0.15857, 0.11433, 0.06309, -0.2533, -0.45117, 0.52953, 0.71527, -0.11103, 0.04238, -0.21201],
        [0.95018, -264e-5, 0.73412, 0.01706, -0.03182, -0.03955, 0.99857, -0.48386, -0.01572, 0.2471, 0.46673, 435e-5, 0.13412, 0.33475, 121e-5, -0.12153, 0.61826, -0.31692, -0.02071, 0.16397, -0.10082, 0.13891, 0.27603, 0.28226, -0.53658, 1.25528, 0.14937, 0.11762, 0.0697, -0.22599, -0.42201, 0.54592, 0.76651, -0.11101, 0.03873, -0.22185],
        [0.99379, 436e-5, 0.73989, 0.03158, -0.03678, -0.041, 0.99798, -0.42089, -0.06422, 0.25594, 0.48594, -0.04003, 0.15596, 0.2356, -0.08065, -0.1181, 0.88063, -0.14234, -709e-5, 0.15647, -0.11886, 0.15084, 0.22786, 0.28326, -0.54089, 1.25148, 0.14217, 0.1213, 0.07857, -0.19588, -0.39132, 0.57391, 0.84185, -0.1165, 0.03672, -0.22724],
        [1.04046, 0.0127, 0.74761, 0.04522, -0.04128, -0.04359, 0.99717, -0.33628, -0.11876, 0.26463, 0.46608, -0.09062, 0.17657, 0.09978, -0.16697, -0.11429, 1.12322, -0.04737, -0.01505, 0.14469, -0.12879, 0.16738, 0.15875, 0.28391, -0.54985, 1.2332, 0.14376, 0.1198, 0.0952, -0.16606, -0.366, 0.60849, 0.93405, -0.13101, 0.0381, -0.22189],
        [1.08451, 0.02059, 0.75573, 0.05201, -0.04486, -0.04439, 0.99665, -0.24038, -0.16234, 0.26216, 0.41309, -0.12654, 0.18943, -0.03419, -0.22153, -0.11611, 1.26952, -0.07545, -0.04005, 0.12404, -0.12072, 0.18566, 0.0808, 0.28513, -0.5625, 1.20275, 0.15555, 0.11115, 0.12288, -0.13157, -0.34719, 0.63962, 1.02684, -0.1512, 0.04288, -0.20853],
        [1.12202, 0.02589, 0.76338, 0.04903, -0.04817, -0.04143, 0.99677, -0.13525, -0.18606, 0.24578, 0.33032, -0.13972, 0.19129, -0.13919, -0.22927, -0.12652, 1.29876, -0.12748, -0.06071, 0.09685, -0.09199, 0.19922, 0.01124, 0.28786, -0.57373, 1.16331, 0.17062, 0.09979, 0.15404, -0.08631, -0.33503, 0.65605, 1.09745, -0.16573, 0.05306, -0.19337],
        [1.15747, 0.02872, 0.77005, 0.04125, -0.05032, -0.03441, 0.99729, -0.02483, -0.19614, 0.22103, 0.23309, -0.13875, 0.18526, -0.22055, -0.2093, -0.14647, 1.24049, -0.16085, -0.07147, 0.06482, -0.0544, 0.20347, -0.05021, 0.2964, -0.57839, 1.11282, 0.17973, 0.0885, 0.17952, -0.02716, -0.32756, 0.65566, 1.14249, -0.16865, 0.06578, -0.1764],
        [1.19331, 0.02998, 0.77435, 0.03411, -0.04995, -0.02348, 0.99789, 0.07741, -0.20141, 0.19384, 0.13998, -0.13528, 0.17434, -0.28775, -0.18158, -0.17458, 1.12234, -0.17876, -0.07479, 0.02798, -0.02025, 0.20078, -0.10614, 0.31472, -0.57769, 1.05029, 0.17888, 0.07945, 0.19254, 0.04041, -0.32609, 0.64103, 1.16575, -0.1628, 0.07765, -0.1559]
      ]
    },
    run: {
      travel: [2.16991, 0],
      frames: [
        [0, 0.01348, 0.77144, -0.01058, -0.02671, -0.07617, 0.99668, -0.42547, -992e-5, 0.15593, 0.42881, -0.15561, 0.04565, 0.18582, -0.06393, -0.08674, 1.38667, -0.04568, -0.04957, 0.24298, -0.069, 0.23877, 0.82253, 0.58047, -0.17273, 0.2512, -0.03289, -0.09561, 0.07306, -0.18233, -0.52053, 0.46999, -0.40859, -0.17157, -0.16181, -0.02385],
        [0.10909, 0.01978, 0.75531, 0.01463, -0.03293, -0.05583, 0.99779, -0.35308, -0.1013, 0.18266, 0.64623, -0.19525, 0.10602, -0.05699, -0.15621, -0.08122, 1.7363, -0.15037, -0.07376, 0.15529, -0.10233, 0.28151, 0.57659, 0.5868, -0.23286, 0.18262, -0.03893, -0.1133, 0.03805, 0.03227, -0.50068, 0.39385, -0.16202, -0.16828, -0.11778, -0.02008],
        [0.2178, 0.02153, 0.74263, 0.03171, -0.03774, -0.02307, 0.99852, -0.25007, -0.17583, 0.17701, 0.77882, -0.32966, 0.13946, -0.34392, -0.17763, -0.06298, 1.92595, -0.20661, -0.06888, 0.03641, -0.11342, 0.31786, 0.27789, 0.59858, -0.27854, 0.07801, -76e-4, -0.13321, -647e-5, 0.25551, -0.5025, 0.31345, 0.06843, -0.14835, -0.07944, -0.0487],
        [0.33237, 0.0186, 0.7384, 0.02591, -0.03536, 0.01433, 0.99894, -0.08695, -0.1833, 0.1225, 0.7641, -0.45521, 0.12556, -0.60783, -0.14064, -0.0514, 1.9113, -0.2384, -0.05887, -0.08298, -0.07223, 0.32305, -0.03453, 0.58133, -0.30114, -0.06613, 0.04281, -0.14802, -0.03346, 0.48677, -0.52355, 0.24409, 0.21823, -0.12294, -0.06817, -0.08457],
        [0.45641, 0.01381, 0.74481, 259e-5, -0.0241, 0.04591, 0.99865, 0.12097, -0.12758, 0.05802, 0.59311, -0.46768, 0.09477, -0.78056, -0.09553, -0.07203, 1.70271, -0.27045, -0.06916, -0.18112, 487e-5, 0.29067, -0.30759, 0.54848, -0.32393, -0.25497, 0.07202, -0.15801, -0.03023, 0.72136, -0.56191, 0.17872, 0.26041, -0.10232, -0.07655, -0.1077],
        [0.58766, 961e-5, 0.7595, -0.01849, -976e-5, 0.06521, 0.99765, 0.3184, -0.06076, 0.0225, 0.3912, -0.27711, 0.05709, -0.82254, -0.05262, -0.10675, 1.31696, -0.28428, -0.07844, -0.24427, 0.07853, 0.2325, -0.48712, 0.54211, -0.36968, -0.46529, 0.06404, -0.15609, -95e-4, 0.92563, -0.60824, 0.10894, 0.24132, -0.07776, -0.09238, -0.1079],
        [0.72043, 588e-5, 0.77565, -0.01803, -31e-4, 0.07409, 0.99708, 0.43816, -0.0355, 0.02432, 0.36567, 0.01858, 0.01103, -0.74664, -0.02049, -0.12462, 0.83883, -0.27156, -0.05541, -0.27907, 0.10531, 0.18611, -0.5511, 0.55187, -0.42811, -0.63256, 0.05436, -0.15289, 0.02237, 1.05284, -0.62667, 0.05234, 0.21431, -0.0445, -0.10209, -0.09968],
        [0.84924, 121e-5, 0.78499, -6e-5, -843e-5, 0.07823, 0.9969, 0.45146, -0.04107, 0.05053, 0.57045, 0.16003, -0.01371, -0.61869, -0.01682, -0.12599, 0.46149, -0.2568, -0.02634, -0.30435, 0.08348, 0.1721, -0.49438, 0.54922, -0.47771, -0.67984, 0.07337, -0.17009, 0.04978, 1.07723, -0.61363, 0.0366, 0.18372, -785e-5, -0.10507, -0.10299],
        [0.97086, -542e-5, 0.78246, 0.01464, -0.01877, 0.07761, 0.9967, 0.36543, -0.02478, 0.07753, 0.93905, 0.10385, 264e-5, -0.5069, -0.01804, -0.13051, 0.32555, -0.22662, -0.021, -0.29982, 0.05096, 0.18821, -0.33536, 0.52703, -0.49044, -0.59313, 0.10915, -0.18454, 0.04527, 1.00127, -0.58572, 0.07363, 0.15742, 0.03012, -0.10662, -0.10957],
        [1.08495, -0.01348, 0.77144, 0.01058, -0.02671, 0.07617, 0.99668, 0.18582, 0.06393, 0.08674, 1.38667, -0.04568, 0.04957, -0.42547, 992e-5, -0.15593, 0.42881, -0.15561, -0.04565, -0.24298, 0.069, 0.23877, -0.18233, 0.52053, -0.46999, -0.40859, 0.17157, -0.16181, 0.02385, 0.82253, -0.58047, 0.17273, 0.2512, 0.03289, -0.09561, -0.07306],
        [1.19404, -0.01978, 0.75531, -0.01463, -0.03293, 0.05583, 0.99779, -0.05699, 0.15621, 0.08122, 1.7363, -0.15037, 0.07376, -0.35308, 0.1013, -0.18266, 0.64623, -0.19525, -0.10602, -0.15529, 0.10233, 0.28151, 0.03227, 0.50068, -0.39385, -0.16202, 0.16828, -0.11778, 0.02008, 0.57659, -0.5868, 0.23286, 0.18262, 0.03893, -0.1133, -0.03805],
        [1.30275, -0.02153, 0.74263, -0.03171, -0.03774, 0.02307, 0.99852, -0.34392, 0.17763, 0.06298, 1.92595, -0.20661, 0.06888, -0.25007, 0.17583, -0.17701, 0.77882, -0.32966, -0.13946, -0.03641, 0.11342, 0.31786, 0.25551, 0.5025, -0.31345, 0.06843, 0.14835, -0.07944, 0.0487, 0.27789, -0.59858, 0.27854, 0.07801, 76e-4, -0.13321, 647e-5],
        [1.41733, -0.0186, 0.7384, -0.02591, -0.03536, -0.01433, 0.99894, -0.60783, 0.14064, 0.0514, 1.9113, -0.2384, 0.05887, -0.08695, 0.1833, -0.1225, 0.7641, -0.45521, -0.12556, 0.08298, 0.07223, 0.32305, 0.48677, 0.52355, -0.24409, 0.21823, 0.12294, -0.06817, 0.08457, -0.03453, -0.58133, 0.30114, -0.06613, -0.04281, -0.14802, 0.03346],
        [1.54136, -0.01381, 0.74481, -259e-5, -0.0241, -0.04591, 0.99865, -0.78056, 0.09553, 0.07203, 1.70271, -0.27045, 0.06916, 0.12097, 0.12758, -0.05802, 0.59311, -0.46768, -0.09477, 0.18112, -487e-5, 0.29067, 0.72136, 0.56191, -0.17872, 0.26041, 0.10232, -0.07655, 0.1077, -0.30759, -0.54848, 0.32393, -0.25497, -0.07202, -0.15801, 0.03023],
        [1.67262, -961e-5, 0.7595, 0.01849, -976e-5, -0.06521, 0.99765, -0.82254, 0.05262, 0.10675, 1.31696, -0.28428, 0.07844, 0.3184, 0.06076, -0.0225, 0.3912, -0.27711, -0.05709, 0.24427, -0.07853, 0.2325, 0.92563, 0.60824, -0.10894, 0.24132, 0.07776, -0.09238, 0.1079, -0.48712, -0.54211, 0.36968, -0.46529, -0.06404, -0.15609, 95e-4],
        [1.80538, -588e-5, 0.77565, 0.01803, -31e-4, -0.07409, 0.99708, -0.74664, 0.02049, 0.12462, 0.83883, -0.27156, 0.05541, 0.43816, 0.0355, -0.02432, 0.36567, 0.01858, -0.01103, 0.27907, -0.10531, 0.18611, 1.05284, 0.62667, -0.05234, 0.21431, 0.0445, -0.10209, 0.09968, -0.5511, -0.55187, 0.42811, -0.63256, -0.05436, -0.15289, -0.02237],
        [1.9342, -121e-5, 0.78499, 6e-5, -843e-5, -0.07823, 0.9969, -0.61869, 0.01682, 0.12599, 0.46149, -0.2568, 0.02634, 0.45146, 0.04107, -0.05053, 0.57045, 0.16003, 0.01371, 0.30435, -0.08348, 0.1721, 1.07723, 0.61363, -0.0366, 0.18372, 785e-5, -0.10507, 0.10299, -0.49438, -0.54922, 0.47771, -0.67984, -0.07337, -0.17009, -0.04978],
        [2.05581, 542e-5, 0.78246, -0.01464, -0.01877, -0.07761, 0.9967, -0.5069, 0.01804, 0.13051, 0.32555, -0.22662, 0.021, 0.36543, 0.02478, -0.07753, 0.93905, 0.10385, -264e-5, 0.29982, -0.05096, 0.18821, 1.00127, 0.58572, -0.07363, 0.15742, -0.03012, -0.10662, 0.10957, -0.33536, -0.52703, 0.49044, -0.59313, -0.10915, -0.18454, -0.04527]
      ]
    }
  }
};
const RAW = {
  rest: [
    [0, 0.9925, 0, 0.7105, 0.7036, -95e-4, 96e-4, -1e-3, 9e-3, -35e-4, 1, -28e-4, 6e-4, -0.0941, 0.9956, -0.0165, 0.0174, 0.1133, 0.9933, 0.0317, -0.0197, 0.1559, 0.9871, -51e-4, -95e-4, -0.0219, 0.9997, -56e-4, 93e-4, -0.2482, 0.9687, -29e-4, 1e-4, -6e-4, 1, -27e-4, 6e-4, 9e-4, 1, -27e-4, -14e-4, 9e-4, 1, -0.1649, 0.7474, 0.016, -0.6434, 0.3121, -0.3664, 0.3578, 0.8002, 35e-4, 0.0156, -0.2845, -0.9586, 0.1533, -0.0872, 0.0481, 0.9831, 0, 0, 0, 1, 0, 0, 0, 1, -0.753, -0.1389, -0.6432, 22e-4, 0.234, -0.4619, 0.2285, 0.8244, -54e-4, 63e-4, 0.2458, 0.9693, 0.3533, -0.1323, 0.0915, 0.9216, 0, 0, 0, 1, 0, 0, 0, 1, 0.0594, -0.055, 0.9956, 0.0466, 15e-4, -1e-3, -0.1391, -0.9903, 0.0209, -0.0367, 0.6334, -0.7727, 0.02, 0.0137, -0.0374, 0.999, -0.1143, -0.0149, -0.015, 0.9932, -4e-4, 4e-4, 0.0889, 0.996, -0.0612, 55e-4, -0.6361, 0.7691, 0.0126, 43e-4, -0.0431, 0.999]
  ],
  walk: [
    [0, 0.9773, 0, 0.7006, 0.7133, 0.0141, -0.0107, -47e-4, -0.0234, 0.0133, 0.9996, -0.0211, -0.011, -0.1034, 0.9944, -0.0535, 0.0195, 0.1673, 0.9843, 56e-4, 0.0367, 0.2434, 0.9692, 5e-4, -82e-4, -0.0255, 0.9996, -0.0155, -14e-4, -0.2947, 0.9555, -5e-4, 1e-3, -1e-3, 1, 2e-4, 34e-4, -74e-4, 1, 11e-4, -16e-4, -74e-4, 1, 0.154, -0.7074, 0.0731, 0.6859, -0.1531, 0.3685, -0.3086, -0.8635, -76e-4, -0.0117, -0.3118, -0.95, -0.0174, -0.0202, -0.0226, 0.9994, 0, 0, 0, 1, 0, 0, 0, 1, -0.76, -0.2, -0.6138, 0.0746, 0.156, -0.3527, 0.2469, 0.889, -78e-4, -0.0506, 0.3742, 0.9259, -0.1247, 0.1139, -0.0476, 0.9845, 0, 0, 0, 1, 0, 0, 0, 1, 0.0828, 0.0607, -0.9904, 0.0926, 49e-4, -39e-4, 0.0259, 0.9997, 0.0551, 0.0863, 0.667, -0.738, 0.0282, 0.0258, -0.0471, 0.9982, 0.0111, -0.0277, -0.2186, 0.9754, -32e-4, 9e-4, 0.333, 0.9429, -0.0789, -0.0238, -0.6258, 0.7756, 0.0132, 33e-4, -0.086, 0.9962],
    [0.0521, 0.9705, 17e-4, 0.7043, 0.7097, 26e-4, -0.02, -0.0115, -0.0259, 0.0119, 0.9995, -0.0217, -0.0104, -0.1027, 0.9944, -0.0589, 0.022, 0.1659, 0.9841, 6e-4, 0.0453, 0.2361, 0.9707, 14e-4, -0.0105, -0.0254, 0.9996, -0.0165, -1e-3, -0.2989, 0.9542, 19e-4, 6e-4, 0, 1, -38e-4, 26e-4, -6e-3, 1, -9e-4, -34e-4, -8e-3, 1, 0.1509, -0.7006, 0.0787, 0.6929, -0.1428, 0.3511, -0.3167, -0.8695, -0.0129, -87e-4, -0.328, -0.9446, -0.0232, -88e-4, -0.0238, 0.9994, 0, 0, 0, 1, 0, 0, 0, 1, -0.7663, -0.2057, -0.604, 0.075, 0.1641, -0.353, 0.2359, 0.8904, -15e-4, -0.0443, 0.3583, 0.9326, -0.1349, 0.1031, -0.051, 0.9842, 0, 0, 0, 1, 0, 0, 0, 1, 0.0795, 0.0542, -0.9883, 0.1183, 3e-3, -32e-4, 0.0256, 0.9997, 0.056, 0.0745, 0.6778, -0.7293, 0.0259, 0.021, -0.06, 0.9976, 39e-4, -0.0168, -0.2134, 0.9768, 39e-4, 37e-4, 0.1899, 0.9818, -0.087, -0.0136, -0.614, 0.7843, 0.012, -37e-4, -0.0835, 0.9964],
    [0.1058, 0.9634, 39e-4, 0.7085, 0.7052, -66e-4, -0.0269, -6e-3, -0.0257, 82e-4, 0.9996, -0.0357, -0.0116, -0.1045, 0.9938, -0.0558, 0.0215, 0.1649, 0.9845, -19e-4, 0.0518, 0.2326, 0.9712, 14e-4, -0.0109, -0.0264, 0.9996, -0.0213, -9e-4, -0.3004, 0.9536, 23e-4, 1e-3, -8e-4, 1, -33e-4, 38e-4, -56e-4, 1, -2e-4, -27e-4, -63e-4, 1, 0.1477, -0.697, 0.0823, 0.6968, -0.136, 0.3416, -0.3189, -0.8736, -89e-4, -41e-4, -0.344, -0.9389, -0.0131, -17e-4, -0.0256, 0.9996, 0, 0, 0, 1, 0, 0, 0, 1, -0.7699, -0.2107, -0.5976, 0.0756, 0.1727, -0.357, 0.2266, 0.8896, 23e-4, -0.036, 0.3464, 0.9374, -0.1527, 0.0928, -0.0525, 0.9825, 0, 0, 0, 1, 0, 0, 0, 1, 0.0735, 0.0536, -0.9873, 0.1301, -16e-4, -33e-4, 0.0527, 0.9986, 0.0569, 0.0602, 0.6816, -0.727, 0.0242, 0.0185, -0.0956, 0.995, 7e-3, -44e-4, -0.2101, 0.9777, 54e-4, 4e-4, 0.0957, 0.9954, -0.1032, 12e-4, -0.6177, 0.7796, 0.0137, 24e-4, -0.0906, 0.9958],
    [0.1604, 0.9594, 73e-4, 0.7114, 0.7022, -0.0126, -0.0277, -46e-4, -0.0215, 71e-4, 0.9997, -0.0433, -0.0111, -0.1061, 0.9933, -0.0526, 0.02, 0.165, 0.9847, 2e-4, 0.0518, 0.2333, 0.971, -58e-4, -0.0103, -0.0278, 0.9995, -0.0266, -18e-4, -0.301, 0.9533, 46e-4, 16e-4, -1e-4, 1, -51e-4, 61e-4, -54e-4, 1, -69e-4, -42e-4, -57e-4, 1, 0.1481, -0.6974, 0.0822, 0.6964, -0.1212, 0.3483, -0.3149, -0.8745, -0.0144, -71e-4, -0.3447, -0.9386, -0.0149, 44e-4, -0.0268, 0.9995, 0, 0, 0, 1, 0, 0, 0, 1, -0.7705, -0.2155, -0.5946, 0.0796, 0.1561, -0.3641, 0.2093, 0.894, 1e-3, -0.0525, 0.3363, 0.9403, -0.1299, 0.0916, -0.0531, 0.9859, 0, 0, 0, 1, 0, 0, 0, 1, 0.0616, 0.0573, -0.987, 0.1372, -15e-4, -4e-3, 0.0853, 0.9963, 0.0474, 0.0447, 0.6735, -0.7363, 0.0265, 0.0215, -0.1389, 0.9897, 3e-3, 68e-4, -0.2017, 0.9794, 6e-4, -26e-4, 0.0639, 0.998, -0.1046, 8e-3, -0.6229, 0.7753, 0.0137, 62e-4, -0.109, 0.9939]
  ],
  run: [
    [0, 0.9654, 0, 0.7192, 0.6947, 14e-4, 0.0103, -0.0165, -0.0157, 0.0525, 0.9984, -0.0625, -26e-4, -0.1022, 0.9928, -0.0622, 0.0147, 0.1252, 0.9901, 0.03, 0.0556, 0.1424, 0.9878, 9e-4, -0.0268, -0.0124, 0.9996, 0.0118, -0.012, -0.276, 0.961, -42e-4, -13e-4, 0.0124, 0.9999, -96e-4, 73e-4, -25e-4, 0.9999, -26e-4, -26e-4, 53e-4, 1, 0.1515, -0.6274, 0.0242, 0.7635, 0.0292, -0.4166, 0.3128, 0.8531, -0.0552, 9e-4, 0.9089, 0.4133, -0.2444, -0.0137, -0.0134, 0.9695, 0, 0, 0, 1, 0, 0, 0, 1, -0.7225, -0.2829, -0.6307, -87e-4, 0.2985, -0.4319, -0.058, 0.8491, -0.0188, -0.0437, 0.8596, 0.5087, -0.2297, -0.0418, -22e-4, 0.9724, 0, 0, 0, 1, 0, 0, 0, 1, -0.029, 0.0385, -0.9655, 0.2558, 0.0445, -0.0292, 0.6382, 0.768, -0.0319, 0.0137, -0.5578, 0.8293, 0.0137, 0.0164, -0.0655, 0.9976, -0.0653, 62e-4, -0.3053, 0.95, -0.0228, -71e-4, 0.1849, 0.9825, -0.0101, 0.1112, -0.6361, 0.7635, -0.0103, 24e-4, -0.1369, 0.9905],
    [0.1288, 0.9339, 0.0221, 0.711, 0.7015, -1e-4, 0.0498, 29e-4, 32e-4, 0.0666, 0.9978, -0.0579, 16e-4, -0.0957, 0.9937, -0.0537, 0.0244, 0.1369, 0.9888, 0.0306, 0.0504, 0.1301, 0.9898, 31e-4, -0.03, -76e-4, 0.9995, 0.0247, -0.0118, -0.3006, 0.9533, -17e-4, -17e-4, 85e-4, 1, -73e-4, 83e-4, -24e-4, 0.9999, -45e-4, -44e-4, 1e-3, 1, 0.1843, -0.6416, 0.0304, 0.744, -0.0751, 0.4346, -0.2901, -0.8493, 0.0485, -3e-3, -0.8907, -0.452, -0.2254, -95e-4, 69e-4, 0.9742, 0, 0, 0, 1, 0, 0, 0, 1, -0.7508, -0.2691, -0.6029, 0.0202, 0.2276, -0.4263, -8e-3, 0.8754, -0.0228, -0.0495, 0.8468, 0.5291, -0.2303, -0.0119, 0.0192, 0.9729, 0, 0, 0, 1, 0, 0, 0, 1, -0.0675, 0.0623, -0.9764, 0.1954, 0.0532, -0.0182, 0.7647, 0.6419, -0.0331, -3e-4, -0.5965, 0.802, 0.0179, 92e-4, -0.077, 0.9968, -0.0515, 0.0479, -0.3126, 0.9473, -0.029, -91e-4, 0.3313, 0.943, 0.0575, 0.0824, -0.6168, 0.7807, 73e-4, 0.0124, -0.1237, 0.9922],
    [0.2546, 0.9119, 0.0442, 0.7047, 0.704, -56e-4, 0.0886, 0.0135, 0.02, 0.0746, 0.9969, -0.0544, 69e-4, -0.0934, 0.9941, -0.0308, 0.039, 0.1568, 0.9864, 0.0332, 0.0361, 0.1228, 0.9912, -19e-4, -0.0223, -0.0145, 0.9996, 0.0278, -0.0124, -0.3304, 0.9434, -19e-4, -9e-4, 0.0135, 0.9999, -0.0138, 85e-4, -22e-4, 0.9999, -86e-4, -67e-4, 22e-4, 0.9999, 0.2159, -0.6549, 0.0388, 0.7232, -0.1121, 0.4638, -0.2685, -0.8368, 0.0401, -0.0195, -0.8542, -0.518, -0.1933, 35e-4, 0.0311, 0.9806, 0, 0, 0, 1, 0, 0, 0, 1, -0.7722, -0.254, -0.5807, 0.0438, 0.1325, -0.4132, 0.0711, 0.8981, -0.0168, -0.0514, 0.8214, 0.5677, -0.235, 0.0247, 0.0362, 0.971, 0, 0, 0, 1, 0, 0, 0, 1, -0.1108, 0.065, -0.9867, 0.0996, 0.0565, -0.01, 0.8548, 0.5158, -0.0109, -0.0112, -0.6247, 0.7807, 0.0319, -0.0157, -0.0827, 0.9959, -0.0408, 0.0958, -0.289, 0.9516, -0.0313, -0.017, 0.4416, 0.8965, 0.094, 0.0655, -0.681, 0.7233, 0.0107, 0.0266, -0.1016, 0.9944],
    [0.3815, 0.9079, 0.0657, 0.7008, 0.7047, -23e-4, 0.1103, 0.012, 0.0273, 0.0799, 0.9964, -0.0319, 0.0107, -0.0907, 0.9953, -0.0179, 0.0477, 0.1645, 0.9851, 0.0368, 0.0232, 0.1182, 0.992, -0.0119, -0.014, -0.0123, 0.9998, 0.0221, -0.0184, -0.3408, 0.9397, -43e-4, -1e-4, 0.0106, 0.9999, -0.0123, 38e-4, -19e-4, 0.9999, -6e-3, -5e-3, 7e-4, 1, 0.2377, -0.667, 0.0351, 0.7052, -0.1678, 0.4869, -0.2342, -0.8246, 0.0464, -0.0432, -0.8146, -0.5766, -0.1758, 9e-3, 0.0478, 0.9832, 0, 0, 0, 1, 0, 0, 0, 1, -0.7793, -0.2333, -0.5788, 0.0577, 0.0622, -0.4049, 0.1717, 0.8959, -0.0235, -0.0443, 0.7956, 0.6037, -0.2177, 0.0518, 0.0555, 0.973, 0, 0, 0, 1, 0, 0, 0, 1, 0.1263, -0.0473, 0.9905, 0.026, -0.0558, 32e-4, -0.8995, -0.4334, 53e-4, -43e-4, -0.6412, 0.7673, 0.0499, -0.0198, -0.0764, 0.9956, -0.0288, 0.1168, -0.2089, 0.9705, -0.0292, -0.0154, 0.4461, 0.8944, 0.1014, 0.0549, -0.7339, 0.6694, 0.0145, 0.0186, -0.0984, 0.9949]
  ],
  cycles: {
    walk: {
      travel: [1.4264, 0],
      frames: [
        [0, 0.9515, 0, 0.6965, 0.7136, 0.0559, 0.0504, 0.0123, 52e-4, 0.0197, 0.9997, 0.0338, -16e-4, -0.112, 0.9931, -0.0307, 84e-4, 0.1583, 0.9869, -19e-4, 46e-4, 0.2606, 0.9654, -36e-4, -0.0151, -0.0305, 0.9994, -0.0299, 0.0278, -0.2849, 0.9577, 88e-4, 19e-4, 8e-4, 1, -15e-4, 42e-4, -0.0168, 0.9999, 45e-4, 16e-4, -0.0188, 0.9998, -0.1726, 0.7618, -0.0156, -0.6242, -0.2272, 0.4833, -0.1983, -0.8219, 0.0123, 0.0306, 0.2572, 0.9658, -0.0612, -0.0917, -0.0431, 0.993, 0, 0, 0, 1, 0, 0, 0, 1, -0.7383, -0.1369, -0.655, 0.0846, 0.0362, -0.3475, 0.2782, 0.8947, 0.012, -0.0793, 0.5075, 0.8579, -0.1717, 0.1574, -0.1139, 0.9658, 0, 0, 0, 1, 0, 0, 0, 1, -0.0436, -0.1244, 0.9738, 0.1853, 0.0201, -2e-3, 0.0925, 0.9955, -0.0574, -0.0964, -0.5703, 0.8138, 0.0363, 0.0329, -0.0806, 0.9955, 54e-4, -0.0158, 0.1381, 0.9903, 65e-4, 0.0113, 0.2367, 0.9715, 56e-4, 0.0253, -0.6489, 0.7605, 91e-4, 31e-4, -0.187, 0.9823],
        [0.0561, 0.954, -34e-4, 0.6938, 0.7161, 0.0667, 0.037, 86e-4, -73e-4, 0.0207, 0.9997, 0.0341, -42e-4, -0.1111, 0.9932, -0.0295, 32e-4, 0.156, 0.9873, -14e-4, 0.0132, 0.262, 0.965, -52e-4, -0.0181, -0.0292, 0.9994, -0.03, 0.0284, -0.285, 0.9577, 81e-4, 14e-4, 9e-4, 1, -36e-4, 33e-4, -0.017, 0.9998, 24e-4, 12e-4, -0.0198, 0.9998, -0.169, 0.7571, -0.0179, -0.6308, -0.2268, 0.4795, -0.2024, -0.8232, 0.0118, 0.0254, 0.2534, 0.967, -0.0506, -0.0858, -0.0419, 0.9941, 0, 0, 0, 1, 0, 0, 0, 1, -0.7403, -0.1428, -0.6518, 0.0818, 0.042, -0.3558, 0.2718, 0.8932, 56e-4, -0.0761, 0.5053, 0.8596, -0.1732, 0.1595, -0.1079, 0.9659, 0, 0, 0, 1, 0, 0, 0, 1, -0.0657, -0.1182, 0.9758, 0.1721, 0.0235, -2e-3, 0.1311, 0.9911, -0.055, -0.114, -0.5716, 0.8107, 0.0355, 0.0357, -0.0674, 0.9964, 6e-4, -0.0473, 0.0989, 0.994, 74e-4, 0.0119, 0.3445, 0.9387, 0.0115, 0.0329, -0.5955, 0.8026, 81e-4, 64e-4, -0.2096, 0.9777],
        [0.1103, 0.9583, -67e-4, 0.69, 0.7199, 0.0707, 0.0256, 56e-4, -0.0136, 0.0231, 0.9996, 0.0284, -73e-4, -0.1085, 0.9937, -0.0263, -8e-4, 0.1548, 0.9876, -15e-4, 0.0169, 0.2604, 0.9654, -3e-3, -0.019, -0.0272, 0.9994, -0.0301, 0.0265, -0.2864, 0.9573, 75e-4, 9e-4, 13e-4, 1, -48e-4, 33e-4, -0.0154, 0.9999, 0, 8e-4, -0.0182, 0.9998, -0.1651, 0.7507, -0.0209, -0.6393, -0.2272, 0.4736, -0.2134, -0.8237, 0.0117, 0.022, 0.2507, 0.9677, -0.0447, -0.0803, -0.0399, 0.995, 0, 0, 0, 1, 0, 0, 0, 1, -0.7387, -0.1502, -0.6521, 0.0809, 0.0533, -0.365, 0.2689, 0.8898, -14e-4, -0.0716, 0.4991, 0.8636, -0.1679, 0.1631, -0.1023, 0.9668, 0, 0, 0, 1, 0, 0, 0, 1, -0.0805, -0.1096, 0.9794, 0.1491, 0.0229, -11e-4, 0.1563, 0.9875, -0.0552, -0.1209, -0.5931, 0.7941, 0.0353, 0.0355, -0.0566, 0.9971, 13e-4, -0.0683, 0.0469, 0.9966, 82e-4, 0.0105, 0.4512, 0.8923, 15e-4, 0.0189, -0.5367, 0.8435, 0.0106, 87e-4, -0.1665, 0.9859],
        [0.1622, 0.964, -9e-3, 0.6876, 0.7225, 0.0701, 0.0167, 26e-4, -0.0175, 0.0235, 0.9996, 0.0215, -9e-3, -0.1062, 0.9941, -0.0245, -27e-4, 0.1563, 0.9874, -9e-4, 0.0176, 0.2585, 0.9658, 18e-4, -0.0186, -0.0266, 0.9995, -0.031, 0.0225, -0.289, 0.9566, 64e-4, 5e-4, 5e-4, 1, -41e-4, 37e-4, -0.0137, 0.9999, -14e-4, 11e-4, -0.0162, 0.9999, -0.1631, 0.745, -0.0261, -0.6463, -0.224, 0.4644, -0.2278, -0.826, 0.0114, 0.0216, 0.2504, 0.9678, -0.0418, -0.0745, -0.0363, 0.9957, 0, 0, 0, 1, 0, 0, 0, 1, -0.7377, -0.1581, -0.6514, 0.0809, 0.0686, -0.3697, 0.2706, 0.8862, -6e-3, -0.0677, 0.4873, 0.8706, -0.1519, 0.1622, -0.0975, 0.9701, 0, 0, 0, 1, 0, 0, 0, 1, -0.0898, -0.0998, 0.9838, 0.1186, -0.0189, 3e-4, -0.1596, -0.987, 0.0507, 0.1213, 0.6124, -0.7795, 0.0343, 0.0329, -0.051, 0.9976, 67e-4, -0.0779, -77e-4, 0.9969, 66e-4, 89e-4, 0.5294, 0.8483, -0.017, -1e-3, -0.5173, 0.8557, 0.0114, 84e-4, -0.101, 0.9948],
        [0.2131, 0.971, -0.0103, 0.6866, 0.724, 0.065, 0.0107, -5e-4, -0.0194, 0.0224, 0.9996, 0.0158, -95e-4, -0.1051, 0.9943, -0.0255, -18e-4, 0.1594, 0.9869, 13e-4, 0.0166, 0.2578, 0.9661, 28e-4, -0.0171, -0.0262, 0.9995, -0.0289, 0.019, -0.2908, 0.9562, 31e-4, 5e-4, 2e-4, 1, -48e-4, 37e-4, -0.0133, 0.9999, -33e-4, 7e-4, -0.0151, 0.9999, -0.1633, 0.7413, -0.0321, -0.6502, -0.2153, 0.4526, -0.2428, -0.8305, 0.0116, 0.023, 0.2536, 0.967, -0.0393, -0.0679, -0.0318, 0.9964, 0, 0, 0, 1, 0, 0, 0, 1, -0.7388, -0.1662, -0.6482, 0.0798, 0.089, -0.3698, 0.2742, 0.8833, -97e-4, -0.0635, 0.47, 0.8803, -0.1366, 0.1558, -0.0953, 0.9737, 0, 0, 0, 1, 0, 0, 0, 1, -0.0943, -0.0907, 0.9882, 0.0798, -0.015, 14e-4, -0.1417, -0.9898, 0.0475, 0.1184, 0.6246, -0.7704, 0.0342, 0.0336, -0.0459, 0.9978, 0.0119, -0.077, -0.0596, 0.9952, 35e-4, 92e-4, 0.5705, 0.8213, -0.034, -0.0149, -0.5389, 0.8415, 0.0129, 87e-4, -0.0752, 0.997],
        [0.2437, 0.9752, -0.0109, 0.6852, 0.7258, 0.0604, 86e-4, -18e-4, -0.0198, 0.0231, 0.9995, 0.0127, -97e-4, -0.1048, 0.9944, -0.0289, 5e-4, 0.1612, 0.9865, 15e-4, 0.016, 0.2583, 0.9659, 24e-4, -0.0167, -0.026, 0.9995, -0.0266, 0.018, -0.2897, 0.9566, 19e-4, 7e-4, 6e-4, 1, -54e-4, 34e-4, -0.0138, 0.9999, -29e-4, 3e-4, -0.0148, 0.9999, -0.1607, 0.7403, -0.0345, -0.6519, -0.2061, 0.4482, -0.2487, -0.8335, 0.0114, 0.0223, 0.2591, 0.9655, -0.0405, -0.0641, -0.0305, 0.9967, 0, 0, 0, 1, 0, 0, 0, 1, -0.7401, -0.1711, -0.6456, 0.0786, 0.0967, -0.3726, 0.2709, 0.8823, -0.0103, -0.0643, 0.457, 0.8871, -0.1388, 0.1509, -0.0935, 0.9743, 0, 0, 0, 1, 0, 0, 0, 1, -0.0948, -0.0891, 0.9902, 0.0514, -0.0147, 22e-4, -0.1241, -0.9921, 0.0466, 0.1168, 0.6298, -0.7665, 0.034, 0.0333, -0.0414, 0.998, 0.0113, -0.0739, -0.0919, 0.993, -3e-4, 0.0109, 0.5816, 0.8134, -0.0405, -0.0209, -0.5602, 0.8271, 0.0154, 97e-4, -0.0744, 0.9971],
        [0.2671, 0.9771, -0.0114, 0.6838, 0.7275, 0.0555, 69e-4, -24e-4, -0.0199, 0.0253, 0.9995, 92e-4, -99e-4, -0.1049, 0.9944, -0.0328, 32e-4, 0.1613, 0.9864, -4e-4, 0.0176, 0.26, 0.9655, 21e-4, -0.0166, -0.0271, 0.9995, -0.0256, 0.0175, -0.2864, 0.9576, 34e-4, 7e-4, 6e-4, 1, -53e-4, 35e-4, -0.0149, 0.9999, -1e-3, 6e-4, -0.0159, 0.9999, -0.1536, 0.7388, -0.0349, -0.6552, -0.1958, 0.4503, -0.2482, -0.835, 0.0113, 0.0202, 0.2711, 0.9623, -0.0485, -0.0618, -0.0314, 0.9964, 0, 0, 0, 1, 0, 0, 0, 1, -0.7416, -0.1751, -0.6428, 0.0779, 0.0961, -0.3787, 0.2598, 0.8831, -0.0102, -0.0701, 0.4459, 0.8923, -0.1584, 0.1471, -0.092, 0.972, 0, 0, 0, 1, 0, 0, 0, 1, -0.0931, -0.0917, 0.991, 0.0273, -0.0169, 14e-4, -0.1074, -0.9941, 0.0458, 0.1159, 0.6325, -0.7644, 0.0331, 0.0314, -0.0383, 0.9982, 36e-4, -0.0703, -0.118, 0.9905, -64e-4, 0.0125, 0.5713, 0.8206, -0.0391, -0.0241, -0.5738, 0.8177, 0.0149, 99e-4, -0.0742, 0.9971],
        [0.3085, 0.9786, -0.0117, 0.684, 0.728, 0.0459, 2e-3, -4e-3, -0.0204, 0.0272, 0.9994, 52e-4, -0.0103, -0.1052, 0.9944, -0.0397, 69e-4, 0.1607, 0.9862, -26e-4, 0.0223, 0.261, 0.9651, 12e-4, -0.0167, -0.0286, 0.9994, -0.0254, 0.0171, -0.2836, 0.9585, 46e-4, 1e-3, 8e-4, 1, -54e-4, 38e-4, -0.0169, 0.9998, 1e-3, 9e-4, -0.0179, 0.9998, -0.1432, 0.735, -0.035, -0.6618, -0.1798, 0.4507, -0.2469, -0.8388, 0.0102, 0.0167, 0.2945, 0.9555, -0.0573, -0.0579, -0.0284, 0.9963, 0, 0, 0, 1, 0, 0, 0, 1, -0.7453, -0.1811, -0.6371, 0.0765, 0.1017, -0.3845, 0.2419, 0.8851, -0.0105, -0.0774, 0.4294, 0.8997, -0.1794, 0.1387, -0.0913, 0.9697, 0, 0, 0, 1, 0, 0, 0, 1, -0.0904, -0.0911, 0.9917, -98e-4, -0.0198, 12e-4, -0.0809, -0.9965, 0.0438, 0.1126, 0.6379, -0.7606, 0.0321, 0.0291, -0.0359, 0.9984, -0.0117, -0.0626, -0.1551, 0.9859, -0.0125, 0.0135, 0.5317, 0.8467, -0.0399, -0.0279, -0.5884, 0.8071, 0.0127, 86e-4, -0.0772, 0.9969],
        [0.3661, 0.9777, -0.0115, 0.6863, 0.7265, 0.034, -73e-4, -48e-4, -0.0223, 0.0281, 0.9993, -2e-4, -0.0112, -0.1053, 0.9944, -0.0485, 0.0102, 0.1596, 0.9859, -32e-4, 0.0312, 0.259, 0.9654, -18e-4, -0.0183, -0.0298, 0.9994, -0.0256, 0.0169, -0.2823, 0.9588, 64e-4, 16e-4, 13e-4, 1, -54e-4, 42e-4, -0.0183, 0.9998, 33e-4, 15e-4, -0.0196, 0.9998, -0.1316, 0.73, -0.0348, -0.6697, -0.1567, 0.4471, -0.2446, -0.846, 0.01, 96e-4, 0.3286, 0.9444, -0.0674, -0.0518, -0.0216, 0.9961, 0, 0, 0, 1, 0, 0, 0, 1, -0.7516, -0.1876, -0.6281, 0.0741, 0.1125, -0.3891, 0.2171, 0.8882, -0.0111, -0.0837, 0.4101, 0.9081, -0.196, 0.1258, -0.0924, 0.9681, 0, 0, 0, 1, 0, 0, 0, 1, -0.0885, -0.0876, 0.9907, -0.0546, -0.024, 19e-4, -0.0523, -0.9983, 0.0422, 0.1074, 0.648, -0.7529, 0.032, 0.0283, -0.033, 0.9985, -0.0329, -0.0533, -0.1919, 0.9794, -0.0148, 0.0156, 0.4472, 0.8942, -0.0441, -0.0291, -0.6014, 0.7972, 93e-4, 65e-4, -0.0823, 0.9965],
        [0.4338, 0.973, -0.0107, 0.6906, 0.7226, 0.0212, -0.0199, -35e-4, -0.0252, 0.0278, 0.9993, -66e-4, -0.0127, -0.106, 0.9943, -0.0587, 0.0128, 0.1582, 0.9856, -3e-3, 0.0416, 0.2531, 0.9665, -36e-4, -0.0195, -0.031, 0.9993, -0.0259, 0.015, -0.2812, 0.9592, 7e-3, 23e-4, 12e-4, 1, -37e-4, 5e-3, -0.0194, 0.9998, 48e-4, 21e-4, -0.0216, 0.9998, -0.1218, 0.7258, -0.0355, -0.6761, -0.1339, 0.4363, -0.2453, -0.8553, 0.0121, 44e-4, 0.3679, 0.9298, -0.0815, -0.0441, -0.0153, 0.9956, 0, 0, 0, 1, 0, 0, 0, 1, -0.759, -0.1944, -0.6172, 0.0729, 0.1256, -0.3913, 0.1892, 0.8918, -0.0123, -0.0861, 0.3916, 0.916, -0.2121, 0.1103, -0.095, 0.9663, 0, 0, 0, 1, 0, 0, 0, 1, -0.0869, -0.0811, 0.9879, -0.0993, -0.0277, 28e-4, -0.0298, -0.9992, 0.0416, 0.1001, 0.6631, -0.7406, 0.0313, 0.0273, -0.031, 0.9987, -0.0558, -0.0436, -0.2181, 0.9734, -0.0119, 0.0196, 0.3146, 0.949, -0.0474, -0.0217, -0.6111, 0.7898, 53e-4, 24e-4, -0.0804, 0.9968],
        [0.501, 0.9653, -91e-4, 0.6953, 0.7179, 87e-4, -0.0333, -15e-4, -0.0282, 0.026, 0.9993, -0.0118, -0.0138, -0.1076, 0.994, -0.0681, 0.0148, 0.1582, 0.9849, -46e-4, 0.0499, 0.2465, 0.9678, -21e-4, -0.0193, -0.0315, 0.9993, -0.0275, 0.0127, -0.2829, 0.9587, 73e-4, 3e-3, 14e-4, 1, -29e-4, 53e-4, -0.0198, 0.9998, 45e-4, 19e-4, -0.0229, 0.9997, -0.1147, 0.7233, -0.0375, -0.6799, -0.1157, 0.4229, -0.2484, -0.8638, 0.0158, 28e-4, 0.4023, 0.9154, -0.0974, -0.0366, -0.0131, 0.9945, 0, 0, 0, 1, 0, 0, 0, 1, -0.7653, -0.2002, -0.6073, 0.0734, 0.1373, -0.3928, 0.1631, 0.8946, -0.0129, -0.0842, 0.3774, 0.9221, -0.2224, 0.0974, -0.0981, 0.9651, 0, 0, 0, 1, 0, 0, 0, 1, -0.0834, -0.0727, 0.9843, -0.1378, -0.029, 34e-4, -0.0173, -0.9994, 0.0411, 0.0896, 0.6782, -0.7283, 0.0306, 0.0258, -0.034, 0.9986, -0.0742, -0.0341, -0.2301, 0.9697, -53e-4, 0.0223, 0.1736, 0.9846, -0.0527, -43e-4, -0.6178, 0.7845, 17e-4, -1e-4, -0.0774, 0.997],
        [0.5554, 0.9575, -67e-4, 0.6986, 0.7143, -16e-4, -0.0423, -3e-4, -0.0283, 0.0234, 0.9993, -0.0169, -0.0144, -0.1098, 0.9937, -0.0712, 0.0151, 0.1597, 0.9845, -64e-4, 0.0536, 0.2421, 0.9687, -5e-4, -0.0174, -0.0315, 0.9993, -0.0272, 0.0108, -0.2851, 0.958, 8e-3, 35e-4, 2e-3, 1, -59e-4, 53e-4, -0.0188, 0.9998, 59e-4, 15e-4, -0.0214, 0.9998, -0.1125, 0.7221, -0.0407, -0.6813, -0.104, 0.4139, -0.252, -0.8686, 0.0174, 47e-4, 0.4222, 0.9063, -0.112, -0.0291, -0.0161, 0.9932, 0, 0, 0, 1, 0, 0, 0, 1, -0.7696, -0.203, -0.6009, 0.0739, 0.1468, -0.394, 0.1463, 0.8954, -0.0114, -0.0795, 0.3698, 0.9256, -0.2276, 0.0914, -0.1019, 0.9641, 0, 0, 0, 1, 0, 0, 0, 1, 0.0755, 0.0668, -0.9817, 0.1617, 0.0265, -35e-4, 0.0235, 0.9994, 0.0403, 0.0759, 0.6862, -0.7223, 0.0294, 0.0231, -0.0502, 0.998, -0.0819, -0.0255, -0.231, 0.9692, -13e-4, 0.0216, 0.0979, 0.995, -0.0617, 96e-4, -0.6242, 0.7788, 18e-4, 22e-4, -0.0815, 0.9967],
        [0.6081, 0.9518, -39e-4, 0.7016, 0.7111, -0.0135, -0.0442, -6e-4, -0.0221, 0.0192, 0.9996, -0.0212, -0.0145, -0.1123, 0.9933, -0.0709, 0.0157, 0.1622, 0.9841, -69e-4, 0.0538, 0.2391, 0.9695, 8e-4, -0.0164, -0.0312, 0.9994, -0.0258, 91e-4, -0.284, 0.9585, 62e-4, 34e-4, 31e-4, 1, -62e-4, 53e-4, -0.0173, 0.9998, 65e-4, 1e-3, -0.0185, 0.9998, -0.1141, 0.7234, -0.0442, -0.6794, -0.0955, 0.4083, -0.2553, -0.8712, 0.0148, 95e-4, 0.4328, 0.9013, -0.126, -0.0235, -0.019, 0.9916, 0, 0, 0, 1, 0, 0, 0, 1, -0.7736, -0.2029, -0.5958, 0.0742, 0.1538, -0.3936, 0.1334, 0.8964, -84e-4, -0.0768, 0.3625, 0.9288, -0.2351, 0.0896, -0.1059, 0.962, 0, 0, 0, 1, 0, 0, 0, 1, 0.0585, 0.0669, -0.981, 0.1725, 0.0224, -35e-4, 0.0532, 0.9983, 0.0374, 0.0596, 0.6863, -0.7239, 0.0288, 0.021, -0.0862, 0.9956, -0.0832, -0.0131, -0.2316, 0.9691, -34e-4, 0.0206, 0.0923, 0.9955, -0.0624, 95e-4, -0.6216, 0.7808, 4e-3, 75e-4, -0.0899, 0.9959],
        [0.6623, 0.9493, -6e-4, 0.7033, 0.7095, -0.0273, -0.0368, -9e-4, -99e-4, 0.0156, 0.9998, -0.0234, -0.0129, -0.1144, 0.9931, -0.0693, 0.0189, 0.1641, 0.9838, -61e-4, 0.0487, 0.2372, 0.9702, 17e-4, -0.0158, -0.0304, 0.9994, -0.0246, 77e-4, -0.2824, 0.9589, 54e-4, 42e-4, 26e-4, 1, -47e-4, 53e-4, -0.0173, 0.9998, 69e-4, 12e-4, -0.0183, 0.9998, -0.1181, 0.7284, -0.0473, -0.6733, -0.0836, 0.4083, -0.2532, -0.873, 0.0133, 0.0115, 0.4355, 0.9, -0.1336, -0.0215, -0.0197, 0.9906, 0, 0, 0, 1, 0, 0, 0, 1, -0.7774, -0.1993, -0.5919, 0.0749, 0.1568, -0.391, 0.1229, 0.8986, -69e-4, -0.0762, 0.3538, 0.9322, -0.2412, 0.0883, -0.1099, 0.9602, 0, 0, 0, 1, 0, 0, 0, 1, 0.0293, 0.0743, -0.9827, 0.1668, 0.0175, -12e-4, 0.1154, 0.9932, 0.0306, 0.0392, 0.6741, -0.737, 0.0278, 0.0172, -0.1393, 0.9897, -0.0807, 6e-3, -0.2325, 0.9692, -72e-4, 0.0185, 0.1289, 0.9915, -0.05, -24e-4, -0.6016, 0.7973, 62e-4, 0.0115, -0.0917, 0.9957],
        [0.7174, 0.9499, 32e-4, 0.7017, 0.711, -0.0382, -0.023, 32e-4, 28e-4, 0.0149, 0.9999, -0.0242, -0.0105, -0.1146, 0.9931, -0.0694, 0.0251, 0.1647, 0.9836, -35e-4, 0.0403, 0.2377, 0.9705, 12e-4, -0.0151, -0.0298, 0.9994, -0.0235, 71e-4, -0.2826, 0.9589, 63e-4, 42e-4, 2e-3, 1, -42e-4, 45e-4, -0.0184, 0.9998, 67e-4, 13e-4, -0.0193, 0.9998, -0.1232, 0.7325, -0.0481, -0.6679, -0.0804, 0.4161, -0.2494, -0.8707, 62e-4, 0.0153, 0.4309, 0.9022, -0.1391, -0.0223, -0.022, 0.9898, 0, 0, 0, 1, 0, 0, 0, 1, -0.7774, -0.1949, -0.593, 0.0771, 0.1576, -0.3896, 0.1216, 0.8992, -52e-4, -0.0768, 0.3434, 0.936, -0.2429, 0.0869, -0.1133, 0.9595, 0, 0, 0, 1, 0, 0, 0, 1, -62e-4, 0.0871, -0.9859, 0.1429, 0.0153, 37e-4, 0.2137, 0.9768, 0.0221, 0.0169, 0.6311, -0.7752, 0.0308, 0.0214, -0.1978, 0.9795, -0.0742, 0.0287, -0.2283, 0.9703, -91e-4, 0.0145, 0.177, 0.9841, -0.0338, -0.0225, -0.5849, 0.8101, 95e-4, 0.0132, -0.0763, 0.997],
        [0.7717, 0.9527, 72e-4, 0.6968, 0.7159, -0.044, -78e-4, 93e-4, 0.0118, 0.0172, 0.9997, -0.0205, -77e-4, -0.1122, 0.9935, -0.0725, 0.0321, 0.1643, 0.9832, -11e-4, 0.0328, 0.2391, 0.9705, 13e-4, -0.0158, -0.029, 0.9994, -0.0223, 81e-4, -0.2831, 0.9588, 66e-4, 33e-4, 15e-4, 1, -32e-4, 33e-4, -0.018, 0.9998, 65e-4, 1e-3, -0.0184, 0.9998, -0.1278, 0.7325, -0.046, -0.6671, -0.0864, 0.4324, -0.2443, -0.8636, -16e-4, 0.0197, 0.4212, 0.9067, -0.1433, -0.0268, -0.0292, 0.9889, 0, 0, 0, 1, 0, 0, 0, 1, -0.7725, -0.1909, -0.6005, 0.079, 0.1591, -0.3911, 0.1328, 0.8967, -38e-4, -0.0784, 0.3338, 0.9394, -0.239, 0.0882, -0.1158, 0.9601, 0, 0, 0, 1, 0, 0, 0, 1, -0.0382, 0.0989, -0.9891, 0.1025, 0.0176, 76e-4, 0.3323, 0.943, 0.0184, 24e-4, 0.5544, -0.832, 0.034, 0.0289, -0.1999, 0.9788, -0.0644, 0.0486, -0.2113, 0.9741, -95e-4, 0.0125, 0.2167, 0.9761, -0.0244, -0.0407, -0.5931, 0.8037, 0.011, 0.0148, -0.0594, 0.9981],
        [0.8239, 0.9568, 0.011, 0.6907, 0.7216, -0.0476, 43e-4, 0.0137, 0.0184, 0.0209, 0.9995, -0.0168, -49e-4, -0.1093, 0.9939, -0.0737, 0.036, 0.1642, 0.983, -1e-3, 0.0294, 0.2411, 0.9701, 28e-4, -0.0182, -0.0284, 0.9994, -0.0212, 0.011, -0.2861, 0.9579, 72e-4, 31e-4, 11e-4, 1, -11e-4, 35e-4, -0.017, 0.9999, 93e-4, 16e-4, -0.0171, 0.9998, -0.1307, 0.7319, -0.042, -0.6674, -0.0925, 0.4496, -0.2361, -0.8565, -28e-4, 0.0177, 0.4088, 0.9124, -0.1404, -0.034, -0.04, 0.9887, 0, 0, 0, 1, 0, 0, 0, 1, -0.7677, -0.1868, -0.6079, 0.0795, 0.1612, -0.3896, 0.1519, 0.894, -43e-4, -0.0804, 0.3302, 0.9405, -0.2361, 0.0909, -0.1184, 0.9602, 0, 0, 0, 1, 0, 0, 0, 1, -0.0614, 0.1034, -0.9914, 0.0525, 0.0204, 84e-4, 0.4399, 0.8978, 0.0194, 21e-4, 0.498, -0.8669, 0.0318, 0.0288, -0.1345, 0.99, -0.0558, 0.0639, -0.1815, 0.9797, -0.0106, 0.0133, 0.2356, 0.9717, -0.0214, -0.0491, -0.6163, 0.7857, 0.011, 0.015, -0.0542, 0.9983],
        [0.8739, 0.9627, 0.0141, 0.6861, 0.7259, -0.0479, 0.0126, 0.0169, 0.0226, 0.0241, 0.9993, -0.0157, -35e-4, -0.1074, 0.9941, -0.0702, 0.0369, 0.1651, 0.9831, -27e-4, 0.0292, 0.2449, 0.9691, 42e-4, -0.0205, -0.0292, 0.9993, -0.0217, 0.0135, -0.2909, 0.9564, 78e-4, 3e-3, 1e-3, 1, -9e-4, 33e-4, -0.0172, 0.9999, 0.0111, 15e-4, -0.016, 0.9998, -0.1342, 0.7342, -0.0392, -0.6643, -0.1011, 0.4598, -0.2308, -0.8516, -33e-4, 0.0139, 0.3916, 0.92, -0.1333, -0.0428, -0.0521, 0.9888, 0, 0, 0, 1, 0, 0, 0, 1, -0.7655, -0.1846, -0.6111, 0.0807, 0.159, -0.3826, 0.1754, 0.8931, -31e-4, -0.0796, 0.3344, 0.9391, -0.2367, 0.093, -0.1215, 0.9595, 0, 0, 0, 1, 0, 0, 0, 1, 0.0726, -0.1044, 0.9919, -9e-4, -0.0209, -71e-4, -0.5133, -0.8579, -0.0254, -0.0109, -0.5014, 0.8648, -0.0288, -0.0285, 0.0854, -0.9955, -0.0484, 0.0737, -0.1444, 0.9856, -0.0125, 0.014, 0.2343, 0.972, -0.0198, -0.0514, -0.6376, 0.7684, 0.0112, 0.014, -0.0529, 0.9984],
        [0.9225, 0.9705, 0.016, 0.6829, 0.729, -0.0423, 0.018, 0.0189, 0.0238, 0.027, 0.9992, -0.0123, -32e-4, -0.1055, 0.9943, -0.0657, 0.0348, 0.1646, 0.9836, -42e-4, 0.0288, 0.2493, 0.968, 38e-4, -0.0198, -0.0296, 0.9994, -0.0217, 0.0152, -0.2923, 0.956, 83e-4, 27e-4, 15e-4, 1, -18e-4, 3e-3, -0.018, 0.9998, 0.0102, 9e-4, -0.016, 0.9998, -0.1395, 0.7383, -0.0376, -0.6589, -0.1141, 0.4658, -0.2298, -0.8469, -34e-4, 0.013, 0.3668, 0.9302, -0.124, -0.0511, -0.0606, 0.9891, 0, 0, 0, 1, 0, 0, 0, 1, -0.7644, -0.182, -0.6132, 0.0812, 0.1534, -0.374, 0.1987, 0.8928, -16e-4, -0.0758, 0.3451, 0.9355, -0.2341, 0.0958, -0.1247, 0.9594, 0, 0, 0, 1, 0, 0, 0, 1, 0.0718, -0.1069, 0.9905, 0.0488, -0.0187, -58e-4, -0.5485, -0.8359, 0.0349, 0.0233, 0.5363, -0.843, 0.029, 0.0282, -0.0775, 0.9962, -0.0399, 0.076, -0.103, 0.991, -0.0137, 0.0142, 0.2186, 0.9756, -0.0175, -0.0492, -0.6523, 0.7561, 0.0109, 0.0121, -0.0502, 0.9986],
        [0.9703, 0.9778, 0.0171, 0.6815, 0.7308, -0.0323, 0.0219, 0.0205, 0.0237, 0.0297, 0.9991, -68e-4, -38e-4, -0.1039, 0.9946, -0.0617, 0.0301, 0.1622, 0.9844, -49e-4, 0.0282, 0.2531, 0.967, 22e-4, -0.0177, -0.0292, 0.9994, -0.0213, 0.0164, -0.2914, 0.9562, 75e-4, 27e-4, 17e-4, 1, -25e-4, 31e-4, -0.0174, 0.9998, 92e-4, 16e-4, -0.0157, 0.9998, -0.1449, 0.7415, -0.0355, -0.6542, -0.1279, 0.4732, -0.2283, -0.8412, -19e-4, 0.0129, 0.3373, 0.9413, -0.1123, -0.0567, -0.0644, 0.99, 0, 0, 0, 1, 0, 0, 0, 1, -0.7625, -0.1761, -0.6174, 0.0796, 0.1466, -0.3682, 0.2176, 0.892, -15e-4, -0.071, 0.3625, 0.9293, -0.2295, 0.0995, -0.1274, 0.9598, 0, 0, 0, 1, 0, 0, 0, 1, 0.0631, -0.1121, 0.987, 0.0966, -0.015, -51e-4, -0.5501, -0.835, 0.0439, 0.0358, 0.5707, -0.8192, 0.03, 0.0273, -0.0795, 0.996, -0.0312, 0.0731, -0.0597, 0.995, -0.0134, 0.014, 0.1928, 0.981, -0.0141, -0.0431, -0.6619, 0.7482, 0.0111, 0.0114, -0.0491, 0.9987],
        [1.0177, 0.9823, 0.0174, 0.6817, 0.7308, -0.0206, 0.026, 0.021, 0.0233, 0.0316, 0.999, -8e-4, -47e-4, -0.1032, 0.9946, -0.0573, 0.0257, 0.1596, 0.9852, -52e-4, 0.0258, 0.2557, 0.9664, 4e-4, -0.0169, -0.0298, 0.9994, -0.0211, 0.0176, -0.2888, 0.957, 69e-4, 28e-4, 4e-4, 1, -23e-4, 32e-4, -0.0174, 0.9998, 69e-4, 17e-4, -0.0168, 0.9998, -0.1499, 0.7447, -0.0323, -0.6496, -0.1448, 0.4805, -0.225, -0.8351, -14e-4, 0.0155, 0.3096, 0.9507, -0.1044, -0.0619, -0.0665, 0.9904, 0, 0, 0, 1, 0, 0, 0, 1, -0.7584, -0.1677, -0.6252, 0.0762, 0.1359, -0.3659, 0.2315, 0.8911, 3e-4, -0.069, 0.3851, 0.9203, -0.2272, 0.1048, -0.1298, 0.9595, 0, 0, 0, 1, 0, 0, 0, 1, 0.0511, -0.1196, 0.9816, 0.1397, -0.0117, -37e-4, -0.5214, -0.8532, 0.0522, 0.0446, 0.5939, -0.8016, 0.031, 0.0276, -0.0814, 0.9958, -0.0238, 0.0678, -0.0168, 0.9973, -0.0111, 0.0141, 0.1619, 0.9867, -0.0101, -0.0359, -0.6686, 0.7427, 0.0105, 95e-4, -0.0484, 0.9987],
        [1.0651, 0.9833, 0.0172, 0.6831, 0.7296, -94e-4, 0.0316, 0.0216, 0.0237, 0.0324, 0.999, 37e-4, -5e-3, -0.1036, 0.9946, -0.0518, 0.0226, 0.1583, 0.9858, -48e-4, 0.0193, 0.2581, 0.9659, -22e-4, -0.0152, -0.0315, 0.9994, -0.0196, 0.0188, -0.2855, 0.958, 6e-3, 27e-4, 6e-4, 1, -33e-4, 31e-4, -0.0192, 0.9998, 51e-4, 14e-4, -0.0185, 0.9998, -0.1544, 0.7491, -0.0274, -0.6437, -0.1655, 0.4859, -0.2203, -0.8295, -5e-4, 0.0217, 0.287, 0.9577, -0.1022, -0.069, -0.0663, 0.9901, 0, 0, 0, 1, 0, 0, 0, 1, -0.7525, -0.1579, -0.6352, 0.0727, 0.1219, -0.3643, 0.2421, 0.891, -6e-4, -0.0711, 0.4104, 0.9091, -0.2195, 0.1111, -0.132, 0.9602, 0, 0, 0, 1, 0, 0, 0, 1, 0.0394, -0.128, 0.9757, 0.1737, -0.01, -15e-4, -0.4636, -0.886, 0.0587, 0.052, 0.6066, -0.7911, 0.0319, 0.0265, -0.083, 0.9957, -0.0174, 0.0627, 0.0236, 0.9976, -66e-4, 0.0144, 0.1319, 0.9911, -66e-4, -0.0313, -0.675, 0.7372, 0.0103, 89e-4, -0.0464, 0.9988],
        [1.1132, 0.9805, 0.0165, 0.6849, 0.7276, -3e-4, 0.0393, 0.0209, 0.0258, 0.0323, 0.9989, 89e-4, -42e-4, -0.1042, 0.9945, -0.0466, 0.0204, 0.1578, 0.9861, -29e-4, 0.0108, 0.2599, 0.9656, -48e-4, -0.0127, -0.0329, 0.9994, -0.0179, 0.0196, -0.2845, 0.9583, 49e-4, 28e-4, 6e-4, 1, -5e-3, 35e-4, -0.0212, 0.9998, 46e-4, 14e-4, -0.0199, 0.9998, -0.1599, 0.7542, -0.0225, -0.6365, -0.1835, 0.4885, -0.214, -0.8257, 2e-3, 0.0278, 0.271, 0.9622, -0.101, -0.0784, -0.0649, 0.9897, 0, 0, 0, 1, 0, 0, 0, 1, -0.7467, -0.1494, -0.6443, 0.0714, 0.1032, -0.36, 0.2524, 0.8922, -5e-4, -0.0765, 0.4357, 0.8969, -0.2085, 0.1179, -0.1326, 0.9618, 0, 0, 0, 1, 0, 0, 0, 1, 0.0304, -0.1354, 0.9711, 0.1943, -0.0105, 2e-4, -0.3738, -0.9274, 0.0639, 0.0601, 0.6131, -0.7852, 0.0318, 0.0252, -0.076, 0.9963, -0.0113, 0.0597, 0.0601, 0.9963, -1e-3, 0.0138, 0.1082, 0.994, -44e-4, -0.0294, -0.6828, 0.73, 0.0106, 0.0106, -0.0459, 0.9988],
        [1.1623, 0.9746, 0.015, 0.6874, 0.7245, 65e-4, 0.0495, 0.0196, 0.0307, 0.0312, 0.9989, 0.0143, -27e-4, -0.1047, 0.9944, -0.0413, 0.019, 0.1584, 0.9863, -7e-4, 22e-4, 0.2603, 0.9655, -53e-4, -0.0113, -0.0336, 0.9994, -0.0181, 0.0201, -0.2854, 0.958, 39e-4, 23e-4, 1e-4, 1, -5e-3, 36e-4, -0.0227, 0.9997, 42e-4, 6e-4, -0.0218, 0.9998, -0.1653, 0.7588, -0.0177, -0.6298, -0.1986, 0.4898, -0.2054, -0.8237, 47e-4, 0.0321, 0.2633, 0.9642, -0.0972, -0.0888, -0.0643, 0.9892, 0, 0, 0, 1, 0, 0, 0, 1, -0.7411, -0.1431, -0.6519, 0.0731, 0.0825, -0.353, 0.2637, 0.8939, 19e-4, -0.0813, 0.4595, 0.8845, -0.1967, 0.1248, -0.1324, 0.9634, 0, 0, 0, 1, 0, 0, 0, 1, 0.0246, -0.1393, 0.9695, 0.2003, -0.012, 8e-4, -0.2545, -0.967, 0.0675, 0.063, 0.6144, -0.7836, 0.0327, 0.0267, -0.0672, 0.9968, -48e-4, 0.0589, 0.0911, 0.9941, 44e-4, 0.0133, 0.0936, 0.9955, -39e-4, -0.0258, -0.6909, 0.7224, 0.0105, 0.0115, -0.0468, 0.9988],
        [1.2125, 0.9669, 0.0131, 0.6899, 0.7212, 0.0129, 0.0616, 0.0196, 0.0358, 0.0301, 0.9987, 0.0184, -14e-4, -0.1052, 0.9943, -0.0354, 0.0188, 0.1596, 0.9864, -13e-4, -57e-4, 0.2595, 0.9657, -9e-4, -0.0112, -0.0336, 0.9994, -0.0209, 0.0209, -0.2884, 0.957, 3e-3, 17e-4, 6e-4, 1, -53e-4, 33e-4, -0.0232, 0.9997, 41e-4, 1e-4, -0.0225, 0.9997, -0.17, 0.7618, -0.0148, -0.6249, -0.208, 0.492, -0.1948, -0.8226, 74e-4, 0.0348, 0.2626, 0.9642, -0.0915, -0.097, -0.0647, 0.9889, 0, 0, 0, 1, 0, 0, 0, 1, -0.7361, -0.1379, -0.6582, 0.0765, 0.0641, -0.3453, 0.2753, 0.8949, 6e-3, -0.0822, 0.4797, 0.8736, -0.1857, 0.1312, -0.1331, 0.9647, 0, 0, 0, 1, 0, 0, 0, 1, 0.0212, -0.1398, 0.9702, 0.197, 0.0126, -17e-4, 0.1306, 0.9913, -0.0693, -0.0619, -0.613, 0.7846, 0.0337, 0.0274, -0.063, 0.9971, 29e-4, 0.0575, 0.1172, 0.9914, 81e-4, 0.0129, 0.0891, 0.9959, -35e-4, -0.0204, -0.6986, 0.7153, 95e-4, 95e-4, -0.0518, 0.9986],
        [1.2637, 0.9588, 0.0108, 0.692, 0.7181, 0.02, 0.0712, 0.0199, 0.0381, 0.0287, 0.9987, 0.0222, -5e-4, -0.1056, 0.9942, -0.0313, 0.0184, 0.1609, 0.9863, -21e-4, -0.0101, 0.2583, 0.966, 49e-4, -0.0119, -0.0326, 0.9994, -0.025, 0.0226, -0.2927, 0.9556, 32e-4, 14e-4, 23e-4, 1, -62e-4, 38e-4, -0.0211, 0.9998, 27e-4, 1e-4, -0.0207, 0.9998, -0.1757, 0.7632, -0.0152, -0.6217, -0.2133, 0.4926, -0.1884, -0.8224, 76e-4, 0.0373, 0.2647, 0.9636, -0.0873, -0.1022, -0.0651, 0.9888, 0, 0, 0, 1, 0, 0, 0, 1, -0.7327, -0.1329, -0.6626, 0.0799, 0.0483, -0.3408, 0.2832, 0.8952, 0.0101, -0.0821, 0.4953, 0.8648, -0.1785, 0.1391, -0.1332, 0.9649, 0, 0, 0, 1, 0, 0, 0, 1, 0.0173, -0.1368, 0.9712, 0.1942, 0.0107, -33e-4, 0.0464, 0.9989, -0.0702, -0.0631, -0.6143, 0.7834, 0.0347, 0.0299, -0.0628, 0.997, 9e-3, 0.0516, 0.139, 0.9889, 9e-3, 0.0117, 0.0945, 0.9954, -35e-4, -0.0108, -0.7036, 0.7105, 8e-3, 66e-4, -0.0654, 0.9978],
        [1.3162, 0.9522, 77e-4, 0.6937, 0.7159, 0.0284, 0.0733, 0.019, 0.0345, 0.0273, 0.9989, 0.0262, 1e-4, -0.1075, 0.9939, -0.0297, 0.0164, 0.1619, 0.9862, -27e-4, -0.0107, 0.2579, 0.9661, 62e-4, -0.0117, -0.0322, 0.9994, -0.0261, 0.0245, -0.2937, 0.9552, 47e-4, 17e-4, 28e-4, 1, -44e-4, 45e-4, -0.0187, 0.9998, 33e-4, 6e-4, -0.0191, 0.9998, -0.1799, 0.7632, -0.0163, -0.6204, -0.2162, 0.4914, -0.1842, -0.8233, 81e-4, 0.0381, 0.2665, 0.963, -0.0816, -0.1045, -0.063, 0.9892, 0, 0, 0, 1, 0, 0, 0, 1, -0.7314, -0.1294, -0.6646, 0.0814, 0.0394, -0.3404, 0.2859, 0.8949, 0.0109, -0.0799, 0.5085, 0.8573, -0.1737, 0.1472, -0.1314, 0.9648, 0, 0, 0, 1, 0, 0, 0, 1, 76e-4, -0.1326, 0.9723, 0.1924, 0.0114, -36e-4, 0.0253, 0.9996, -0.0724, -0.0689, -0.6138, 0.7831, 0.0355, 0.0296, -0.0685, 0.9966, 99e-4, 0.0368, 0.1542, 0.9873, 83e-4, 0.0103, 0.1141, 0.9934, -35e-4, 29e-4, -0.7007, 0.7134, 74e-4, 56e-4, -0.0965, 0.9953],
        [1.3706, 0.9484, 41e-4, 0.695, 0.7148, 0.042, 0.065, 0.0154, 0.0213, 0.0252, 0.9993, 0.0331, 1e-4, -0.11, 0.9934, -0.0303, 0.0132, 0.1614, 0.9863, -25e-4, -62e-4, 0.2587, 0.9659, 41e-4, -0.013, -0.0319, 0.9994, -0.0258, 0.0261, -0.2905, 0.9562, 73e-4, 24e-4, 14e-4, 1, -12e-4, 49e-4, -0.0179, 0.9998, 64e-4, 18e-4, -0.0197, 0.9998, -0.1802, 0.7643, -0.0174, -0.619, -0.216, 0.4879, -0.1795, -0.8265, 92e-4, 0.0361, 0.2654, 0.9634, -0.0744, -0.1035, -0.0601, 0.99, 0, 0, 0, 1, 0, 0, 0, 1, -0.7352, -0.1294, -0.6605, 0.0802, 0.0351, -0.341, 0.282, 0.8961, 0.0128, -0.0784, 0.5187, 0.8512, -0.1729, 0.1518, -0.1271, 0.9648, 0, 0, 0, 1, 0, 0, 0, 1, -0.0142, -0.1306, 0.9729, 0.1901, 0.0167, -25e-4, 0.0523, 0.9985, -0.0685, -0.0827, -0.5958, 0.7959, 0.0357, 0.0274, -0.0764, 0.9961, 55e-4, 83e-4, 0.155, 0.9879, 72e-4, 0.0104, 0.1611, 0.9869, 16e-4, 0.0157, -0.6856, 0.7278, 85e-4, 51e-4, -0.1447, 0.9894]
      ]
    },
    run: {
      travel: [3.1071, 0],
      frames: [
        [0, 0.9487, 0, 0.7226, 0.6905, 0.0178, 0.0288, 0.0221, 0.0152, 0.0685, 0.9973, -0.022, 0.0157, -0.1147, 0.993, -0.0552, 0.019, 0.1321, 0.9895, 0.0617, -0.0233, 0.1336, 0.9888, -0.0318, -0.0502, -83e-4, 0.9982, -0.0168, -0.0156, -0.3201, 0.9471, 0.0113, 74e-4, 0.0179, 0.9998, 48e-4, 0.0201, -12e-4, 0.9998, 0.0116, -4e-4, 0.0187, 0.9998, 0.2225, -0.7093, 0.1015, 0.6612, -0.017, -0.418, 0.079, 0.9048, -0.0821, 96e-4, 0.8018, 0.5919, -0.1987, 0.1398, 0.1287, 0.9615, 0, 0, 0, 1, 0, 0, 0, 1, -0.7468, -0.1566, -0.645, 0.0415, 0.0132, -0.4022, 0.2296, 0.8862, -22e-4, -0.1029, 0.897, 0.4299, -0.0796, 0.1602, -0.0833, 0.9803, 0, 0, 0, 1, 0, 0, 0, 1, -0.0321, 0.0719, -0.9942, -0.0728, 0.0536, -0.0221, 0.8998, 0.4324, -0.0384, 0.0152, -0.6207, 0.783, 0.0281, -68e-4, -0.0289, 0.9992, 0.0323, 0.0973, -0.2272, 0.9684, -0.0771, -0.0419, 0.3782, 0.9215, 0.0616, 38e-4, -0.669, 0.7407, 0.0417, -0.0106, -0.035, 0.9985],
        [0.1481, 0.9439, -32e-4, 0.7207, 0.691, 0.0331, 0.0458, 0.0297, 0.026, 0.0752, 0.9964, 18e-4, 0.0183, -0.1127, 0.9935, -0.0368, 0.0219, 0.1331, 0.9902, 0.0614, -0.0341, 0.1273, 0.9894, -0.04, -0.0514, -0.0101, 0.9978, -0.0257, -0.0199, -0.3183, 0.9474, 82e-4, 54e-4, 0.0172, 0.9998, 4e-3, 0.019, -4e-3, 0.9998, 73e-4, -2e-4, 0.0163, 0.9998, -0.2416, 0.7245, -0.0863, -0.6398, 0.0592, -0.4103, 0.0175, 0.9099, 0.0767, -68e-4, -0.8051, -0.5881, -0.2341, 0.1273, 0.1375, 0.954, 0, 0, 0, 1, 0, 0, 0, 1, -0.7319, -0.1078, -0.6713, 0.0448, -0.0965, -0.4096, 0.3238, 0.8474, 95e-4, -0.0989, 0.8996, 0.4252, -0.1007, 0.1922, -0.093, 0.9717, 0, 0, 0, 1, 0, 0, 0, 1, 0.0442, -0.0675, 0.9729, 0.2169, -0.0526, 0.0209, -0.9248, -0.3763, -0.0497, 0.0286, -0.6471, 0.7603, 0.0466, -0.0118, -74e-4, 0.9988, 0.0416, 0.0945, -0.1472, 0.9837, -0.0737, -0.0373, 0.3941, 0.9154, 0.0621, -3e-3, -0.7134, 0.698, 0.0371, -0.0239, -0.0433, 0.9981],
        [0.3009, 0.9554, -63e-4, 0.7234, 0.6871, 0.0564, 0.0369, 0.026, 0.0235, 0.0759, 0.9965, 0.0341, 0.0173, -0.1107, 0.9931, -0.0204, 0.011, 0.1258, 0.9918, 0.0595, -0.0314, 0.1291, 0.9893, -0.0449, -0.0546, -0.0141, 0.9974, -0.0451, -0.0222, -0.306, 0.9507, 0.0128, 77e-4, 0.0158, 0.9998, 91e-4, 0.0202, -51e-4, 0.9997, 48e-4, 29e-4, 0.015, 0.9999, -0.2538, 0.7279, -0.0673, -0.6334, 0.1226, -0.3806, -0.0506, 0.9152, 0.0765, 0.0276, -0.8274, -0.5557, -0.2424, 0.1186, 0.1388, 0.9528, 0, 0, 0, 1, 0, 0, 0, 1, -0.7041, -0.0669, -0.7051, 0.0513, -0.1807, -0.4218, 0.3734, 0.8062, 0.0115, -0.083, 0.9054, 0.4162, 0.1168, -0.2138, 0.0941, -0.9653, 0, 0, 0, 1, 0, 0, 0, 1, 0.015, -0.0626, 0.9349, 0.3491, -0.0461, 0.0275, -0.9095, -0.4122, -0.0891, 0.0177, -0.6789, 0.7286, 0.0552, -12e-4, 89e-4, 0.9984, 0.0403, 0.0555, -0.0384, 0.9969, -0.0681, -0.0254, 0.3389, 0.938, 0.055, 0.02, -0.6974, 0.7143, 0.0307, -0.0327, -0.1121, 0.9927],
        [0.4602, 0.9761, -95e-4, 0.732, 0.6766, 0.0791, 0.0117, 0.018, 0.0134, 0.0671, 0.9975, 0.0641, 0.0149, -0.1115, 0.9916, -75e-4, -67e-4, 0.1148, 0.9933, 0.0585, -0.0231, 0.1382, 0.9884, -0.0517, -0.0585, -0.0189, 0.9968, -0.066, -0.0183, -0.2852, 0.956, 0.0205, 0.0103, 0.0136, 0.9996, 0.0173, 0.0209, -53e-4, 0.9996, 74e-4, 47e-4, 0.0132, 0.9999, -0.2677, 0.7227, -0.0559, -0.6347, 0.1715, -0.3427, -0.0985, 0.9184, 0.0811, 0.0575, -0.85, -0.5174, -0.2377, 0.1093, 0.1309, 0.9562, 0, 0, 0, 1, 0, 0, 0, 1, -0.6743, -0.0432, -0.7349, 0.0593, -0.2199, -0.421, 0.3942, 0.7868, 89e-4, 0.0675, -0.9116, -0.4055, -0.1316, 0.2192, -0.0843, 0.9631, 0, 0, 0, 1, 0, 0, 0, 1, -0.0331, -0.0684, 0.895, 0.4396, -0.0398, 0.0376, -0.8477, -0.5276, 0.1163, 0.0141, 0.7032, -0.7013, 0.047, 0.0112, 0.0129, 0.9988, 0.0198, 29e-4, 0.0636, 0.9978, -0.0586, -0.0193, 0.2551, 0.9649, 0.0434, 0.0514, -0.5993, 0.7977, 0.0166, -0.0394, -0.2094, 0.9769],
        [0.6243, 0.9972, -0.0124, 0.74, 0.6664, 0.0907, -0.0101, 0.0124, 68e-4, 0.056, 0.9983, 0.0775, 0.0134, -0.1138, 0.9904, 83e-4, -0.0215, 0.1065, 0.994, 0.0558, -0.0163, 0.1471, 0.9874, -0.0548, -0.0619, -0.0226, 0.9963, -0.083, -0.0119, -0.26, 0.962, 0.0278, 0.0117, 94e-4, 0.9995, 0.0241, 0.0218, -86e-4, 0.9994, 0.0144, 69e-4, 65e-4, 0.9999, -0.2802, 0.7143, -0.05, -0.6394, 0.1975, -0.3157, -0.128, 0.9192, 0.0752, 0.0711, -0.8689, -0.4841, -0.2343, 0.1013, 0.1153, 0.96, 0, 0, 0, 1, 0, 0, 0, 1, -0.6566, -0.0404, -0.75, 0.0688, -0.2255, -0.4063, 0.3967, 0.7916, 0.0318, 0.066, -0.9201, -0.3848, -0.1386, 0.2092, -0.0769, 0.9649, 0, 0, 0, 1, 0, 0, 0, 1, -0.0683, -0.0741, 0.8762, 0.4713, -0.0336, 0.0414, -0.7273, -0.6842, 0.1103, 0.0316, 0.715, -0.6896, 0.0365, 0.0134, 0.018, 0.9991, -42e-4, -0.0334, 0.1356, 0.9902, -0.0454, -0.0231, 0.2054, 0.9774, 0.0156, 0.0639, -0.4713, 0.8795, 0.0146, -0.0325, -0.2182, 0.9752],
        [0.7893, 1.0112, -0.0146, 0.7418, 0.6648, 0.0875, -0.0133, 0.01, 0.0121, 0.05, 0.9986, 0.0815, 0.0144, -0.1146, 0.99, 0.018, -0.0287, 0.1037, 0.994, 0.0581, -0.012, 0.1541, 0.9863, -0.059, -0.0642, -0.0234, 0.9959, -0.0909, -46e-4, -0.2462, 0.9649, 0.0313, 0.01, 0.0111, 0.9994, 0.0206, 0.0212, -0.01, 0.9995, 0.017, 74e-4, 36e-4, 0.9998, -0.2875, 0.707, -0.0415, -0.6449, 0.1991, -0.3101, -0.148, 0.9178, 0.0553, 0.0741, -0.8794, -0.4669, -0.2366, 0.0998, 0.106, 0.9607, 0, 0, 0, 1, 0, 0, 0, 1, -0.6547, -0.0466, -0.7505, 0.0778, -0.2052, -0.3939, 0.3916, 0.8058, 0.0391, 0.0719, -0.928, -0.3636, -0.1272, 0.1905, -0.0762, 0.9704, 0, 0, 0, 1, 0, 0, 0, 1, -0.079, -0.0741, 0.8876, 0.4478, -0.0249, 0.0397, -0.5487, -0.8347, 0.0997, 0.0291, 0.704, -0.7026, 0.0305, 81e-4, 0.0198, 0.9993, -0.0106, -0.0392, 0.1763, 0.9835, -0.0359, -0.0376, 0.2392, 0.9696, -0.0147, 0.0678, -0.4045, 0.9119, 0.0284, -0.0225, -0.1419, 0.9892],
        [0.9523, 1.0142, -0.0155, 0.7387, 0.6701, 0.0737, 7e-4, 0.0108, 0.0251, 0.0509, 0.9983, 0.0829, 0.0155, -0.1158, 0.9897, 0.0144, -0.025, 0.104, 0.9942, 0.0633, -0.0125, 0.1594, 0.9851, -0.0604, -0.0618, -0.022, 0.996, -0.093, 16e-4, -0.2486, 0.9641, 0.0331, 81e-4, 0.0134, 0.9993, 0.02, 0.0214, -56e-4, 0.9996, 0.0163, 66e-4, 8e-3, 0.9998, -0.2887, 0.7028, -0.0271, -0.6496, 0.203, -0.3322, -0.1545, 0.908, 0.041, 0.0624, -0.8808, -0.4676, -0.2371, 0.1003, 0.1061, 0.9605, 0, 0, 0, 1, 0, 0, 0, 1, -0.6605, -0.056, -0.7439, 0.0856, -0.1726, -0.3953, 0.3868, 0.815, 0.0247, 0.0779, -0.9312, -0.3553, -0.1132, 0.1711, -0.0742, 0.9759, 0, 0, 0, 1, 0, 0, 0, 1, -0.0732, -0.0703, 0.9138, 0.3933, -0.014, 0.0378, -0.3559, -0.9337, 0.095, 0.0105, 0.6729, -0.7336, 0.0298, 6e-4, 0.0173, 0.9994, 6e-4, -0.0189, 0.1871, 0.9822, -0.0341, -0.0556, 0.3537, 0.9331, -0.0386, 0.0759, -0.4114, 0.9075, 0.0353, -0.0201, -0.0939, 0.9948],
        [1.1117, 1.0066, -0.0154, 0.7346, 0.6759, 0.0557, 0.0185, 87e-4, 0.037, 0.0516, 0.9979, 0.084, 0.0146, -0.1174, 0.9894, 11e-4, -0.0147, 0.1066, 0.9942, 0.0686, -0.0106, 0.1593, 0.9848, -0.0617, -0.0573, -0.0194, 0.9963, -0.092, 44e-4, -0.2615, 0.9608, 0.0362, 83e-4, 0.0127, 0.9992, 0.0254, 0.0229, -1e-4, 0.9994, 0.0161, 69e-4, 0.0135, 0.9998, -0.2846, 0.7026, -0.0152, -0.6521, 0.2015, -0.3787, -0.1446, 0.8917, 0.038, 0.0562, -0.874, -0.4811, -0.2323, 0.0964, 0.1099, 0.9616, 0, 0, 0, 1, 0, 0, 0, 1, -0.6704, -0.0679, -0.7328, 0.0944, -0.1326, -0.4082, 0.385, 0.8171, 22e-4, -0.0818, 0.9274, 0.3649, 0.0995, -0.1638, 0.0653, -0.9793, 0, 0, 0, 1, 0, 0, 0, 1, -0.0648, -0.064, 0.9381, 0.3341, 64e-4, -0.0356, 0.2159, 0.9758, -0.093, 0.0157, -0.6478, 0.756, 0.0253, 52e-4, 0.0214, 0.9994, 0.0126, 0.0123, 0.1647, 0.9862, -0.0344, -0.0726, 0.5093, 0.8568, -0.0556, 0.0877, -0.4482, 0.8879, 0.0424, -0.0214, -0.1006, 0.9938],
        [1.2665, 0.99, -0.0153, 0.7307, 0.6811, 0.0416, 0.0208, 2e-3, 0.0352, 0.0531, 0.998, 0.0766, 0.0124, -0.1201, 0.9897, -0.0149, -75e-4, 0.1115, 0.9936, 0.073, -1e-4, 0.1501, 0.986, -0.0674, -0.0553, -0.0157, 0.9961, -0.0852, 69e-4, -0.2768, 0.9571, 0.0403, 92e-4, 0.0134, 0.9991, 0.027, 0.0238, 17e-4, 0.9993, 0.0156, 73e-4, 0.018, 0.9997, -0.2707, 0.7129, -0.0161, -0.6467, 0.1702, -0.4317, -0.1127, 0.8786, 0.0312, 0.0519, -0.8608, -0.5053, -0.2418, 0.0925, 0.1142, 0.9591, 0, 0, 0, 1, 0, 0, 0, 1, -0.6898, -0.0823, -0.7127, 0.0972, -0.086, -0.4269, 0.3764, 0.8177, 0.0364, -0.086, 0.9131, 0.3969, -0.0882, 0.1746, -0.0505, 0.9794, 0, 0, 0, 1, 0, 0, 0, 1, -0.0728, -0.054, 0.9567, 0.2765, 49e-4, -0.033, 0.1618, 0.9863, -0.0747, 0.0212, -0.6328, 0.7704, 0.0243, 0.0121, 0.029, 0.9992, 89e-4, 0.0242, 0.1077, 0.9939, -0.0267, -0.0839, 0.6671, 0.7398, -0.0627, 0.0962, -0.4822, 0.8685, 0.0524, -0.0277, -0.1283, 0.99],
        [1.4082, 0.9693, -0.0158, 0.7285, 0.6838, 0.0415, -16e-4, -0.0104, 0.0143, 0.0568, 0.9982, 0.0678, 81e-4, -0.1212, 0.9903, -0.0354, -0.0119, 0.1169, 0.9924, 0.0694, 0.0201, 0.13, 0.9889, -0.0628, -0.0585, -0.0132, 0.9962, -0.0811, 98e-4, -0.2978, 0.9511, 0.041, 8e-3, 0.0154, 0.999, 0.0268, 0.0252, 6e-4, 0.9993, 0.0186, 75e-4, 0.0187, 0.9996, -0.2491, 0.7308, -0.0342, -0.6345, 0.0985, -0.4688, -0.0462, 0.8766, 0.0236, 0.0561, -0.8455, -0.5305, -0.2545, 0.0899, 0.1208, 0.9553, 0, 0, 0, 1, 0, 0, 0, 1, -0.7178, -0.1084, -0.6814, 0.0926, -0.0439, -0.4447, 0.3552, 0.8211, 0.0694, -0.0938, 0.8878, 0.4452, -0.089, 0.194, -0.0326, 0.9764, 0, 0, 0, 1, 0, 0, 0, 1, -0.1071, -0.0414, 0.9655, 0.2338, 61e-4, -0.0347, 0.1916, 0.9808, -0.0338, -0.0122, -0.6357, 0.7711, 0.0356, 0.0206, 0.0459, 0.9981, -26e-4, -84e-4, 0.0137, 0.9999, -0.0103, -0.0813, 0.7951, 0.601, -0.0559, 0.0816, -0.5119, 0.8533, 0.059, -0.0369, -0.1392, 0.9878],
        [1.5572, 0.9521, -0.0167, 0.7262, 0.6849, 0.0461, -0.0362, -0.0294, -87e-4, 0.0625, 0.9976, 0.0629, 4e-4, -0.1227, 0.9905, -0.0625, -0.028, 0.1268, 0.9896, 0.0595, 0.0464, 0.1088, 0.9912, -0.0461, -0.0615, -0.0131, 0.997, -0.0805, 0.0104, -0.3266, 0.9417, 0.037, 8e-3, 0.0168, 0.9991, 0.0205, 0.0263, 11e-4, 0.9994, 0.0163, 43e-4, 0.0204, 0.9997, -0.2248, 0.7373, -0.0621, -0.634, -0.0162, -0.4812, 0.0666, 0.8739, 0.0204, 0.0691, -0.8354, -0.5449, -0.2531, 0.1021, 0.1158, 0.955, 0, 0, 0, 1, 0, 0, 0, 1, -0.7429, -0.1493, -0.6463, 0.0896, -59e-4, -0.4603, 0.3038, 0.8341, 0.0858, -0.1012, 0.8546, 0.5021, -0.1052, 0.198, -52e-4, 0.9745, 0, 0, 0, 1, 0, 0, 0, 1, -0.1462, -0.0272, 0.9714, 0.185, 75e-4, -0.0409, 0.2537, 0.9664, -37e-4, -0.0456, -0.6818, 0.7301, 0.041, 0.0204, 0.0653, 0.9968, -31e-4, -0.0612, -0.1228, 0.9905, 36e-4, -0.0706, 0.8846, 0.461, -0.0366, 0.0531, -0.5468, 0.8348, 0.073, -0.0442, -0.1227, 0.9888],
        [1.7056, 0.9512, -0.0172, 0.7248, 0.6858, 0.0363, -0.0548, -0.0413, -0.0137, 0.0688, 0.9967, 0.0421, -67e-4, -0.1227, 0.9915, -0.0793, -0.0371, 0.1317, 0.9874, 0.054, 0.0594, 0.0997, 0.9918, -0.0372, -0.0581, -0.0119, 0.9976, -0.0716, 0.0124, -0.3453, 0.9357, 0.0302, 72e-4, 0.0168, 0.9994, 0.015, 0.0258, 13e-4, 0.9996, 0.0103, 17e-4, 0.0188, 0.9998, -0.1937, 0.7228, -0.0781, -0.6588, 0.124, 0.4784, -0.1703, -0.8525, -0.0216, -0.0699, 0.8426, 0.5336, -0.2534, 0.1224, 0.0924, 0.9551, 0, 0, 0, 1, 0, 0, 0, 1, -0.7519, -0.1866, -0.6269, 0.0833, 0.0416, -0.4647, 0.2254, 0.8553, 0.076, -0.108, 0.8343, 0.5353, -0.1196, 0.1787, 0.0136, 0.9765, 0, 0, 0, 1, 0, 0, 0, 1, -0.1488, -0.0122, 0.9842, 0.0951, -55e-4, 0.0414, -0.2556, -0.9659, -58e-4, 0.0412, 0.7325, -0.6795, 0.039, 0.0202, 0.0536, 0.9976, 91e-4, -0.0802, -0.2725, 0.9588, -5e-4, -0.0594, 0.9226, 0.3811, -0.0147, 0.0394, -0.5821, 0.812, 0.0836, -0.0373, -0.1013, 0.9906],
        [1.8562, 0.9655, -0.0175, 0.7283, 0.6835, 0.012, -0.0481, -0.0435, -48e-4, 0.0702, 0.9966, 0.0142, -68e-4, -0.1218, 0.9924, -0.0921, -0.0291, 0.1271, 0.9872, 0.0564, 0.0568, 0.1026, 0.9915, -0.0353, -0.0505, -0.0102, 0.998, -0.0607, 0.0109, -0.3464, 0.9361, 0.0225, 56e-4, 0.014, 0.9996, 0.0139, 0.0252, -1e-4, 0.9996, 6e-3, 49e-4, 0.0152, 0.9999, -0.1576, 0.6938, -0.0807, -0.698, 0.2079, 0.4796, -0.2197, -0.8237, -0.0165, -0.0583, 0.8676, 0.4936, -0.2713, 0.1404, 0.063, 0.9501, 0, 0, 0, 1, 0, 0, 0, 1, -0.7461, -0.2171, -0.6254, 0.0712, 0.1028, -0.4511, 0.1523, 0.8733, 0.0486, -0.1123, 0.8365, 0.5341, -0.1136, 0.1535, 0.0181, 0.9814, 0, 0, 0, 1, 0, 0, 0, 1, -0.1137, -75e-4, 0.9932, -0.0214, -47e-4, 0.0394, -0.1858, -0.9818, -0.0127, 0.0178, 0.7287, -0.6845, 0.0432, 0.0226, 92e-4, 0.9988, 0.0175, -0.0571, -0.4059, 0.912, -0.0219, -0.0516, 0.9143, 0.4013, -58e-4, 0.035, -0.6152, 0.7876, 0.0715, -0.0215, -0.0935, 0.9928],
        [2.013, 0.9881, -0.017, 0.7348, 0.6773, -0.0187, -0.0296, -0.0407, 98e-4, 0.0639, 0.9971, -0.0151, -47e-4, -0.1222, 0.9924, -0.1024, -0.0111, 0.1216, 0.9872, 0.0626, 0.0471, 0.1123, 0.9906, -0.0331, -0.039, -97e-4, 0.9987, -0.0521, 38e-4, -0.3347, 0.9409, 0.0196, 45e-4, 0.0139, 0.9997, 0.0129, 0.0237, 17e-4, 0.9996, 24e-4, 52e-4, 0.0149, 0.9999, -0.1296, 0.6597, -0.0825, -0.7356, 0.2638, 0.4757, -0.2346, -0.8057, -0.0216, -0.0437, 0.8946, 0.4442, -0.2858, 0.1511, 0.0397, 0.9455, 0, 0, 0, 1, 0, 0, 0, 1, -0.7325, -0.2424, -0.6333, 0.0599, 0.1777, -0.4279, 0.099, 0.8807, 0.0131, -0.0969, 0.8517, 0.5148, -0.0965, 0.131, 0.0204, 0.9865, 0, 0, 0, 1, 0, 0, 0, 1, 0.0622, 0.0144, -0.9896, 0.1289, 83e-4, -0.0408, 0.0965, 0.9945, -0.0189, -0.0105, 0.6505, -0.7592, 0.0366, 0.0116, -0.0599, 0.9975, 81e-4, -0.014, -0.4925, 0.8702, -0.0492, -0.0507, 0.8554, 0.5132, -85e-4, 0.0231, -0.6292, 0.7768, 0.0537, -0.0127, -0.0952, 0.9939],
        [2.1743, 1.0117, -0.0157, 0.7389, 0.6726, -0.0381, -0.0168, -0.0391, 0.0178, 0.0568, 0.9975, -0.0335, -53e-4, -0.1235, 0.9918, -0.113, 31e-4, 0.1189, 0.9864, 0.0678, 0.0437, 0.1218, 0.9893, -0.0295, -0.0318, -0.011, 0.999, -0.0442, -27e-4, -0.3154, 0.9479, 0.0229, 54e-4, 0.0131, 0.9996, 0.0141, 0.0216, -16e-4, 0.9997, 92e-4, 69e-4, 0.0118, 0.9999, 0.1181, -0.6298, 0.0878, 0.7627, -0.2849, -0.4545, 0.2449, 0.8077, -0.0438, -0.0249, 0.912, 0.4071, -0.2892, 0.1531, 0.0262, 0.9446, 0, 0, 0, 1, 0, 0, 0, 1, -0.7202, -0.2592, -0.6414, 0.0522, 0.2349, -0.4066, 0.0533, 0.8813, -7e-3, -0.0763, 0.8677, 0.4912, -0.0801, 0.1143, 0.0202, 0.99, 0, 0, 0, 1, 0, 0, 0, 1, 0.0225, 0.0227, -0.9785, 0.2036, 0.0147, -0.0447, 0.0524, 0.9975, -0.0121, -0.0261, 0.5258, -0.8501, 0.0245, 0.01, -0.0919, 0.9954, -78e-4, 0.0184, -0.5143, 0.8574, -0.07, -0.0503, 0.7299, 0.6781, 0, 0.0187, -0.6135, 0.7895, 0.0387, -0.0145, -0.0945, 0.9947],
        [2.3371, 1.0287, -0.0143, 0.7372, 0.6745, -0.0335, -0.0207, -0.0434, 78e-4, 0.053, 0.9976, -0.0343, -63e-4, -0.1243, 0.9916, -0.1215, 54e-4, 0.1194, 0.9854, 0.0662, 0.0441, 0.1301, 0.9883, -0.0239, -0.0318, -0.0148, 0.9991, -0.0373, -68e-4, -0.2966, 0.9543, 0.0225, 66e-4, 0.0105, 0.9997, 0.0165, 0.0213, -81e-4, 0.9996, 0.017, 91e-4, 84e-4, 0.9998, 0.12, -0.6124, 0.0954, 0.7756, -0.2829, -0.4292, 0.2535, 0.8195, -0.0574, -84e-4, 0.9164, 0.3961, -0.2814, 0.1567, 0.0251, 0.9464, 0, 0, 0, 1, 0, 0, 0, 1, -0.7119, -0.2708, -0.6461, 0.0492, 0.2535, -0.3965, 0.0188, 0.8821, -0.0117, -0.0719, 0.8787, 0.4719, -0.0622, 0.1022, 0.0162, 0.9927, 0, 0, 0, 1, 0, 0, 0, 1, 0.0163, 0.0228, -0.9693, 0.2444, 0.0212, -0.0508, 0.1014, 0.9933, 0.0108, -0.0207, 0.4627, -0.8862, 0.0213, 0.0133, -0.0467, 0.9986, -0.0177, 0.0206, -0.4741, 0.8801, -0.0802, -0.0424, 0.5283, 0.8442, 0.0139, 0.0374, -0.582, 0.8122, 0.0279, -0.024, -0.0869, 0.9955],
        [2.4987, 1.0344, -0.0125, 0.7294, 0.683, -0.0116, -0.0366, -0.0483, -0.016, 0.0528, 0.9973, -0.0256, -54e-4, -0.1244, 0.9919, -0.123, -8e-4, 0.1234, 0.9847, 0.058, 0.041, 0.1373, 0.988, -0.0132, -0.0362, -0.0166, 0.9991, -0.0388, -62e-4, -0.292, 0.9556, 0.0219, 75e-4, 81e-4, 0.9997, 0.0177, 0.022, -48e-4, 0.9996, 0.0173, 65e-4, 99e-4, 0.9998, 0.1273, -0.6112, 0.1009, 0.7746, -0.2534, -0.4138, 0.2594, 0.835, -0.0609, -15e-4, 0.9059, 0.4191, -0.2664, 0.1559, 0.0447, 0.9501, 0, 0, 0, 1, 0, 0, 0, 1, -0.7083, -0.2772, -0.6474, 0.0492, 0.2425, -0.4017, 0.0138, 0.883, -0.0149, -0.0803, 0.883, 0.4623, -0.0456, 0.0976, 0.0146, 0.9941, 0, 0, 0, 1, 0, 0, 0, 1, 0.0392, 0.0162, -0.9665, 0.2531, 0.0253, -0.0565, 0.2541, 0.9652, -0.0367, 0.0162, -0.493, 0.8691, -0.0207, -0.0112, 3e-3, -0.9997, -0.0186, -6e-4, -0.4022, 0.9154, -0.082, -0.0299, 0.3127, 0.9458, 69e-4, 0.0688, -0.5622, 0.8241, 0.025, -0.0293, -0.0796, 0.9961],
        [2.6566, 1.0273, -99e-4, 0.7206, 0.6917, 94e-4, -0.0467, -0.0426, -0.0326, 0.0554, 0.997, -0.021, -42e-4, -0.1256, 0.9918, -0.1157, -88e-4, 0.1283, 0.9849, 0.0526, 0.0348, 0.1406, 0.9881, -0.0106, -0.038, -0.015, 0.9991, -0.043, -59e-4, -0.2987, 0.9534, 0.0192, 81e-4, 99e-4, 0.9997, 0.0135, 0.0216, -11e-4, 0.9997, 0.0103, 43e-4, 0.0142, 0.9998, 0.1362, -0.6217, 0.1002, 0.7648, -0.1955, -0.4146, 0.2608, 0.8496, -0.06, -3e-4, 0.8827, 0.466, -0.2453, 0.145, 0.0791, 0.9553, 0, 0, 0, 1, 0, 0, 0, 1, -0.7099, -0.2738, -0.6469, 0.0501, 0.2094, -0.4151, 0.0464, 0.8841, -0.0105, -0.0864, 0.883, 0.4612, -0.0515, 0.1035, 0.0192, 0.9931, 0, 0, 0, 1, 0, 0, 0, 1, 0.059, 0.0186, -0.973, 0.2221, 0.0295, -0.0535, 0.4669, 0.8822, -0.0534, 0.0234, -0.5397, 0.8398, 0.0141, 0.0152, -0.0135, 0.9997, -0.0118, -0.0172, -0.3364, 0.9415, -0.0827, -0.0205, 0.185, 0.979, -94e-4, 0.099, -0.5697, 0.8158, 0.0242, -0.0247, -0.0766, 0.9965],
        [2.8108, 1.0089, -62e-4, 0.7142, 0.6987, 0.0165, -0.0385, -0.0284, -0.0281, 0.0571, 0.9976, -0.025, -26e-4, -0.1247, 0.9919, -0.095, -0.0126, 0.1328, 0.9865, 0.0496, 0.0277, 0.1397, 0.9886, -0.0214, -0.0392, -0.0131, 0.9989, -0.0395, -66e-4, -0.3044, 0.9517, 0.0119, 69e-4, 0.0143, 0.9998, 66e-4, 0.0194, 22e-4, 0.9998, 43e-4, 45e-4, 0.0202, 0.9998, 0.1521, -0.6402, 0.0998, 0.7464, -0.1435, -0.4288, 0.2495, 0.8563, -0.0489, -94e-4, 0.8545, 0.5171, -0.222, 0.1261, 0.1137, 0.9601, 0, 0, 0, 1, 0, 0, 0, 1, -0.7204, -0.2542, -0.6435, 0.0486, 0.1589, -0.4199, 0.105, 0.8874, -36e-4, -0.0938, 0.8822, 0.4614, -0.0722, 0.1134, 0.0301, 0.9905, 0, 0, 0, 1, 0, 0, 0, 1, 0.044, 0.0369, -0.9863, 0.1545, 0.0397, -0.0403, 0.6665, 0.7434, -0.0558, 0.0301, -0.5748, 0.8159, 9e-3, 0.0232, -0.0383, 0.999, -69e-4, -96e-4, -0.2834, 0.9589, -0.0838, -0.0157, 0.1695, 0.9818, 0.0111, 0.1107, -0.5699, 0.8142, 0.0247, -0.0145, -0.0767, 0.9966],
        [2.9608, 0.9839, -23e-4, 0.7085, 0.7055, 0.0131, -0.0112, -0.0103, -62e-4, 0.0615, 0.998, -0.0232, -16e-4, -0.1226, 0.9922, -0.073, -78e-4, 0.1401, 0.9874, 0.0468, 0.0191, 0.1339, 0.9897, -0.0345, -0.0441, -0.0101, 0.9984, -0.034, -5e-3, -0.3141, 0.9488, 0.0104, 75e-4, 0.0143, 0.9998, 57e-4, 0.0177, 41e-4, 0.9998, 29e-4, 43e-4, 0.0228, 0.9997, 0.1787, -0.6645, 0.0991, 0.7188, -0.1042, -0.4424, 0.2255, 0.8617, -0.0344, -0.0192, 0.8351, 0.5487, -0.2039, 0.1102, 0.1333, 0.9636, 0, 0, 0, 1, 0, 0, 0, 1, -0.7373, -0.2182, -0.6377, 0.0469, 0.0928, -0.4045, 0.1811, 0.8916, -68e-4, -0.1089, 0.8862, 0.4504, -0.0908, 0.1237, 0.0302, 0.9877, 0, 0, 0, 1, 0, 0, 0, 1, -59e-4, 0.0637, -0.9959, 0.0638, 0.0554, -0.0248, 0.8057, 0.5892, -0.0529, 0.0259, -0.6061, 0.7932, 0.0111, 0.0177, -0.051, 0.9985, -33e-4, 0.0217, -0.2455, 0.9691, -0.0854, -0.0194, 0.2461, 0.9653, 0.0646, 0.0846, -0.5775, 0.8095, 0.0345, -0.0134, -0.0631, 0.9973]
      ]
    }
  }
};
function expand(frame) {
  const k = SOMA_SKELETON_INFO, pose = k.referencePose();
  pose.set(frame.slice(0, 7), 0);
  k.modelToPose.slice(1).forEach((j, i) => pose.set(frame.slice(7 + i * 4, 11 + i * 4), j * 7 + 3));
  return Array.from(pose);
}
const cycle = (c) => ({ travel: c.travel, frames: c.frames.map(expand) });
const SOMA_BRICKS = {
  speeds: { walk: 1.528, run: 4.62 },
  rest: RAW.rest.map(expand),
  walk: RAW.walk.map(expand),
  run: RAW.run.map(expand),
  cycles: { walk: cycle(RAW.cycles.walk), run: cycle(RAW.cycles.run) }
};
const BONES = [
  // from, to, radius
  ["Hips", "Spine1", 0.11],
  ["Spine1", "Spine2", 0.11],
  ["Spine2", "Chest", 0.12],
  ["Chest", "Neck1", 0.13],
  ["Neck1", "Head", 0.045],
  ["LeftShoulder", "LeftArm", 0.05],
  ["LeftArm", "LeftForeArm", 0.048],
  ["LeftForeArm", "LeftHand", 0.04],
  ["RightShoulder", "RightArm", 0.05],
  ["RightArm", "RightForeArm", 0.048],
  ["RightForeArm", "RightHand", 0.04],
  ["LeftLeg", "LeftShin", 0.07],
  ["LeftShin", "LeftFoot", 0.055],
  ["RightLeg", "RightShin", 0.07],
  ["RightShin", "RightFoot", 0.055]
];
class SOMAMannequin {
  constructor({ shadows = true, color = "#f4f2ee", accent = "#16171a" } = {}) {
    this.object = new Group();
    const k = SOMA_SKELETON_INFO;
    const body = new MeshStandardMaterial({ color, roughness: 0.55 });
    const ink = new MeshStandardMaterial({ color: accent, roughness: 0.45 });
    this.materials = [body, ink];
    this.joints = k.names.map((name) => {
      const o = new Object3D();
      o.name = name;
      return o;
    });
    k.parents.forEach((p, j) => (p < 0 ? this.object : this.joints[p]).add(this.joints[j]));
    this.setPose(k.referencePose());
    this.object.updateMatrixWorld(true);
    const world = (name) => this.joints[k.index(name)].getWorldPosition(new Vector3());
    const attach = (name, mesh, at) => {
      const joint = this.joints[k.index(name)];
      joint.worldToLocal(mesh.position.copy(at));
      mesh.quaternion.premultiply(joint.getWorldQuaternion(new Quaternion()).invert());
      joint.add(mesh);
    };
    for (const [from, to, r] of BONES) {
      const a = world(from), b = world(to), length = a.distanceTo(b);
      const mesh = new Mesh(new CapsuleGeometry(r, Math.max(0.01, length - r * 1.2), 6, 16), body);
      mesh.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), b.clone().sub(a).normalize());
      attach(from, mesh, a.clone().add(b).multiplyScalar(0.5));
    }
    for (const name of ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftShin", "RightShin", "LeftLeg", "RightLeg"]) {
      attach(name, new Mesh(new SphereGeometry(name.endsWith("Leg") ? 0.075 : 0.05, 16, 12), ink), world(name));
    }
    const head = world("Head"), headTop = world("HeadEnd");
    attach("Head", new Mesh(new SphereGeometry(0.105, 24, 18).scale(0.92, 1.08, 1), body), head.clone().lerp(headTop, 0.42));
    for (const side of ["Left", "Right"]) {
      const hand = world(side + "Hand"), tip = world(side + "HandMiddleEnd");
      const m = new Mesh(new BoxGeometry(0.04, 0.16, 0.085), ink);
      m.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), tip.clone().sub(hand).normalize());
      attach(side + "Hand", m, hand.clone().lerp(tip, 0.45));
      const ankle = world(side + "Foot"), toe = world(side + "ToeBase");
      const foot = new Mesh(new BoxGeometry(0.1, 0.07, 0.24), ink);
      const along = toe.clone().sub(ankle).setY(0).normalize();
      foot.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), along);
      attach(side + "Foot", foot, ankle.clone().addScaledVector(along, 0.06).setY(0.035));
    }
    this.object.traverse((o) => {
      if (o.isMesh) o.castShadow = o.receiveShadow = shadows;
    });
  }
  /** Show a SOMA pose: local transforms of SOMA_SKELETON, the root in world space. */
  setPose(pose) {
    this.joints.forEach((o, j) => {
      o.position.set(pose[j * 7], pose[j * 7 + 1], pose[j * 7 + 2]);
      o.quaternion.set(pose[j * 7 + 3], pose[j * 7 + 4], pose[j * 7 + 5], pose[j * 7 + 6]);
    });
  }
}
const workerUrl = "" + new URL("motionbricks.worker.js", import.meta.url).href;
class MotionBricksWorkerError extends Error {
  constructor() {
    super(...arguments);
    this.name = "MotionBricksWorkerError";
  }
}
function startWorker() {
  const url = new URL(workerUrl, import.meta.url);
  if (url.origin === location.origin) return new Worker(url, { type: "module" });
  const blob = new Blob([`import ${JSON.stringify(url.href)}`], { type: "text/javascript" });
  const blobUrl = URL.createObjectURL(blob);
  const worker = new Worker(blobUrl, { type: "module" });
  setTimeout(() => URL.revokeObjectURL(blobUrl), 1e4);
  return worker;
}
class MotionBricksRemote {
  constructor(worker) {
    this.backend = "";
    this._id = 0;
    this._pending = /* @__PURE__ */ new Map();
    this.worker = worker;
    worker.onmessage = (e) => {
      const m = e.data;
      const p = this._pending.get(m.id);
      if (!p) return;
      if (m.type === "progress") return p.progress?.(m.loaded, m.total);
      this._pending.delete(m.id);
      if (m.type === "done") p.resolve(m.result);
      else p.reject(new Error(m.message));
    };
    worker.onerror = (e) => {
      const error = new MotionBricksWorkerError(e.message || "The MotionBricks worker failed to start");
      for (const p of this._pending.values()) p.reject(error);
      this._pending.clear();
    };
  }
  async _call(request, progress) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, progress });
      this.worker.postMessage({ ...request, id });
    });
  }
  /**
   * Start the worker and load the model in it. URLs are resolved against the page. It throws a
   * MotionBricksWorkerError when the worker cannot run, and the model's own error when the model fails to load.
   * @param worker a worker running MotionBricksWorker; the default is the one bundled with this package
   */
  static async Load(options, onProgress, worker) {
    let w = worker;
    try {
      w ??= startWorker();
    } catch (e) {
      throw new MotionBricksWorkerError(e instanceof Error ? e.message : String(e));
    }
    const remote = new MotionBricksRemote(w);
    const abs = (u) => new URL(u, location.href).href;
    try {
      const result = await remote._call({ type: "load", options: {
        ...options,
        ort: abs(options.ort),
        model: abs(options.model),
        wasm: options.wasm?.map(abs)
      } }, onProgress);
      remote.backend = result.backend;
      return remote;
    } catch (e) {
      remote.worker.terminate();
      throw e;
    }
  }
  async generate(context, target, options) {
    return this._call({
      type: "generate",
      context: context.map((p) => Float32Array.from(p)),
      target: target.map((p) => Float32Array.from(p)),
      options
    });
  }
}
const K = SOMA_SKELETON_INFO;
const v3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
};
const quat = {
  mul(a, b) {
    return [
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    ];
  },
  inv: (q) => [-q[0], -q[1], -q[2], q[3]],
  norm(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  },
  axisAngle(axis, angle) {
    const a = v3.norm(axis), s = Math.sin(angle / 2);
    return [a[0] * s, a[1] * s, a[2] * s, Math.cos(angle / 2)];
  },
  /** The shortest turn taking direction a onto direction b. */
  between(a, b) {
    const u = v3.norm(a), w = v3.norm(b);
    const c = v3.cross(u, w), d = v3.dot(u, w);
    if (d < -0.999999) {
      const axis = Math.abs(u[0]) < 0.9 ? v3.cross(u, [1, 0, 0]) : v3.cross(u, [0, 1, 0]);
      return quat.axisAngle(axis, Math.PI);
    }
    return quat.norm([c[0], c[1], c[2], 1 + d]);
  },
  rotate(q, v) {
    const [x, y, z, w] = [q[0], q[1], q[2], q[3]];
    const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
  },
  slerp(a, b, t) {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const s = d < 0 ? -1 : 1;
    d = Math.abs(d);
    if (d > 0.9995) return quat.norm([0, 1, 2, 3].map((k) => a[k] + (b[k] * s - a[k]) * t));
    const th = Math.acos(d), sa = Math.sin((1 - t) * th) / Math.sin(th), sb = Math.sin(t * th) / Math.sin(th) * s;
    return [0, 1, 2, 3].map((k) => a[k] * sa + b[k] * sb);
  }
};
class SOMAPoseEditor {
  constructor(pose) {
    this._dirty = false;
    this.pose = Float32Array.from(pose);
    this._world = K.world(this.pose, new Float32Array(K.J * 7));
  }
  get world() {
    if (this._dirty) {
      K.world(this.pose, this._world);
      this._dirty = false;
    }
    return this._world;
  }
  _j(joint) {
    const j = typeof joint === "number" ? joint : K.index(joint);
    if (j < 0) throw new Error("No SOMA joint " + joint);
    return j;
  }
  position(joint) {
    const w = this.world, o = this._j(joint) * 7;
    return [w[o], w[o + 1], w[o + 2]];
  }
  rotation(joint) {
    const w = this.world, o = this._j(joint) * 7 + 3;
    return [w[o], w[o + 1], w[o + 2], w[o + 3]];
  }
  /** Turn a joint, and all below it, by a world-space rotation about the joint. */
  turn(joint, q) {
    const j = this._j(joint), p = K.parents[j];
    const world = quat.mul(q, this.rotation(j));
    const local = p < 0 ? world : quat.mul(quat.inv(this.rotation(p)), world);
    this.pose.set(quat.norm(local), j * 7 + 3);
    this._dirty = true;
    return this;
  }
  /** Turn a joint by an angle about a world axis. */
  turnAbout(joint, axis, angle) {
    return this.turn(joint, quat.axisAngle(axis, angle));
  }
  /** Set a joint's world rotation. */
  setRotation(joint, q) {
    return this.turn(joint, quat.mul(q, quat.inv(this.rotation(joint))));
  }
  /** Move the whole body (the root) in world space. */
  move(d) {
    this.pose[0] += d[0];
    this.pose[1] += d[1];
    this.pose[2] += d[2];
    this._dirty = true;
    return this;
  }
  /**
   * Two-bone IK: turn `upper` and `mid` so that `end` reaches `target`, with `mid` (the knee or elbow) bending
   * toward `pole`, a world direction. The end joint keeps its world rotation unless `keepEnd` is false.
   */
  reach(upper, mid, end, target, pole, keepEnd = true) {
    const endRotation = this.rotation(end);
    let a = this.position(upper), b = this.position(mid), c = this.position(end);
    const lab = v3.len(v3.sub(b, a)), lcb = v3.len(v3.sub(c, b));
    const lat = Math.min(Math.max(v3.len(v3.sub(target, a)), Math.abs(lab - lcb) + 1e-4), lab + lcb - 1e-4);
    const clamp = (x) => Math.min(1, Math.max(-1, x));
    const angle = (u, w) => Math.acos(clamp(v3.dot(v3.norm(u), v3.norm(w))));
    let axis = v3.cross(v3.sub(c, a), v3.sub(b, a));
    if (v3.len(axis) < 1e-5) axis = v3.cross(v3.sub(c, a), pole);
    const acab0 = angle(v3.sub(c, a), v3.sub(b, a)), babc0 = angle(v3.sub(a, b), v3.sub(c, b));
    const acab1 = Math.acos(clamp((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat)));
    const babc1 = Math.acos(clamp((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb)));
    this.turnAbout(upper, axis, acab1 - acab0);
    this.turnAbout(mid, axis, babc1 - babc0);
    a = this.position(upper);
    c = this.position(end);
    this.turn(upper, quat.between(v3.sub(c, a), v3.sub(target, a)));
    a = this.position(upper);
    b = this.position(mid);
    const along = v3.norm(v3.sub(target, a));
    const flat = (v) => v3.sub(v, v3.scale(along, v3.dot(v, along)));
    const now = flat(v3.sub(b, a)), want = flat([pole[0], pole[1], pole[2]]);
    if (v3.len(now) > 1e-5 && v3.len(want) > 1e-5) {
      const twist = Math.atan2(v3.dot(along, v3.cross(now, want)), v3.dot(now, want));
      this.turnAbout(upper, along, twist);
    }
    if (keepEnd) this.setRotation(end, endRotation);
    return this;
  }
  /** Reach with a leg, the knee toward `pole` (by default the way the hips face). */
  leg(side, target, pole) {
    return this.reach(side + "Leg", side + "Shin", side + "Foot", target, pole ?? this.forward());
  }
  /** Reach with an arm, the elbow toward `pole` (by default down and back). */
  arm(side, target, pole, keepHand = false) {
    const f = this.forward();
    return this.reach(side + "Arm", side + "ForeArm", side + "Hand", target, pole ?? [-f[0], -1, -f[2]], keepHand);
  }
  /** The way the hips face, on the ground. */
  forward() {
    const r = this.position("RightLeg"), l = this.position("LeftLeg");
    const across = v3.norm([r[0] - l[0], 0, r[2] - l[2]]);
    return v3.norm(v3.cross([0, 1, 0], across));
  }
  /** The body's left, on the ground. */
  left() {
    const f = this.forward();
    return [f[2], 0, -f[0]];
  }
}
const other = (s) => s === "Left" ? "Right" : "Left";
const sideZ = (s) => s === "Right" ? 1 : -1;
function standing() {
  return new SOMAPoseEditor(SOMA_BRICKS.rest[0]);
}
function trunk(e, lean, turn = 0, tilt = 0, pelvis = 0) {
  const left = e.left(), forward = e.forward();
  if (pelvis) e.turnAbout("Hips", left, pelvis);
  for (const [joint, share] of [["Spine1", 0.4], ["Spine2", 0.35], ["Chest", 0.25]]) {
    if (lean) e.turnAbout(joint, left, lean * share);
    if (turn) e.turnAbout(joint, [0, 1, 0], turn * share);
    if (tilt) e.turnAbout(joint, forward, tilt * share);
  }
  return e;
}
function look(e, at, share = 0.6) {
  const head = e.position("Head");
  const neck = e.position("Neck1");
  const now = v3.sub(head, neck);
  const want = v3.sub(at, neck);
  const q = quatToward(now, want, share);
  e.turn("Neck1", q);
  return e;
}
function quatToward(a, b, share) {
  const u = v3.norm(a), w = v3.norm(b);
  const axis = v3.cross(u, w);
  const angle = Math.acos(Math.min(1, Math.max(-1, v3.dot(u, w)))) * share;
  const s = Math.sin(angle / 2), l = v3.len(axis) || 1;
  return [axis[0] / l * s, axis[1] / l * s, axis[2] / l * s, Math.cos(angle / 2)];
}
function kickPoses({ foot = "Right", radius = 0.11, power = 1 } = {}) {
  const z = sideZ(foot), support = other(foot);
  const swing = 0.22 + 0.1 * power;
  const path = [
    [-radius - 0.14 - swing, 0.2 + 0.08 * power, 0.04 * z],
    [-radius - 0.14, 0.09, 0.02 * z],
    [-radius - 0.14 + swing, 0.2 + 0.1 * power, 0]
  ];
  return [-1, 0, 1].map((k) => {
    const e = standing();
    e.move([-0.24 + 0.03 * k, -0.07, -0.08 * z]);
    trunk(e, -0.1 - 0.08 * k, -0.3 * z * (1 - 0.6 * k), 0.1 * z);
    e.leg(support, [-0.1, 0.07, -0.22 * z]);
    e.leg(foot, path[k + 1], [0.5, k < 0 ? -0.2 : 0.3, 0]);
    const hips = e.position("Hips");
    e.arm(foot, v3.add(hips, [-0.35 + 0.1 * k, 0.05, 0.3 * z]));
    e.arm(support, v3.add(hips, [0.3, 0.3, -0.45 * z]));
    look(e, [0, 0, 0], 0.5);
    return e.pose;
  });
}
function carryHands(hands, width = 0.3, side = "Right") {
  if (hands === "two") return { Left: [0.32, 1.02, -width / 2 - 0.01], Right: [0.32, 1.02, width / 2 + 0.01] };
  return { [side]: [0.28, 0.98, 0.17 * sideZ(side)] };
}
function reachPoses({ height = 0.1, hands = "one", width = 0.3, side = "Right" } = {}) {
  const z = sideZ(side);
  const two = hands === "two";
  const targets = two ? { Right: [0, height + 0.02, width / 2 + 0.02], Left: [0, height + 0.02, -width / 2 - 0.02] } : { [side]: [0.02, height + 0.03, 0.04 * z] };
  const pose = (depth22, k) => {
    const e = standing();
    const back = two ? 0.4 : 0.36;
    e.move([-back - 0.06 * depth22, -(two ? 0.4 : 0.3) * depth22 - 0.01 * k, two ? 0 : -0.08 * z]);
    trunk(e, (two ? 0.25 : 0.3) + (two ? 0.5 : 0.6) * depth22, two ? 0 : -0.2 * z, 0, (two ? 0.5 : 0.8) * depth22);
    const f = two ? 0 : 0.1;
    e.leg("Right", [-back - 0.02 + f * z, 0.07, 0.19 + (two ? 0.02 : -0.08 * z)]);
    e.leg("Left", [-back - 0.02 - f * z, 0.07, -0.19 + (two ? -0.02 : -0.08 * z)]);
    let miss = 0;
    for (const s of ["Left", "Right"]) {
      const t = targets[s];
      if (!t) continue;
      e.arm(s, t, two ? [-0.3, -0.2, sideZ(s)] : [-0.5, 0.2, 0.6 * z]);
      miss = Math.max(miss, v3.len(v3.sub(e.position(s + "Hand"), t)));
    }
    if (!two) {
      const hips = e.position("Chest");
      e.arm(other(side), v3.add(hips, [0.25, 0.05, -0.22 * z]));
    }
    look(e, [0, height, 0], 0.3);
    return { pose: e.pose, miss };
  };
  let depth2 = 0;
  while (depth2 < 1.6 && pose(depth2, 0).miss > 0.015) depth2 += 0.05;
  reachPoses.lastDepth = depth2;
  return [0, 1].map((k) => pose(depth2, k).pose);
}
reachPoses.lastDepth = 0;
function carryPoses({ hands = "one", width = 0.3, side = "Right" } = {}) {
  const e = standing();
  const at = carryHands(hands, width, side);
  for (const s of ["Left", "Right"]) {
    const p = at[s];
    if (p) e.arm(s, p, [-0.3, -1, 0.6 * sideZ(s)]);
  }
  return [e.pose, Float32Array.from(e.pose)];
}
function throwPoses({ side = "Right", power = 1 } = {}) {
  const z = sideZ(side), lead = other(side);
  const speed = 0.14 + 0.1 * power;
  let release = [0, 0, 0];
  const poses = [-1, 0, 1].map((k) => {
    const e = standing();
    e.move([0.05 + 0.04 * k, -0.06, 0]);
    trunk(e, 0.12 + 0.1 * k, (0.15 + 0.25 * k) * z * -1 + 0.2 * z, -0.05 * z);
    e.leg(lead, [0.38, 0.07, -0.13 * z]);
    e.leg(side, [-0.36, 0.1, 0.18 * z], [0.3, 0.1, 0.2 * z]);
    const hand = [0.12 + speed * k, 1.72 - 0.08 * k - 0.06 * k * k, 0.18 * z];
    e.arm(side, hand, [-0.4, -0.3, 0.8 * z]);
    const hips = e.position("Hips");
    e.arm(lead, v3.add(hips, [0.1, 0, -0.3 * z]));
    if (k === 0) release = e.position(side + "Hand");
    return e.pose;
  });
  return { poses, release };
}
function standPoses() {
  const e = standing();
  return [e.pose, Float32Array.from(e.pose)];
}
const NVIDIA_ASSETS = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets";
const ASSETS = "https://cdn.jsdelivr.net/gh/repalash/three-motionbricks@v0.2.0/assets/";
const MOTIONBRICKS_DEFAULTS = {
  /** NVIDIA's original fp32 model, 730 MB. A pack from tools/pack_model.py is about 190 MB. */
  model: NVIDIA_ASSETS + "/onnx/unitree_g1/mmm_v6_g1_fp32.onnx",
  /** NVIDIA's original fp32 SOMA (human) model, 758 MB, for `rig: 'soma'`. A pack is about 195 MB. */
  somaModel: NVIDIA_ASSETS + "/onnx/soma/mmm_v6_soma_fp32.onnx",
  /** This package's assets/ folder on jsDelivr: the robot and the other characters (see MOTIONBRICKS_CHARACTERS). */
  assets: ASSETS,
  /** The robot, for your GLTFLoader: assets/unitree_g1.glb in this package (6 MB, BSD-3-Clause, Unitree). */
  robot: ASSETS + "unitree_g1.glb",
  /** onnxruntime-web's WebAssembly-only build: half the size of the full one, and what the default backend uses. */
  ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs",
  /** The build with WebGPU, for `backend: 'webgpu'`. */
  ortWebGpu: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs"
};
const MOTIONBRICKS_CHARACTERS = [
  { id: "g1", label: "Unitree G1", file: "unitree_g1.glb" },
  { id: "h1", label: "Unitree H1", file: "robots/unitree_h1.glb", rig: "unitreeH1" },
  { id: "t1", label: "Booster T1", file: "robots/booster_t1.glb", rig: "boosterT1" },
  { id: "op3", label: "Robotis OP3", file: "robots/robotis_op3.glb", rig: "robotisOP3" },
  { id: "robot-expressive", label: "RobotExpressive", file: "characters/robot_expressive.glb", rig: "robotExpressive" },
  { id: "astronaut", label: "Astronaut", file: "characters/astronaut.glb", rig: "mixamo" },
  { id: "polybot", label: "Polybot", file: "characters/polybot.glb", rig: "mixamo" }
];
const range = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const tempA = new Vector3();
const tempB = new Vector3();
function ballistic(from, to, gravity = 9.81) {
  const d = to.clone().sub(from);
  const flat = Math.hypot(d.x, d.z);
  const t = 0.35 + flat * 0.07;
  return new Vector3(d.x / t, (d.y + 0.5 * gravity * t * t) / t, d.z / t);
}
class MotionBricksCharacter {
  constructor(options = {}) {
    this.object = new Group();
    this.gait = "walk";
    this._pendingGoal = null;
    this._drive = { active: false, angle: 0, gait: "walk" };
    this._task = null;
    this._carry = null;
    this._shownWorld = new Float32Array(SOMA_SKELETON_INFO.J * 7);
    this._lastWorld = new Float32Array(SOMA_SKELETON_INFO.J * 7);
    this._lastDt = 1 / 60;
    this._obstacles = [];
    this.options = options;
    this.object.name = "MotionBricks G1";
  }
  get _rig() {
    return this.options.rig === "soma" ? SOMA_RIG : G1_RIG;
  }
  get _bricks() {
    return this.options.bricks ?? (this.options.rig === "soma" ? SOMA_BRICKS : G1_BRICKS);
  }
  /** For `rig: 'soma'`: show the motion on a mannequin built on the SOMA skeleton. It replaces the current model. */
  setMannequin(options = {}) {
    if (this.robot) this.object.remove(this.robot.object);
    const mannequin = new SOMAMannequin({ shadows: this.options.shadows ?? true, ...options });
    this.robot = mannequin;
    mannequin.setPose(this.controller?.pose ?? this._bricks.rest[0]);
    this.object.add(mannequin.object);
    return mannequin;
  }
  /** The model is loaded and the robot responds to moveTo and drive. */
  get ready() {
    return !!this.controller;
  }
  /** 'wasm' or 'webgpu', once the model is loaded. */
  get backend() {
    return this.engine?.backend ?? "";
  }
  get stats() {
    return this.controller?.stats;
  }
  /** The segment being played, for a progress display. */
  get playback() {
    return this.controller?.playback ?? null;
  }
  /** The robot is walking or running. */
  get moving() {
    return !!this.controller?.moving;
  }
  /** Standing still, with no goal and no generation running. */
  get idle() {
    const c = this.controller;
    return !!c && !c.moving && !c.generating && !c.goal;
  }
  /**
   * Use a loaded G1 model, such as `gltf.scene` of MOTIONBRICKS_DEFAULTS.robot, which then stands in its rest pose.
   * Load it with whatever loader your project uses.
   */
  setRobot(root) {
    if (this.robot) this.object.remove(this.robot.object);
    this.robot = G1Robot.FromObject(root, { shadows: this.options.shadows ?? true });
    this.robot.setPose(this.controller?.pose ?? this._bricks.rest[0]);
    this.object.add(root);
    return this.robot;
  }
  /**
   * Use another model, such as a robot from assets/robots or a character with a Mixamo skeleton: the G1's motion
   * is retargeted to it with a rig from CHARACTER_RIGS (or your own). It replaces the current robot.
   */
  setCharacter(root, rig, options = {}) {
    if (this.robot) this.object.remove(this.robot.object);
    const character = new RetargetedCharacter(root, rig, { shadows: this.options.shadows ?? true, source: this.options.rig === "soma" ? "soma" : "g1", ...options });
    this.robot = character;
    character.setPose(this.controller?.pose ?? this._bricks.rest[0]);
    this.object.add(character.object);
    return character;
  }
  /**
   * Load ONNX Runtime and the model, in a worker unless `worker: false`, and warm it up. The robot can be set
   * before or after. Calling it again while loading waits for the same load.
   */
  async load(onProgress) {
    if (this.controller) return this;
    this._loading ??= this._load(onProgress).finally(() => this._loading = void 0);
    return this._loading;
  }
  async _load(onProgress) {
    const o = this.options;
    const useLite = !!o.liteModel && (o.lite === true || (o.lite ?? "auto") === "auto" && isLowMemoryDevice());
    const soma = o.rig === "soma";
    const model = useLite ? o.liteModel : o.model ?? (soma ? MOTIONBRICKS_DEFAULTS.somaModel : MOTIONBRICKS_DEFAULTS.model);
    const backend = o.backend ?? "auto";
    const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
    const cache = o.cache === false ? false : typeof o.cache === "string" ? o.cache : MOTIONBRICKS_CACHE;
    const progress = (l, t) => onProgress?.("model", l, t);
    const create = async (gpu) => {
      const ortUrl = o.ort ?? (gpu ? MOTIONBRICKS_DEFAULTS.ortWebGpu : MOTIONBRICKS_DEFAULTS.ort);
      const providers = backend === "webgpu" ? ["webgpu", "wasm"] : ["webgpu"];
      const sessionOptions = { executionProviders: gpu ? providers : ["wasm"], graphOptimizationLevel: "all" };
      let engine2 = null;
      if (o.worker !== false) {
        engine2 = await MotionBricksRemote.Load({ ort: ortUrl, wasm: o.wasm, model, sessionOptions, cache, rig: soma ? "soma" : "g1" }, progress).catch((e) => {
          if (!(e instanceof MotionBricksWorkerError)) throw e;
          console.warn("MotionBricks: the worker could not start, running the model on the page instead", e);
          return null;
        });
      }
      return engine2 ?? await (soma ? MotionBricksSOMA : MotionBricksG1).Load(await loadOrt(ortUrl, o.wasm), model, sessionOptions, progress, cache);
    };
    const stop = (engine2) => {
      if (engine2 instanceof MotionBricksRemote) engine2.worker.terminate();
    };
    const rest = Float32Array.from(this._bricks.rest[0]);
    const time = async (engine2) => {
      try {
        return (await engine2.generate([rest, rest, rest, rest], [rest])).inferenceMs;
      } catch (e) {
        stop(engine2);
        throw e;
      }
    };
    onProgress?.("runtime", 0, 1);
    const auto = backend === "auto" && hasGpu && !useLite;
    let engine;
    if (auto) {
      engine = await create(true).catch(async (e) => {
        console.warn("MotionBricks: WebGPU did not load, using WebAssembly", e);
        return create(false);
      });
    } else engine = await create(backend === "webgpu" && hasGpu);
    onProgress?.("warmup", 0, 1);
    if (useLite) await new Promise((resolve) => setTimeout(resolve, 2e3));
    let warmupMs = await time(engine);
    if (auto && engine.backend === "webgpu") {
      const limit = o.gpuLimitMs ?? 300;
      if (warmupMs < limit * 10) warmupMs = await time(engine);
      if (warmupMs > limit) {
        console.info(`MotionBricks: WebGPU took ${Math.round(warmupMs)} ms a generation, using WebAssembly`);
        stop(engine);
        engine = await create(false);
        warmupMs = await time(engine);
      }
    }
    this.engine = new SharedEngine(engine);
    this._startController(warmupMs);
    return this;
  }
  /**
   * Share another character's model instead of loading one: several characters, one download and one worker. Their
   * generations take turns, so each waits a little longer as more share it. Both must use the same rig.
   */
  async loadFrom(other2) {
    if (this.controller) return this;
    if ((other2.options.rig ?? "g1") !== (this.options.rig ?? "g1")) throw new Error("MotionBricks: loadFrom needs a character with the same rig");
    await other2.load();
    this.engine = other2.engine;
    this._startController(other2.controller?.stats.averageInferenceMs ?? 400);
    return this;
  }
  _startController(warmupMs) {
    const engine = this.engine;
    const p = this._lastPose;
    const c = p ? new G1Controller(engine, this._bricks, ...this._rig.ground(p), this._rig.heading(p), this._rig) : new G1Controller(engine, this._bricks, 0, 0, 0, this._rig);
    c.gait = this.gait;
    c.stats.averageInferenceMs = warmupMs;
    c.onError = (e) => this.onError ? this.onError(e) : console.error("MotionBricks: generation failed", e);
    this.controller = c;
    this._syncObstacles();
    if (this._pendingGoal) c.moveTo(...this._pendingGoal, this.gait);
    this._pendingGoal = null;
  }
  /** Advance the motion by dt seconds and pose the robot. Call it every frame. */
  update(dt) {
    const c = this.controller;
    if (!c) return;
    dt = Math.min(dt, 0.1);
    if (this._task) this._updateTask();
    c.update(dt);
    let pose = c.pose;
    if (this.options.rig === "soma") {
      pose = this._carryPose(pose, dt);
      this._lastWorld.set(this._shownWorld);
      SOMA_SKELETON_INFO.world(pose, this._shownWorld);
      this._lastDt = dt;
      if (this._carry) this._placeCarried();
    }
    this.robot?.setPose(pose);
  }
  // ---- interactions (the human, rig: 'soma') ----
  /** The interaction under way, or null. */
  get interaction() {
    return this._task ? { kind: this._task.kind, stage: this._task.stage, item: this._task.item } : null;
  }
  /** The item being carried, or null. */
  get carrying() {
    return this._carry?.item ?? null;
  }
  /**
   * While a kick is under way, the way the ball should go (a world direction on the ground) and how hard (1 is a
   * firm pass): for your ball physics, which detects the foot meeting the ball with jointPosition.
   */
  get kicking() {
    const t = this._task;
    if (!t || t.kind !== "kick" || t.stage === "approach") return null;
    const direction = this._fromG1Direction(t.angle);
    return { direction, power: t.power, foot: t.foot };
  }
  /** A SOMA joint's position in world space as shown, such as 'RightToeBase', 'LeftFoot' or 'RightHand'. */
  jointPosition(name, target = new Vector3()) {
    const j = SOMA_SKELETON_INFO.index(name);
    if (this.options.rig !== "soma" || j < 0) return null;
    const w = this._shownWorld;
    return this.object.localToWorld(target.set(w[j * 7], w[j * 7 + 1], w[j * 7 + 2]));
  }
  /** A SOMA joint's velocity in world space, m/s, over the last frame. */
  jointVelocity(name, target = new Vector3()) {
    const j = SOMA_SKELETON_INFO.index(name);
    if (this.options.rig !== "soma" || j < 0) return null;
    const a = this._shownWorld, b = this._lastWorld, o = j * 7;
    const now = this.object.localToWorld(target.set(a[o], a[o + 1], a[o + 2]));
    const before = this.object.localToWorld(tempB.set(b[o], b[o + 1], b[o + 2]));
    return now.sub(before).divideScalar(this._lastDt);
  }
  /**
   * Kick a ball at a world point: walk up to it (lining up first when it is to the side) and swing through it.
   * `direction` is where it should go (by default straight on from the human); `power` 0.5 to 2.
   */
  kick(ball, { direction, power = 1, foot = "Right" } = {}) {
    if (!this._canInteract()) return false;
    const at = this._toG1(ball);
    const [x, y] = this._rig.ground(this.controller.pose);
    const angle = direction ? this._toG1Direction(direction) : Math.atan2(at[1] - y, at[0] - x);
    const speed = this.controller.speedOf(this._drive.active ? this._drive.gait : this.gait);
    this._startTask({ kind: "kick", stage: "approach", at, angle, trigger: Math.min(4, speed * 0.9 + 1), power, foot }, 1.6);
    return true;
  }
  /** Walk to an item, crouch and pick it up; it is then carried until thrown or put down. */
  pickUp(item) {
    if (!this._canInteract() || this._carry) return false;
    const at = this._toG1(item.position);
    const [x, y] = this._rig.ground(this.controller.pose);
    const angle = Math.atan2(at[1] - y, at[0] - x);
    this._startTask({ kind: "pick", stage: "approach", at, angle, trigger: 2, item, power: 1, foot: "Right" }, 0);
    return true;
  }
  /** Throw the carried item at a world point, turning to face it. onRelease gives the velocity that lands it there. */
  throwAt(target, { power = 1 } = {}) {
    if (!this._canInteract() || !this._carry) return false;
    const [x, y] = this._rig.ground(this.controller.pose);
    const g = this._toG1(target);
    const angle = Math.atan2(g[1] - y, g[0] - x);
    this._task = { kind: "throw", stage: "Throw", at: [x, y], angle, trigger: 0, target: target.clone(), power, foot: "Right" };
    this._drive.active = false;
    const th = throwPoses({ power });
    const ahead = 0.1;
    this.controller.perform([
      { name: "Throw", window: this._rig.place(th.poses, x + Math.cos(angle) * ahead, y + Math.sin(angle) * ahead, angle), transitionEarly: 2, tokens: range(6, 9) },
      { name: "Recover", window: this._rig.place(standPoses(), x + Math.cos(angle) * 0.35, y + Math.sin(angle) * 0.35, angle), transitionEarly: 0, tokens: range(6, 9) }
    ], { name: "throw" });
    return true;
  }
  /**
   * Put the carried item down at a world point (by default on the ground just ahead), crouching to it. A point above
   * the ground, such as the top of a stack, puts the item's centre there.
   */
  putDown(at) {
    if (!this._canInteract() || !this._carry) return false;
    const [x, y] = this._rig.ground(this.controller.pose);
    const h = this.heading;
    const g = at ? this._toG1(at) : [x + Math.cos(h) * 0.7, y + Math.sin(h) * 0.7];
    const angle = Math.hypot(g[0] - x, g[1] - y) > 0.3 ? Math.atan2(g[1] - y, g[0] - x) : h;
    const size = this._carry.item.size ?? 0.12;
    const height = at ? Math.max(size / 2 + 0.02, this.object.worldToLocal(tempA.copy(at)).y) : size / 2 + 0.02;
    this._startTask({ kind: "put", stage: "approach", at: g, angle, trigger: 2, item: this._carry.item, power: 1, foot: "Right", height }, 0);
    return true;
  }
  /** Stop the interaction under way, if it has not started its motion. */
  cancelInteraction() {
    if (this._task?.stage === "approach") this._task = null;
  }
  _canInteract() {
    if (this.options.rig !== "soma") {
      console.warn("MotionBricks: interactions are for the human model (rig: 'soma')");
      return false;
    }
    const c = this.controller;
    return !!c && !c.busy && !this.seated;
  }
  _startTask(task, lineUp) {
    const c = this.controller;
    const [x, y] = this._rig.ground(c.pose);
    const dx = task.at[0] - x, dy = task.at[1] - y;
    const off = Math.abs(Math.atan2(Math.sin(Math.atan2(dy, dx) - task.angle), Math.cos(Math.atan2(dy, dx) - task.angle)));
    if (lineUp && off > 0.6 && Math.hypot(dx, dy) > 0.5) task.via = [task.at[0] - Math.cos(task.angle) * lineUp, task.at[1] - Math.sin(task.angle) * lineUp];
    this._task = task;
    this._drive.active = false;
    if (task.via) c.moveTo(task.via[0], task.via[1], this.gait);
    else this._approach(task);
    this._updateTask();
  }
  /** Walk toward the task's spot, aiming a little short of it so that a late generation never steps onto it. */
  _approach(task) {
    const c = this.controller;
    const [x, y] = this._rig.ground(c.pose);
    const dx = task.at[0] - x, dy = task.at[1] - y, d = Math.hypot(dx, dy);
    if (d <= task.trigger) return;
    const short = Math.min(0.6, d);
    c.moveTo(task.at[0] - dx / d * short, task.at[1] - dy / d * short, this.gait);
  }
  _updateTask() {
    const t = this._task, c = this.controller;
    if (t.stage === "approach") {
      const [x, y] = this._rig.ground(c.pose);
      if (t.via) {
        if (Math.hypot(t.via[0] - x, t.via[1] - y) > 0.6) return;
        t.via = void 0;
        this._approach(t);
      }
      if (Math.hypot(t.at[0] - x, t.at[1] - y) > t.trigger || c.busy) return;
      this._act(t);
      return;
    }
    const a = c.action;
    if (a?.name === t.kind) t.stage = a.step;
    const p = c.playback;
    if (t.kind === "pick" && !this._carry && t.item) {
      const hand = this._handHold(t.item);
      if (hand.distanceTo(t.item.position) < (t.item.size ?? 0.12) * 0.5 + 0.08 || t.stage === "Lift" || a?.step === "Reach" && p && p.frame >= p.transition) {
        this._carry = { item: t.item, hands: this._hands(t.item), side: "Right", blend: 0 };
        this.onPickUp?.(t.item);
      }
    }
    if ((t.kind === "throw" || t.kind === "put") && this._carry && !t.done) {
      const release = t.kind === "throw" ? t.stage === "Recover" || a?.step === "Throw" && p && p.frame >= p.transition : t.stage === "Lift" || a?.step === "Reach" && p && p.frame >= p.transition;
      if (release) {
        t.done = true;
        const item = this._carry.item;
        this._carry = null;
        this.onRelease?.(item, t.kind === "throw" && t.target ? ballistic(item.position, t.target) : new Vector3());
      }
    }
    if (!c.busy && !a) this._task = null;
  }
  _act(t) {
    const c = this.controller, rig = this._rig;
    const [x, y] = rig.ground(c.pose);
    const back = (d) => [t.at[0] - Math.cos(t.angle) * d, t.at[1] - Math.sin(t.angle) * d];
    let steps;
    if (t.kind === "kick") {
      const [fx, fy] = [t.at[0] + Math.cos(t.angle) * 0.45, t.at[1] + Math.sin(t.angle) * 0.45];
      steps = [
        { name: "Kick", window: rig.place(kickPoses({ foot: t.foot, power: t.power }), t.at[0], t.at[1], t.angle), transitionEarly: 4, tokens: range(6, 12) },
        { name: "Recover", window: rig.place(standPoses(), fx, fy, t.angle), transitionEarly: 0, tokens: range(6, 10) }
      ];
    } else {
      if (Math.hypot(t.at[0] - x, t.at[1] - y) > 0.3) t.angle = Math.atan2(t.at[1] - y, t.at[0] - x);
      const item = t.item;
      const hands = this._hands(item), size = item.size ?? 0.12;
      const height = t.kind === "pick" ? Math.max(0.03, this.object.worldToLocal(tempA.copy(item.position)).y - (hands === "two" ? 0 : size * 0.3)) : (t.height ?? size / 2 + 0.02) - (hands === "two" ? 0 : size * 0.3);
      const [sx, sy] = back(0.3);
      steps = [
        { name: "Reach", window: rig.place(reachPoses({ height, hands, width: size }), t.at[0], t.at[1], t.angle), transitionEarly: 2, tokens: range(6, 12) },
        { name: "Lift", window: rig.place(t.kind === "pick" ? carryPoses({ hands, width: size }) : standPoses(), sx, sy, t.angle), transitionEarly: 0, tokens: range(6, 9) }
      ];
    }
    t.stage = steps[0].name;
    c.perform(steps, { name: t.kind });
  }
  _hands(item) {
    return item.hands ?? ((item.size ?? 0.12) > 0.22 ? "two" : "one");
  }
  /** Where the hand (or hands) hold an item, in world space. */
  _handHold(item, target = new Vector3()) {
    const w = this._shownWorld, k = SOMA_SKELETON_INFO;
    const at = (name) => tempB.set(w[k.index(name) * 7], w[k.index(name) * 7 + 1], w[k.index(name) * 7 + 2]);
    if (this._hands(item) === "two") target.copy(at("RightHand")).add(at("LeftHand")).multiplyScalar(0.5);
    else {
      const hand = at("RightHand").clone(), knuckle = at("RightHandMiddle1").clone();
      target.copy(hand).lerp(knuckle, 1.1);
      target.y -= (item.size ?? 0.12) * 0.35;
    }
    return this.object.localToWorld(target);
  }
  _placeCarried() {
    const item = this._carry.item;
    this._handHold(item, item.position);
    if (item.object) {
      const parent = item.object.parent;
      if (parent) parent.worldToLocal(item.object.position.copy(item.position));
      else item.object.position.copy(item.position);
    }
  }
  /** The controller's pose with the arms holding the carried item, blended in over a third of a second. */
  _carryPose(pose, dt) {
    const carry = this._carry;
    if (!carry) return pose;
    carry.blend = Math.min(1, carry.blend + dt * 3);
    const a = this.controller?.action;
    if (a && (a.name === "throw" || a.name === "put" || a.name === "pick" && a.step === "Reach")) return pose;
    const e = new SOMAPoseEditor(pose);
    const hips = e.position("Hips"), f = e.forward(), l = e.left();
    const at = carryHands(carry.hands, carry.item.size ?? 0.3, carry.side);
    const w = carry.blend * carry.blend * (3 - 2 * carry.blend);
    for (const side of ["Left", "Right"]) {
      const p = at[side];
      if (!p) continue;
      const target = v3.add([hips[0], hips[1] - 0.993, hips[2]], v3.add(v3.scale(f, p[0]), v3.add([0, p[1], 0], v3.scale(l, -p[2]))));
      const now = e.position(side + "Hand");
      e.arm(side, v3.lerp(now, target, w), v3.add(v3.scale(f, -0.3), [0, -1, 0]));
    }
    this._shown ??= new Float32Array(pose.length);
    this._shown.set(e.pose);
    return this._shown;
  }
  _toG1Direction(direction) {
    this.object.updateWorldMatrix(true, false);
    const o = this.object.worldToLocal(tempA.set(0, 0, 0)).clone();
    const d = this.object.worldToLocal(tempB.copy(direction)).sub(o);
    return Math.atan2(-d.z, d.x);
  }
  _fromG1Direction(angle) {
    this.object.updateWorldMatrix(true, false);
    const d = new Vector3(Math.cos(angle), 0, -Math.sin(angle));
    return d.transformDirection(this.object.matrixWorld);
  }
  /**
   * Load one of NVIDIA's Smart Primitives for this model: 'jump_0_5m' (a box), 'jump_0_75m' (a wall), 'avoidance'
   * (a bar to duck under) or 'sit' (a stool); primitive.box.size gives its size. Its clips download from NVIDIA's
   * repository.
   */
  async loadPrimitive(name) {
    return loadPrimitive(name, { rig: this.options.rig === "soma" ? "soma" : "g1", sources: this.options.primitives });
  }
  /** The obstacles the robot jumps or vaults over on its way, replacing any set before. */
  setObstacles(obstacles) {
    this._obstacles = obstacles;
    this._syncObstacles();
  }
  /** Sit on an obstacle whose primitive is 'sit' (one of setObstacles'), from within about two metres. */
  sit(obstacle) {
    const i = this._obstacles.indexOf(obstacle);
    const o = this.controller?.obstacles[i];
    if (i < 0 || !o) return;
    this._drive.active = false;
    this.controller.sit(o);
  }
  /** Stand up from the seat. Moving with moveTo or drive also stands up first. */
  standUp() {
    this.controller?.standUp();
  }
  /** The robot is sitting, or sitting down. */
  get seated() {
    return !!this.controller?.seat;
  }
  _syncObstacles() {
    const c = this.controller;
    if (!c) return;
    this.object.updateWorldMatrix(true, false);
    const turn = this.object.getWorldQuaternion(new Quaternion());
    const objectYaw = Math.atan2(2 * (turn.w * turn.y + turn.x * turn.z), 1 - 2 * (turn.y * turn.y + turn.z * turn.z));
    c.obstacles = this._obstacles.map((o) => {
      const [x, y] = this._toG1(o.position);
      return { primitive: o.primitive, x, y, yaw: (o.rotation ?? 0) - objectYaw };
    });
  }
  /** Convert a world point to the character's G1 ground coordinates. */
  _toG1(point) {
    this.object.updateWorldMatrix(true, false);
    return G1Robot.ToG1(this.object.worldToLocal(tempA.copy(point)));
  }
  /** Walk or run to a world point and stop there, facing the way it walked. */
  moveTo(point, gait = this.gait) {
    this.cancelInteraction();
    this.gait = gait;
    this._drive.active = false;
    const g = this._toG1(point);
    if (this.controller) this.controller.moveTo(g[0], g[1], gait);
    else this._pendingGoal = g;
  }
  /**
   * Steer with a world direction on the ground (WASD, a joystick), or null to stop. Call it every frame
   * while the input is held. The gait applies while driving; `gait` stays the default for moveTo. A new direction or gait generates at once; small changes steer the goal,
   * kept a few metres ahead, from the next segment.
   */
  drive(direction, gait = this.gait) {
    const c = this.controller;
    if (!c) return;
    if (direction && (direction.x !== 0 || direction.z !== 0)) this.cancelInteraction();
    if (!direction || direction.x === 0 && direction.z === 0) {
      if (this._drive.active) c.stop();
      this._drive.active = false;
      return;
    }
    this.object.updateWorldMatrix(true, false);
    const origin = this.object.worldToLocal(tempA.set(0, 0, 0));
    const [ox, oy] = G1Robot.ToG1(origin);
    const [dx, dy] = this._toG1(direction);
    const angle = Math.atan2(dy - oy, dx - ox);
    const ahead = c.speedOf(gait) * (2 * c.lead + 0.5) + 3;
    const goal = [c.pose[0] + Math.cos(angle) * ahead, c.pose[1] + Math.sin(angle) * ahead];
    const d = this._drive;
    const turn = Math.abs(Math.atan2(Math.sin(angle - d.angle), Math.cos(angle - d.angle)));
    if (!d.active || gait !== d.gait || turn > 0.26) {
      Object.assign(d, { active: true, angle, gait });
      c.moveTo(goal[0], goal[1], gait);
    } else c.goal = goal;
  }
  /** Come to a stop a short way ahead. */
  stop() {
    this._drive.active = false;
    this._pendingGoal = null;
    this.controller?.stop();
  }
  /** The robot's ground position, in world space. */
  getPosition(target = new Vector3()) {
    const pose = this.controller?.pose;
    if (pose) G1Robot.FromG1(...this._rig.ground(pose), 0, target);
    else target.set(0, 0, 0);
    return this.object.localToWorld(target);
  }
  /** The current goal, in world space, or null. */
  getGoal(target = new Vector3()) {
    const goal = this.controller?.goal ?? this._pendingGoal;
    if (!goal) return null;
    return this.object.localToWorld(G1Robot.FromG1(goal[0], goal[1], 0, target));
  }
  /** The direction the robot faces, as an angle around Y in the character's space (0 faces +X). */
  get heading() {
    const p = this.controller?.pose;
    return p ? this._rig.heading(p) : 0;
  }
  /** Stop the worker. The robot stays in the scene, and load() starts the model again from where it stands. */
  dispose() {
    if (this.controller) this._lastPose = Float32Array.from(this.controller.pose);
    const raw = this.engine instanceof SharedEngine ? this.engine.engine : this.engine;
    if (raw instanceof MotionBricksRemote) raw.worker.terminate();
    this.engine = void 0;
    this.controller = void 0;
  }
}
function isLowMemoryDevice() {
  if (typeof navigator === "undefined") return false;
  const memory = navigator.deviceMemory;
  if (memory && memory <= 4) return true;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) || navigator.platform.includes("Mac") && navigator.maxTouchPoints > 1;
}
async function loadOrt(url, wasm) {
  const ort = await import(
    /* @vite-ignore */
    new URL(url, location.href).href
  );
  await setupOrtWeb(ort, wasm);
  return ort;
}
export {
  BOUNDARY_FRAMES,
  CHARACTER_RIGS,
  DEFAULT_TOKEN_COUNTS,
  G1Controller,
  G1Robot,
  G1_BODY_TREE,
  G1_BRICKS,
  G1_JOINT_BODIES,
  G1_JOINT_NAMES,
  G1_PARTS,
  G1_PRIMITIVE_SOURCES,
  G1_RIG,
  GAIT_SPEEDS,
  MOTIONBRICKS_CACHE,
  MOTIONBRICKS_CHARACTERS,
  MOTIONBRICKS_DEFAULTS,
  MotionBricksCharacter,
  MotionBricksG1,
  MotionBricksRemote,
  MotionBricksSOMA,
  MotionBricksWorkerError,
  POSES_PER_TOKEN,
  POSE_SIZE,
  ROOT_SIZE,
  RetargetedCharacter,
  SAMPLE_RATE,
  SOMAMannequin,
  SOMAPoseEditor,
  SOMASkeletonInfo,
  SOMA_BRICKS,
  SOMA_MODEL_JOINTS,
  SOMA_PARTS,
  SOMA_PRIMITIVE_SOURCES,
  SOMA_RIG,
  SOMA_SKELETON,
  SOMA_SKELETON_INFO,
  SharedEngine,
  ballistic,
  carryHands,
  carryPoses,
  correctEndpoints$1 as correctEndpoints,
  dequantizePack,
  heading,
  isLowMemoryDevice,
  kickPoses,
  lerpPose,
  loadG1Primitive,
  loadOrt,
  loadPrimitive,
  parseSOMABvh,
  placePoses,
  qinv$2 as qinv,
  qmul$2 as qmul,
  qnorm$2 as qnorm,
  qpow,
  qrot$2 as qrot,
  qslerp,
  quat,
  qz,
  reachPoses,
  setupOrtWeb,
  standPoses,
  throwPoses,
  v3,
  withSpeed
};
//# sourceMappingURL=index.mjs.map
