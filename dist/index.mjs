import { Quaternion, Vector3, Group, Euler, Object3D } from "three";
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
const MIN_TOKENS = 6;
const MAX_TOKENS = 16;
const MAX_RAW_FRAMES = MAX_TOKENS * POSES_PER_TOKEN;
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
function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bs = d < 0 ? -1 : 1;
  d *= bs;
  if (d > 0.9995) return qnorm$1([0, 1, 2, 3].map((i) => a[i] + (b[i] * bs - a[i]) * t));
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s * bs;
  return [0, 1, 2, 3].map((i) => a[i] * wa + b[i] * wb);
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
    const engine = manifest.format === "motionbricks-external" ? await MotionBricksG1._loadLite(ort, baseUrl, manifest, sessionOptions, onProgress, cache) : await MotionBricksG1._loadQ8(ort, baseUrl, manifest, sessionOptions, onProgress, cache);
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
    return MotionBricksG1._create(ort, graph, {
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
    return MotionBricksG1._create(ort, graph, {
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
    return model.endsWith("/") ? MotionBricksG1.LoadPacked(ort, model, sessionOptions, onProgress, cacheName) : MotionBricksG1.LoadOnnx(ort, model, sessionOptions, onProgress);
  }
  /** Load an original MotionBricks G1 .onnx file, such as mmm_v6_g1_fp32.onnx. */
  static async LoadOnnx(ort, url, sessionOptions = {}, onProgress) {
    let loaded = 0;
    const res = await fetch(url, { method: "HEAD" }).catch(() => null);
    const total = Number(res?.headers.get("content-length")) || 0;
    const bytes = await fetchBytes(url, (n) => onProgress?.(loaded += n, total));
    return MotionBricksG1._create(ort, bytes, sessionOptions);
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
        return new MotionBricksG1(ort, session, providers[i]);
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
    const allowed = new Int32Array(MAX_TOKENS - MIN_TOKENS + 1);
    for (const t of tokens) allowed[t - MIN_TOKENS] = 1;
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
    if (options.correct !== false && frameCount >= 2) correctEndpoints(poses, last, targets[0]);
    return { poses, footContacts, tokenCount, inferenceMs };
  }
}
function correctEndpoints(poses, start, end) {
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
function lerpPose(a, b, t, out = new Float32Array(POSE_SIZE)) {
  for (let i = 0; i < POSE_SIZE; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  out.set(qslerp([a[3], a[4], a[5], a[6]], [b[3], b[4], b[5], b[6]], t), 3);
  return out;
}
function placePoses(poses, x, y, angle) {
  const q = qz(angle);
  return poses.map((src) => {
    const p = Float32Array.from(src);
    const pos = qrot$1(q, [src[0], src[1], 0]);
    p[0] = pos[0] + x;
    p[1] = pos[1] + y;
    p.set(qnorm$1(qmul$1(q, [src[3], src[4], src[5], src[6]])), 3);
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
   * @param matchSize scale the model so its legs match the G1's (default true); otherwise its steps slide
   */
  constructor(model, rig, { shadows = true, matchSize = true } = {}) {
    this.object = new Group();
    this._g1 = g1Skeleton();
    this._parts = [];
    this._pelvisRest = new Vector3();
    this._g1PelvisRest = new Vector3();
    this._attach = [];
    this.model = model;
    this.rig = rig;
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
    const yaw = { "+x": 0, "-x": Math.PI, "+z": Math.PI / 2, "-z": -Math.PI / 2 }[rig.forward];
    holder.quaternion.setFromAxisAngle(new Vector3(0, 1, 0), yaw);
    this.object.updateMatrixWorld(true);
    const g1 = this._g1;
    g1.robot.setPose(referencePose());
    g1.root.updateMatrixWorld(true);
    const g1Part = (part) => g1.root.getObjectByName(G1_PARTS[part]);
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
    this._g1Pelvis = g1.root.getObjectByName("pelvis");
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
    const g1 = this._g1;
    g1.robot.setPose(pose);
    g1.root.updateMatrixWorld(true);
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
const NVIDIA_REPO = "https://raw.githubusercontent.com/NVlabs/motionbricks/main/assets/";
const NVIDIA_LFS = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/assets/";
const G1_PRIMITIVE_SOURCES = {
  documents: NVIDIA_REPO + "smart_primitives/unitree_g1/",
  /** Clips are in Git LFS, so they come from GitHub's media host; documents name them relative to this folder. */
  clips: NVIDIA_LFS + "smart_primitives/unitree_g1/"
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
async function loadClip(url) {
  let clip = clipCache.get(url);
  if (!clip) {
    clip = fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`Could not download ${url}: HTTP ${r.status}`);
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
function sampleClip(frames, seconds) {
  const f = Math.min(Math.max(seconds * 120, 0), frames.length - 1);
  const i = Math.min(Math.floor(f), frames.length - 2), t = f - i;
  const a = frames[i], b = frames[i + 1], out = new Float32Array(POSE_SIZE);
  for (let k = 0; k < POSE_SIZE; k++) out[k] = a[k] * (1 - t) + b[k] * t;
  let qa = [a[3], a[4], a[5], a[6]], qb = [b[3], b[4], b[5], b[6]];
  if (dot([qa[0], qa[1], qa[2]], [qb[0], qb[1], qb[2]]) + qa[3] * qb[3] < 0) qb = qb.map((v) => -v);
  qa = qnorm([0, 1, 2, 3].map((k) => qa[k] * (1 - t) + qb[k] * t));
  out.set(qa, 3);
  return out;
}
async function loadG1Primitive(name, sources = G1_PRIMITIVE_SOURCES) {
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
      const frames = await loadClip(new URL(c.clip, sources.clips).href);
      const start = c.playback_mode === "frames" ? c.frame_index / 120 : c.sample_time_seconds;
      const count = Math.min(Math.max(c.num_poses_to_use ?? POSES_PER_TOKEN, 1), POSES_PER_TOKEN);
      const toLocal = tmul(toPivot, world);
      const window = Array.from({ length: count }, (_, k) => placePose(toLocal, sampleClip(frames, start + k / SAMPLE_RATE)));
      const tokens = [];
      for (let t = c.min_tokens ?? 6; t <= (c.max_tokens ?? 14); t++) tokens.push(t);
      return { name: node.name, window, transitionEarly: c.transition_early_by_frames ?? 8, tokens };
    }));
    const w = steps[0].window;
    const startSpeed = w.length > 1 ? Math.hypot(w[1][0] - w[0][0], w[1][1] - w[0][1]) * SAMPLE_RATE : 0;
    return { name: s.name, startSpeed, steps, pivot, pivotId, aligns: own >= 0 };
  }));
  const names = sequences.map((q) => q.name);
  return { name, kind: names.includes("Sit") && names.includes("Stand") ? "sit" : "pass", box, sequences };
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
  return sequence.steps.map((step) => ({ ...step, window: step.window.map((pose) => placePose(pivot, pose)) }));
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
  constructor(engine, bricks, x = 0, y = 0, angle = 0) {
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
    this._segmentPose = new Float32Array(POSE_SIZE);
    this._fadePose = new Float32Array(POSE_SIZE);
    this.engine = engine;
    this.bricks = bricks;
    this.pose = placePoses(bricks.rest, x, y, angle)[0];
  }
  get moving() {
    return this._segment !== null;
  }
  get generating() {
    return this._request !== null;
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
    const angle = Math.atan2(o.y - this.pose[1], o.x - this.pose[0]);
    const steps = placeSequence(o, sequence, this.pose[0], this.pose[1], angle, GAIT_SPEEDS.walk);
    this.goal = null;
    this._action = { steps, next: 0, hold: true, seat: o };
    if (!this._request && !this._pending) this._generate(true);
  }
  /** Stand up from the seat and stay there. */
  standUp() {
    const o = this.seat;
    const sequence = o?.primitive.sequences.find((q) => q.name === "Stand");
    if (!o || !sequence) return;
    this.goal = null;
    this._action = { steps: placeSequence(o, sequence, this.pose[0], this.pose[1], 0, 0), next: 0, hold: true };
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
    if (seq && seq.next < seq.steps.length) return;
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
    samplePoses(current.poses, this._frame, this._segmentPose);
    const fade = this._fade;
    if (fade) {
      const w = Math.min(fade.t / FADE_FRAMES, 1);
      samplePoses(fade.segment.poses, fade.frame, this._fadePose);
      lerpPose(this._fadePose, this._segmentPose, w * w * (3 - 2 * w), this.pose);
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
    const ahead = (p) => (p[0] - from[0]) * cos + (p[1] - from[1]) * sin;
    for (const o of this.obstacles) {
      if (o.primitive.kind !== "pass" || !probeHits(o, from[0], from[1], angle, speed, speed * this.lead + 3)) continue;
      const sequence = o.primitive.sequences.reduce((a, b) => Math.abs(b.startSpeed - speed) < Math.abs(a.startSpeed - speed) ? b : a);
      const steps = placeSequence(o, sequence, from[0], from[1], angle, speed);
      const entry = ahead(steps[0].window[0]);
      if (entry > speed * this.lead + 0.3 || entry >= bestAhead) continue;
      const first = steps.findIndex((s) => ahead(s.window[0]) > 0.3);
      if (first < 0 || ahead(steps[steps.length - 1].window[0]) > distance + 0.5) continue;
      best = { steps, next: first };
      bestAhead = entry;
    }
    return best;
  }
  _distanceToGoal() {
    return this.goal ? Math.hypot(this.goal[0] - this.pose[0], this.goal[1] - this.pose[1]) : 0;
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
    const speed = GAIT_SPEEDS[this.gait];
    const facing = heading([from[3], from[4], from[5], from[6]]);
    const stopAhead = this._segment ? 0.5 * speed : 0;
    const goal = this.goal ?? [from[0] + Math.cos(facing) * stopAhead, from[1] + Math.sin(facing) * stopAhead];
    const dx = goal[0] - from[0], dy = goal[1] - from[1];
    const distance = Math.hypot(dx, dy);
    const angle = distance > 0.3 ? Math.atan2(dy, dx) : facing;
    let targets;
    let stopping;
    let options;
    let sequence;
    const lead = this.lead;
    const cycle = this.bricks.cycles?.[this.gait];
    const playing = this._segment?.sequence;
    let jump = playing && playing.next < playing.steps.length ? playing : void 0;
    if (!jump && this._action) {
      jump = this._action;
      this._action = null;
    }
    if (!jump && this.seat && this.goal) {
      const stand = this.seat.primitive.sequences.find((q) => q.name === "Stand");
      if (stand) jump = { steps: placeSequence(this.seat, stand, from[0], from[1], 0, 0), next: 0 };
    }
    if (!jump && this._segment && !this._segment.stopping && distance > 1) jump = this._findJump(from, angle, speed, distance);
    if (jump) {
      const step = jump.steps[jump.next];
      targets = step.window;
      options = { tokens: step.tokens };
      sequence = { ...jump, next: jump.next + 1 };
      stopping = !!jump.hold && jump.next === jump.steps.length - 1;
    } else if (distance > speed * lead + 0.8 && cycle && this._segment && !this._segment.stopping) {
      const tokens = Math.round((lead * SAMPLE_RATE + POSES_PER_TOKEN + BOUNDARY_FRAMES) / POSES_PER_TOKEN);
      const frames = Math.round(segmentSeconds(tokens) * SAMPLE_RATE);
      const ahead = speed * frames / SAMPLE_RATE;
      const window = cycleWindow(cycle, cyclePhase(cycle, context) + frames);
      targets = placePoses(withSpeed(window, speed), from[0] + Math.cos(angle) * ahead, from[1] + Math.sin(angle) * ahead, angle);
      options = { tokens: [tokens] };
      stopping = false;
    } else if (distance > speed * lead + 0.8) {
      const ahead = speed * lead;
      targets = placePoses(withSpeed(this.bricks[this.gait], speed), from[0] + Math.cos(angle) * ahead, from[1] + Math.sin(angle) * ahead, angle);
      stopping = false;
    } else {
      if (!this._segment && distance < 0.35) {
        this.goal = null;
        this._dirty = false;
        return;
      }
      targets = placePoses(this.bricks.rest, goal[0], goal[1], angle);
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
function cyclePhase(cycle, context) {
  const frames = cycle.frames, n = frames.length;
  let best = 0, bestError = Infinity;
  for (let i = 0; i < n; i++) {
    let error = 0;
    for (let c = 0; c < 2; c++) {
      const a = context[2 + c], b = frames[(i - 1 + c + n) % n];
      for (let k = POSE_SIZE - 29; k < POSE_SIZE; k++) error += (a[k] - b[k]) ** 2;
    }
    if (error < bestError) [best, bestError] = [i, error];
  }
  return best;
}
function cycleWindow(cycle, start) {
  const n = cycle.frames.length;
  const window = [0, 1, 2, 3].map((f) => {
    const i = start + f, stride = Math.floor(i / n);
    const p = cycle.frames[(i % n + n) % n].slice();
    p[0] += stride * cycle.travel[0];
    p[1] += stride * cycle.travel[1];
    return p;
  });
  const [x, y] = window[0];
  for (const p of window) {
    p[0] -= x;
    p[1] -= y;
  }
  return window;
}
function samplePoses(poses, frame, out) {
  const last = poses.length - 1;
  const f = Math.min(Math.max(frame, 0), last);
  const i = Math.floor(f);
  return lerpPose(poses[i], poses[Math.min(i + 1, last)], f - i, out);
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
const NVIDIA_ASSETS = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets";
const ASSETS = "https://cdn.jsdelivr.net/gh/repalash/three-motionbricks@v0.2.0/assets/";
const MOTIONBRICKS_DEFAULTS = {
  /** NVIDIA's original fp32 model, 730 MB. A pack from tools/pack_model.py is about 190 MB. */
  model: NVIDIA_ASSETS + "/onnx/unitree_g1/mmm_v6_g1_fp32.onnx",
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
const tempA = new Vector3();
class MotionBricksCharacter {
  constructor(options = {}) {
    this.object = new Group();
    this.gait = "walk";
    this._pendingGoal = null;
    this._drive = { active: false, angle: 0, gait: "walk" };
    this._obstacles = [];
    this.options = options;
    this.object.name = "MotionBricks G1";
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
    this.robot.setPose(this.controller?.pose ?? (this.options.bricks ?? G1_BRICKS).rest[0]);
    this.object.add(root);
    return this.robot;
  }
  /**
   * Use another model, such as a robot from assets/robots or a character with a Mixamo skeleton: the G1's motion
   * is retargeted to it with a rig from CHARACTER_RIGS (or your own). It replaces the current robot.
   */
  setCharacter(root, rig, options = {}) {
    if (this.robot) this.object.remove(this.robot.object);
    const character = new RetargetedCharacter(root, rig, { shadows: this.options.shadows ?? true, ...options });
    this.robot = character;
    character.setPose(this.controller?.pose ?? (this.options.bricks ?? G1_BRICKS).rest[0]);
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
    const model = useLite ? o.liteModel : o.model ?? MOTIONBRICKS_DEFAULTS.model;
    const useGpu = o.backend === "webgpu" && typeof navigator !== "undefined" && "gpu" in navigator;
    const ortUrl = o.ort ?? (useGpu ? MOTIONBRICKS_DEFAULTS.ortWebGpu : MOTIONBRICKS_DEFAULTS.ort);
    const cache = o.cache === false ? false : typeof o.cache === "string" ? o.cache : MOTIONBRICKS_CACHE;
    const sessionOptions = { executionProviders: useGpu ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "all" };
    const progress = (l, t) => onProgress?.("model", l, t);
    onProgress?.("runtime", 0, 1);
    let engine = null;
    if (o.worker !== false) {
      engine = await MotionBricksRemote.Load({ ort: ortUrl, wasm: o.wasm, model, sessionOptions, cache }, progress).catch((e) => {
        if (!(e instanceof MotionBricksWorkerError)) throw e;
        console.warn("MotionBricks: the worker could not start, running the model on the page instead", e);
        return null;
      });
    }
    if (!engine) engine = await MotionBricksG1.Load(await loadOrt(ortUrl, o.wasm), model, sessionOptions, progress, cache);
    onProgress?.("warmup", 0, 1);
    if (useLite) await new Promise((resolve) => setTimeout(resolve, 2e3));
    const rest = Float32Array.from((o.bricks ?? G1_BRICKS).rest[0]);
    let warmupMs = 0;
    try {
      warmupMs = (await engine.generate([rest, rest, rest, rest], [rest])).inferenceMs;
    } catch (e) {
      if (engine instanceof MotionBricksRemote) engine.worker.terminate();
      throw e;
    }
    this.engine = engine;
    const p = this._lastPose;
    const c = p ? new G1Controller(engine, o.bricks ?? G1_BRICKS, p[0], p[1], heading([p[3], p[4], p[5], p[6]])) : new G1Controller(engine, o.bricks ?? G1_BRICKS);
    c.gait = this.gait;
    c.stats.averageInferenceMs = warmupMs;
    c.onError = (e) => this.onError ? this.onError(e) : console.error("MotionBricks: generation failed", e);
    this.controller = c;
    this._syncObstacles();
    if (this._pendingGoal) c.moveTo(...this._pendingGoal, this.gait);
    this._pendingGoal = null;
    return this;
  }
  /** Advance the motion by dt seconds and pose the robot. Call it every frame. */
  update(dt) {
    const c = this.controller;
    if (!c) return;
    c.update(Math.min(dt, 0.1));
    this.robot?.setPose(c.pose);
  }
  /**
   * Load one of NVIDIA's G1 Smart Primitives: 'jump_0_5m' (a box 0.55 m wide and 0.39 m high), 'jump_0_75m' (a
   * wall 1.57 m long and 0.64 m high), and others. Its clips download from NVIDIA's repository.
   */
  async loadPrimitive(name) {
    return loadG1Primitive(name, this.options.primitives);
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
    const ahead = GAIT_SPEEDS[gait] * c.lead + 4;
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
    if (pose) G1Robot.FromG1(pose[0], pose[1], 0, target);
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
    return p ? heading([p[3], p[4], p[5], p[6]]) : 0;
  }
  /** Stop the worker. The robot stays in the scene, and load() starts the model again from where it stands. */
  dispose() {
    if (this.controller) this._lastPose = Float32Array.from(this.controller.pose);
    if (this.engine instanceof MotionBricksRemote) this.engine.worker.terminate();
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
  GAIT_SPEEDS,
  MOTIONBRICKS_CACHE,
  MOTIONBRICKS_CHARACTERS,
  MOTIONBRICKS_DEFAULTS,
  MotionBricksCharacter,
  MotionBricksG1,
  MotionBricksRemote,
  MotionBricksWorkerError,
  POSES_PER_TOKEN,
  POSE_SIZE,
  ROOT_SIZE,
  RetargetedCharacter,
  SAMPLE_RATE,
  correctEndpoints,
  dequantizePack,
  heading,
  isLowMemoryDevice,
  lerpPose,
  loadG1Primitive,
  loadOrt,
  placePoses,
  qinv$1 as qinv,
  qmul$1 as qmul,
  qnorm$1 as qnorm,
  qpow,
  qrot$1 as qrot,
  qslerp,
  qz,
  setupOrtWeb,
  withSpeed
};
//# sourceMappingURL=index.mjs.map
