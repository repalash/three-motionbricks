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
function qmul(a, b) {
  const [ax, ay, az, aw] = [a[0], a[1], a[2], a[3]];
  const [bx, by, bz, bw] = [b[0], b[1], b[2], b[3]];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz
  ];
}
function qinv(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}
function qnorm(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qrot(q, v) {
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
  let [x, y, z, w] = qnorm(q);
  if (w < 0) [x, y, z, w] = [-x, -y, -z, -w];
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0, 0, 1];
  const angle = 2 * Math.atan2(s, w) * t;
  const k = Math.sin(angle / 2) / s;
  return [x * k, y * k, z * k, Math.cos(angle / 2)];
}
function heading(q) {
  const f = qrot(qnorm(q), [1, 0, 0]);
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
async function fetchBytes(url, onBytes, target, offset = 0) {
  const res = await fetch(url);
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
  if (target) return target.subarray(offset, offset + size);
  const out = new Uint8Array(size);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}
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
   */
  static async LoadPacked(ort, baseUrl, sessionOptions = {}, onProgress) {
    const manifest = await (await fetch(baseUrl + "pack.json")).json();
    if (manifest.format === "motionbricks-external") return MotionBricksG1._loadLite(ort, baseUrl, manifest, sessionOptions, onProgress);
    if (manifest.format !== "motionbricks-q8") throw new Error("Unknown model pack format");
    let loaded = 0;
    const total = manifest.q8Length;
    const count = (n) => {
      loaded += n;
      onProgress?.(loaded, total);
    };
    const q8 = new Uint8Array(manifest.q8Length);
    const [graph] = await Promise.all([
      fetchBytes(baseUrl + (manifest.graph ?? "graph.onnx")),
      ...manifest.shards.map(async (s, i) => fetchBytes(baseUrl + s, count, q8, i * manifest.shardBytes))
    ]);
    const weights = dequantizePack(manifest, q8);
    return MotionBricksG1._create(ort, graph, {
      ...sessionOptions,
      externalData: [{ path: manifest.weights, data: new Uint8Array(weights.buffer) }]
    });
  }
  /** A pack from tools/pack_lite.py: the weights go to the session as they are, with no fp32 copy. */
  static async _loadLite(ort, baseUrl, manifest, sessionOptions, onProgress) {
    let loaded = 0;
    const count = (n) => onProgress?.(loaded += n, manifest.bytes);
    const data = new Uint8Array(manifest.bytes);
    const [graph] = await Promise.all([
      fetchBytes(baseUrl + manifest.graph),
      ...manifest.shards.map(async (s, i) => fetchBytes(baseUrl + s, count, data, i * manifest.shardBytes))
    ]);
    return MotionBricksG1._create(ort, graph, {
      enableCpuMemArena: false,
      enableMemPattern: false,
      ...sessionOptions,
      externalData: [{ path: manifest.weights, data }]
    });
  }
  /** Load a pack folder from tools/pack_model.py or tools/pack_lite.py (a URL ending in /), or an original .onnx file. */
  static async Load(ort, model, sessionOptions = {}, onProgress) {
    return model.endsWith("/") ? MotionBricksG1.LoadPacked(ort, model, sessionOptions, onProgress) : MotionBricksG1.LoadOnnx(ort, model, sessionOptions, onProgress);
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
    const aqi = qinv(aq);
    const encode = (poses2) => {
      const out = new Float32Array(POSES_PER_TOKEN * POSE_SIZE);
      poses2.forEach((p, i) => {
        const o = i * POSE_SIZE;
        for (let k = 0; k < POSE_SIZE; k++) out[o + k] = p[k];
        const pos = qrot(aqi, [p[0] - ap[0], p[1] - ap[1], p[2] - ap[2]]);
        out[o] = pos[0];
        out[o + 1] = pos[1];
        out.set(qnorm(qmul(aqi, qnorm([p[3], p[4], p[5], p[6]]))), o + 3);
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
      const wpos = qrot(aq, pos);
      const pose = new Float32Array(POSE_SIZE);
      pose[0] = wpos[0] + ap[0];
      pose[1] = wpos[1] + ap[1];
      pose[2] = pos[2];
      pose.set(qnorm(qmul(aq, qnorm([predQpos[o + 3], predQpos[o + 4], predQpos[o + 5], predQpos[o + 6]]))), 3);
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
    q: qnorm(qmul(qinv(qnorm([p[3], p[4], p[5], p[6]])), qnorm([guide[3], guide[4], guide[5], guide[6]]))),
    a: Float32Array.from({ length: POSE_SIZE - ROOT_SIZE }, (_, i) => guide[ROOT_SIZE + i] - p[ROOT_SIZE + i])
  });
  const d0 = delta(poses[0], start);
  const d1 = delta(poses[n - 1], end);
  const fadeFrames = Math.min(JOINT_ANGLE_FADE_SECONDS * SAMPLE_RATE, 0.5 * (n - 1));
  for (let f = 0; f < n; f++) {
    const p = poses[f];
    const w = f / (n - 1);
    const sw = 1 - w;
    p.set(qnorm(qmul(qmul([p[3], p[4], p[5], p[6]], qpow(d0.q, sw)), qpow(d1.q, w))), 3);
    for (let k = 0; k < 3; k++) p[k] += d0.p[k] * sw + d1.p[k] * w;
    const ws = fadeWeight(f / fadeFrames);
    const we = fadeWeight((n - 1 - f) / fadeFrames);
    for (let i = ROOT_SIZE; i < POSE_SIZE; i++) p[i] += d0.a[i - ROOT_SIZE] * ws + d1.a[i - ROOT_SIZE] * we;
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
  engine = await MotionBricksG1.Load(ort, o.model, o.sessionOptions, progress);
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
