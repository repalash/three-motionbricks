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
function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bs = d < 0 ? -1 : 1;
  d *= bs;
  if (d > 0.9995) return qnorm([0, 1, 2, 3].map((i) => a[i] + (b[i] * bs - a[i]) * t));
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s * bs;
  return [0, 1, 2, 3].map((i) => a[i] * wa + b[i] * wb);
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
function lerpPose(a, b, t, out = new Float32Array(POSE_SIZE)) {
  for (let i = 0; i < POSE_SIZE; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  out.set(qslerp([a[3], a[4], a[5], a[6]], [b[3], b[4], b[5], b[6]], t), 3);
  return out;
}
function placePoses(poses, x, y, angle) {
  const q = qz(angle);
  return poses.map((src) => {
    const p = Float32Array.from(src);
    const pos = qrot(q, [src[0], src[1], 0]);
    p[0] = pos[0] + x;
    p[1] = pos[1] + y;
    p.set(qnorm(qmul(q, [src[3], src[4], src[5], src[6]])), 3);
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
const FADE_FRAMES = 8;
const GAIT_SPEEDS = { walk: 1.13, run: 3.43 };
class G1Controller {
  constructor(engine, bricks, x = 0, y = 0, angle = 0) {
    this.goal = null;
    this.gait = "walk";
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
   * Frames before a gait segment's end where the next one takes over. A run keeps its natural cadence with 10, a
   * walk with 2. Slow inference also gets 2 when running: more time to generate, for a slightly quicker cadence.
   */
  _transitionEarly(gait) {
    return gait === "run" && Math.max(this.stats.averageInferenceMs, this.stats.lastInferenceMs) < 600 ? 10 : 2;
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
    const lead = this.lead;
    if (distance > speed * lead + 0.8) {
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
    }
    const request = { id: ++this._requestId, segment: this._segment, startFrame, stopping, gait: this.gait };
    this._request = request;
    this._dirty = false;
    this.engine.generate(context, targets).then((motion) => {
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
      transitionEarly: request.stopping ? 0 : this._transitionEarly(request.gait),
      stopping: request.stopping
    };
  }
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
  ]
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
  ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs"
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
    const ortUrl = o.ort ?? MOTIONBRICKS_DEFAULTS.ort;
    const useLite = !!o.liteModel && (o.lite === true || (o.lite ?? "auto") === "auto" && isLowMemoryDevice());
    const model = useLite ? o.liteModel : o.model ?? MOTIONBRICKS_DEFAULTS.model;
    const useGpu = o.backend === "webgpu" && typeof navigator !== "undefined" && "gpu" in navigator;
    const sessionOptions = { executionProviders: useGpu ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "all" };
    const progress = (l, t) => onProgress?.("model", l, t);
    onProgress?.("runtime", 0, 1);
    let engine = null;
    if (o.worker !== false) {
      engine = await MotionBricksRemote.Load({ ort: ortUrl, wasm: o.wasm, model, sessionOptions }, progress).catch((e) => {
        if (!(e instanceof MotionBricksWorkerError)) throw e;
        console.warn("MotionBricks: the worker could not start, running the model on the page instead", e);
        return null;
      });
    }
    if (!engine) engine = await MotionBricksG1.Load(await loadOrt(ortUrl, o.wasm), model, sessionOptions, progress);
    onProgress?.("warmup", 0, 1);
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
    if (!d.active || gait !== d.gait || turn > 0.35) {
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
  CHARACTER_RIGS,
  DEFAULT_TOKEN_COUNTS,
  G1Controller,
  G1Robot,
  G1_BODY_TREE,
  G1_BRICKS,
  G1_JOINT_BODIES,
  G1_JOINT_NAMES,
  G1_PARTS,
  GAIT_SPEEDS,
  MOTIONBRICKS_CHARACTERS,
  MOTIONBRICKS_DEFAULTS,
  MotionBricksCharacter,
  MotionBricksG1,
  MotionBricksRemote,
  MotionBricksWorkerError,
  POSE_SIZE,
  ROOT_SIZE,
  RetargetedCharacter,
  SAMPLE_RATE,
  correctEndpoints,
  dequantizePack,
  heading,
  isLowMemoryDevice,
  lerpPose,
  loadOrt,
  placePoses,
  qinv,
  qmul,
  qnorm,
  qpow,
  qrot,
  qslerp,
  qz,
  setupOrtWeb,
  withSpeed
};
//# sourceMappingURL=index.mjs.map
