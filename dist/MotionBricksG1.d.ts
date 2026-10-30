/** A G1 pose: root XYZ, root XYZW quaternion, then 29 joint angles. G1 space is Z up and +X forward. */
export type G1Pose = Float32Array;
export declare const G1_JOINT_NAMES: readonly ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"];
export declare const ROOT_SIZE = 7;
export declare const POSE_SIZE: number;
export declare const SAMPLE_RATE = 30;
export declare const POSES_PER_TOKEN = 4;
export declare const BOUNDARY_FRAMES: number;
export declare const DEFAULT_TOKEN_COUNTS: number[];
export type Quat = [number, number, number, number];
export type Vec3 = [number, number, number];
export declare function qmul(a: ArrayLike<number>, b: ArrayLike<number>): Quat;
export declare function qinv(q: ArrayLike<number>): Quat;
export declare function qnorm(q: ArrayLike<number>): Quat;
export declare function qrot(q: ArrayLike<number>, v: ArrayLike<number>): Vec3;
/** Rotation about +Z (up) by angle radians. */
export declare function qz(angle: number): Quat;
/** Slerp from the identity to q by t, along the shortest arc. */
export declare function qpow(q: ArrayLike<number>, t: number): Quat;
export declare function qslerp(a: ArrayLike<number>, b: ArrayLike<number>, t: number): Quat;
/** Ground heading of a root rotation: the angle of its +X axis around +Z. */
export declare function heading(q: ArrayLike<number>): number;
/** The ONNX Runtime API this engine uses, from onnxruntime-web or onnxruntime-node. */
export interface OrtApi {
    InferenceSession: {
        create(model: any, options?: any): Promise<any>;
    };
    Tensor: new (type: string, data: any, dims: number[]) => any;
}
export interface GenerateOptions {
    /** Token counts the model may choose, each 6 to 16. Each token is four frames. */
    tokens?: number[];
    /** Match the motion's ends to the last context pose and the first target pose. Default true. */
    correct?: boolean;
}
export interface GeneratedMotion {
    /** World space poses at 30 Hz. The first is near the last context pose, the last near the first target pose. */
    poses: G1Pose[];
    /** Left heel, left toe, right heel, right toe contact for each pose, 0 to 1. */
    footContacts: Float32Array[];
    tokenCount: number;
    inferenceMs: number;
}
/** What G1Controller needs from an engine: MotionBricksG1 here, or MotionBricksRemote in a worker. */
export interface MotionGenerator {
    readonly backend: string;
    generate(context: ArrayLike<number>[], target: ArrayLike<number>[], options?: GenerateOptions): Promise<GeneratedMotion>;
}
/**
 * Manifest written by tools/pack_lite.py: a graph whose weights stay 8-bit in memory (MatMulNBits and int8
 * convolutions), with its ONNX external data in shards.
 */
export interface LitePackManifest {
    format: 'motionbricks-external';
    graph: string;
    /** The external data file name the graph refers to. */
    weights: string;
    bytes: number;
    shardBytes: number;
    shards: string[];
}
/** Manifest written by tools/pack_model.py. */
export interface PackManifest {
    format: 'motionbricks-q8';
    /** File name of the graph, graph.onnx by default. */
    graph?: string;
    block: number;
    weights: string;
    f32Length: number;
    q8Length: number;
    /** Size of every shard but the last. */
    shardBytes: number;
    shards: string[];
    tensors: {
        name: string;
        n: number;
        blocks: number;
        q: number;
        s: number;
        f: number;
    }[];
}
export type ProgressCallback = (loaded: number, total: number) => void;
/**
 * Quiet onnxruntime-web's logs, use up to four threads when the page is cross-origin isolated, and give it
 * its .wasm file joined from `wasm` parts, for hosts that limit file size.
 */
export declare function setupOrtWeb(ort: any, wasm?: string[]): Promise<void>;
/** Expand int8 blocks with fp16 scales into the fp32 external data the graph expects. */
export declare function dequantizePack(manifest: PackManifest, q8: Uint8Array): Float32Array;
export declare class MotionBricksG1 implements MotionGenerator {
    readonly ort: OrtApi;
    readonly session: any;
    /** The execution provider the session was created with, such as webgpu or wasm. */
    readonly backend: string;
    private _busy;
    constructor(ort: OrtApi, session: any, backend?: string);
    /**
     * Load a model packed by tools/pack_model.py: a graph, a manifest, and int8 weight shards.
     * @param baseUrl folder that holds pack.json, ending with a slash
     */
    static LoadPacked(ort: OrtApi, baseUrl: string, sessionOptions?: any, onProgress?: ProgressCallback): Promise<MotionBricksG1>;
    /** A pack from tools/pack_lite.py: the weights go to the session as they are, with no fp32 copy. */
    private static _loadLite;
    /** Load a pack folder from tools/pack_model.py or tools/pack_lite.py (a URL ending in /), or an original .onnx file. */
    static Load(ort: OrtApi, model: string, sessionOptions?: any, onProgress?: ProgressCallback): Promise<MotionBricksG1>;
    /** Load an original MotionBricks G1 .onnx file, such as mmm_v6_g1_fp32.onnx. */
    static LoadOnnx(ort: OrtApi, url: string, sessionOptions?: any, onProgress?: ProgressCallback): Promise<MotionBricksG1>;
    /**
     * Create the session with one execution provider at a time, in the given order, so the backend
     * that runs is known and a WebGPU failure falls back to WebAssembly.
     */
    private static _create;
    get busy(): boolean;
    /**
     * Generate motion from four recent poses toward one to four target poses, all in world space.
     * The model picks the duration from the allowed token counts.
     */
    generate(context: ArrayLike<number>[], target: ArrayLike<number>[], options?: GenerateOptions): Promise<GeneratedMotion>;
}
/**
 * Pull the first frame onto the last context pose and the last frame onto the first target pose.
 * Root corrections span the motion. Joint angle corrections fade in and out over 0.3 s.
 */
export declare function correctEndpoints(poses: G1Pose[], start: ArrayLike<number>, end: ArrayLike<number>): void;
/** Interpolate two poses: positions and joint angles linearly, the root rotation by slerp. */
export declare function lerpPose(a: ArrayLike<number>, b: ArrayLike<number>, t: number, out?: G1Pose): G1Pose;
/**
 * Place poses authored at the ground origin facing +X at a ground position, turned to a heading.
 * Only the root moves, as in the reference runtime's place_target_dof.
 */
export declare function placePoses(poses: ArrayLike<number>[], x: number, y: number, angle: number): G1Pose[];
/** Scale the ground travel of target poses at 30 Hz to a mean speed in m/s. */
export declare function withSpeed(poses: ArrayLike<number>[], speed: number): G1Pose[];
//# sourceMappingURL=MotionBricksG1.d.ts.map