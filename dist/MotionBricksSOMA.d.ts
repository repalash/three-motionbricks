import { GenerateOptions, GeneratedMotion, MotionBricksG1, OrtApi, ProgressCallback } from './MotionBricksG1';
export type SOMAPose = Float32Array;
/** The model's 30 joints, in its order. The rest of the skeleton (fingers) keeps its reference pose. */
export declare const SOMA_MODEL_JOINTS: readonly ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"];
/** The runtime skeleton with lookups the engine and its users need. */
export declare class SOMASkeletonInfo {
    readonly names: string[];
    readonly parents: number[];
    readonly J: number;
    readonly reference: Float32Array;
    readonly modelToPose: number[];
    readonly poseToModel: Int32Array;
    readonly rightLeg: number;
    readonly leftLeg: number;
    constructor();
    index(name: string): number;
    /** A pose in the reference (bind) pose. */
    referencePose(): SOMAPose;
    /** World transforms of a pose, J * 7. */
    world(pose: ArrayLike<number>, out?: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer>;
    /** The pose's heading from its hips: an angle around Y, 0 facing +Z. */
    heading(pose: ArrayLike<number>): number;
}
export declare const SOMA_SKELETON_INFO: SOMASkeletonInfo;
/**
 * The SOMA model's engine. Load it with MotionBricksSOMA.Load (a pack folder from tools/pack_model.py or
 * pack_lite.py, or the original mmm_v6_soma_fp32.onnx), as MotionBricksG1.Load.
 */
export declare class MotionBricksSOMA extends MotionBricksG1 {
    readonly skeleton: SOMASkeletonInfo;
    static Load(ort: OrtApi, model: string, sessionOptions?: any, onProgress?: ProgressCallback, cacheName?: string | false): Promise<MotionBricksSOMA>;
    generate(context: ArrayLike<number>[], target: ArrayLike<number>[], options?: GenerateOptions): Promise<GeneratedMotion>;
}
/**
 * Read a BVH clip onto the SOMA skeleton at 30 Hz: joints by name, centimetres to metres, Euler angles in each
 * joint's channel order; a joint without position channels keeps the file's offset, and one missing from the file
 * keeps its reference transform. NVIDIA's sample clips are BVH (120 Hz). `rate` is the output rate (default 30).
 */
export declare function parseSOMABvh(text: string, skeleton?: SOMASkeletonInfo, rate?: number): SOMAPose[];
//# sourceMappingURL=MotionBricksSOMA.d.ts.map