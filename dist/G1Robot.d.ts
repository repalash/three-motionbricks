import { Material, Object3D, Quaternion, Vector3 } from 'three';
interface RobotJoint {
    node: Object3D;
    rest: Quaternion;
    axis: Vector3;
}
/**
 * The body each of the 29 G1 joints turns, and its axis, in G1_JOINT_NAMES order (from Unitree's MJCF).
 * A robot model needs nodes with these names, in MuJoCo's Z-up space under a `pelvis` node.
 */
export declare const G1_JOINT_BODIES: readonly [string, [number, number, number]][];
/**
 * Poses a Unitree G1 model that you loaded, such as `gltf.scene` of the unitree_g1.glb that ships with
 * this package. The model keeps MuJoCo's Z-up bodies under a node turned to three.js's Y-up.
 */
export declare class G1Robot {
    /** The model's root, as given. */
    readonly object: Object3D;
    readonly pelvis: Object3D;
    readonly joints: RobotJoint[];
    /** Every material in the model, such as the joint housings and the shells. */
    readonly materials: Material[];
    private constructor();
    /**
     * Wrap a loaded G1 model. Its meshes cast and receive shadows unless `shadows` is false.
     */
    static FromObject(root: Object3D, { shadows }?: {
        shadows?: boolean | undefined;
    }): G1Robot;
    /** Show a G1 pose: root XYZ, root XYZW quaternion, then the 29 joint angles. */
    setPose(pose: ArrayLike<number>): void;
    /** Convert a ground point from three.js space (Y up) to G1 space (Z up). */
    static ToG1(p: {
        x: number;
        z: number;
    }): [number, number];
    /** Convert a G1 space point to three.js space. */
    static FromG1(x: number, y: number, z?: number, target?: Vector3): Vector3;
}
export {};
//# sourceMappingURL=G1Robot.d.ts.map