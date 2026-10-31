export interface PoseRig {
    readonly name: 'g1' | 'soma';
    readonly poseSize: number;
    /** The pose's position on the ground, in controller space. */
    ground(pose: ArrayLike<number>): [number, number];
    /** The way the pose faces, in controller space. */
    heading(pose: ArrayLike<number>): number;
    /** Poses authored at the origin facing +X, placed at (x, y) and turned by angle. */
    place(poses: ArrayLike<number>[], x: number, y: number, angle: number): Float32Array[];
    /** Poses with their ground travel scaled to a mean speed in m/s at 30 Hz. */
    withSpeed(poses: ArrayLike<number>[], speed: number): Float32Array[];
    /** Move a pose along the ground, in place. */
    translate(pose: Float32Array, dx: number, dy: number): void;
    /** Blend two poses. */
    lerp(a: ArrayLike<number>, b: ArrayLike<number>, t: number, out: Float32Array): Float32Array;
    /** How unlike two poses' bodies are, ignoring where they stand: for finding the point of a stride. */
    distance(a: ArrayLike<number>, b: ArrayLike<number>): number;
}
export declare const G1_RIG: PoseRig;
export declare const SOMA_RIG: PoseRig;
//# sourceMappingURL=PoseRig.d.ts.map