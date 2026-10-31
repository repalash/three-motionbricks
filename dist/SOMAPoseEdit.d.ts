export type V3 = [number, number, number];
export type Q = [number, number, number, number];
export declare const v3: {
    sub: (a: ArrayLike<number>, b: ArrayLike<number>) => V3;
    add: (a: ArrayLike<number>, b: ArrayLike<number>) => V3;
    scale: (a: ArrayLike<number>, s: number) => V3;
    dot: (a: ArrayLike<number>, b: ArrayLike<number>) => number;
    cross: (a: ArrayLike<number>, b: ArrayLike<number>) => V3;
    len: (a: ArrayLike<number>) => number;
    norm: (a: ArrayLike<number>) => V3;
    lerp: (a: ArrayLike<number>, b: ArrayLike<number>, t: number) => V3;
};
export declare const quat: {
    mul(a: ArrayLike<number>, b: ArrayLike<number>): Q;
    inv: (q: ArrayLike<number>) => Q;
    norm(q: ArrayLike<number>): Q;
    axisAngle(axis: ArrayLike<number>, angle: number): Q;
    /** The shortest turn taking direction a onto direction b. */
    between(a: ArrayLike<number>, b: ArrayLike<number>): Q;
    rotate(q: ArrayLike<number>, v: ArrayLike<number>): V3;
    slerp(a: ArrayLike<number>, b: ArrayLike<number>, t: number): Q;
};
/** A SOMA pose being edited. Joints are named as in SOMA_SKELETON ('RightLeg' is the thigh, 'RightShin' the knee). */
export declare class SOMAPoseEditor {
    readonly pose: Float32Array;
    private _world;
    private _dirty;
    constructor(pose: ArrayLike<number>);
    get world(): Float32Array<ArrayBuffer>;
    private _j;
    position(joint: string | number): V3;
    rotation(joint: string | number): Q;
    /** Turn a joint, and all below it, by a world-space rotation about the joint. */
    turn(joint: string | number, q: ArrayLike<number>): this;
    /** Turn a joint by an angle about a world axis. */
    turnAbout(joint: string | number, axis: ArrayLike<number>, angle: number): this;
    /** Set a joint's world rotation. */
    setRotation(joint: string | number, q: ArrayLike<number>): this;
    /** Move the whole body (the root) in world space. */
    move(d: ArrayLike<number>): this;
    /**
     * Two-bone IK: turn `upper` and `mid` so that `end` reaches `target`, with `mid` (the knee or elbow) bending
     * toward `pole`, a world direction. The end joint keeps its world rotation unless `keepEnd` is false.
     */
    reach(upper: string, mid: string, end: string, target: ArrayLike<number>, pole: ArrayLike<number>, keepEnd?: boolean): this;
    /** Reach with a leg, the knee toward `pole` (by default the way the hips face). */
    leg(side: 'Left' | 'Right', target: ArrayLike<number>, pole?: ArrayLike<number>): this;
    /** Reach with an arm, the elbow toward `pole` (by default down and back). */
    arm(side: 'Left' | 'Right', target: ArrayLike<number>, pole?: ArrayLike<number>, keepHand?: boolean): this;
    /** The way the hips face, on the ground. */
    forward(): V3;
    /** The body's left, on the ground. */
    left(): V3;
}
//# sourceMappingURL=SOMAPoseEdit.d.ts.map