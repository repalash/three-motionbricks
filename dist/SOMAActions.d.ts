import { V3 } from './SOMAPoseEdit';
export type Side = 'Left' | 'Right';
/**
 * A kick: three frames about the moment the foot meets a ball at the origin (radius `radius`), the leg swinging
 * through it along +X, the other foot planted beside the ball.
 */
export declare function kickPoses({ foot, radius, power }?: {
    foot?: Side | undefined;
    radius?: number | undefined;
    power?: number | undefined;
}): Float32Array[];
/** Where a carried load sits relative to the body when standing: before the waist, in one hand or two. */
export declare function carryHands(hands: 'one' | 'two', width?: number, side?: Side): {
    Left?: V3;
    Right?: V3;
};
/**
 * Reaching down to something at (0, height, 0): two frames of a squat with the hand (or hands, `width` apart) on it,
 * the body behind it facing +X.
 */
export declare function reachPoses({ height, hands, width, side }?: {
    height?: number | undefined;
    hands?: "one" | "two" | undefined;
    width?: number | undefined;
    side?: Side | undefined;
}): Float32Array[];
export declare namespace reachPoses {
    var lastDepth: number;
}
/** Standing with a load, two frames: the hands where carryHands puts them. */
export declare function carryPoses({ hands, width, side }?: {
    hands?: "one" | "two" | undefined;
    width?: number | undefined;
    side?: Side | undefined;
}): Float32Array[];
/**
 * An overhand throw along +X: three frames about the release, the throwing hand high and moving forward, the other
 * foot stepped forward. Returns the poses and the hand's position at release (the middle frame).
 */
export declare function throwPoses({ side, power }?: {
    side?: Side | undefined;
    power?: number | undefined;
}): {
    poses: Float32Array[];
    release: V3;
};
/** Standing still, two frames, for the end of an action. */
export declare function standPoses(): Float32Array[];
//# sourceMappingURL=SOMAActions.d.ts.map