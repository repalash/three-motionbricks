import { G1Pose } from './MotionBricksG1';
type Vec3 = [number, number, number];
type Quat = [number, number, number, number];
interface Xform {
    p: Vec3;
    q: Quat;
}
/** Where Smart Primitive documents and their clips load from: NVIDIA's MotionBricks repository on GitHub. */
export declare const G1_PRIMITIVE_SOURCES: {
    documents: string;
    /** Clips are in Git LFS, so they come from GitHub's media host; documents name them relative to this folder. */
    clips: string;
};
/** The same for the SOMA (human) model's primitives, whose clips are BVH. */
export declare const SOMA_PRIMITIVE_SOURCES: {
    documents: string;
    clips: string;
};
export type PrimitiveRig = 'g1' | 'soma';
export interface G1PrimitiveStep {
    name: string;
    /** Four poses in the pivot's space. */
    window: G1Pose[];
    /** Frames before the end of this step's motion where the next step takes over. */
    transitionEarly: number;
    /** Allowed token counts for the generation toward this step. */
    tokens: number[];
}
export interface G1PrimitivePivot {
    /** The pivot's authored transform in the primitive's space. */
    transform: Xform;
    /** Direction of travel through the primitive, in the pivot's space. */
    motion: Vec3;
    translationAxis: Vec3 | null;
    lineMin: number;
    lineMax: number;
    rotationAxis: Vec3 | null;
    /** Half angle of the allowed turn in radians, or Infinity when unconstrained. */
    coneHalfAngle: number;
    coneCenters: number[];
    rotationFirst: boolean;
}
export interface G1PrimitiveSequence {
    name: string;
    /** Speed of the first target, m/s: a sequence is picked by how close this is to the robot's speed. */
    startSpeed: number;
    steps: G1PrimitiveStep[];
    /** The pivot the steps are placed by: the sequence's own, or the nearest one above its targets. */
    pivot: G1PrimitivePivot;
    /** Node id of that pivot, or -1. */
    pivotId: number;
    /** Whether the sequence turns its pivot to the approach (its own pivot), or keeps its last placement. */
    aligns: boolean;
}
export interface G1Primitive {
    name: string;
    /** The model whose poses the targets are: 'g1' or 'soma'. */
    rig: PrimitiveRig;
    /**
     * 'pass' for primitives the robot goes through on its way (jumps, vaults, ducking), which start by themselves;
     * 'sit' for one with Sit and Stand sequences, which start on request (G1Controller.sit).
     */
    kind: 'pass' | 'sit';
    /** The interaction box, in the primitive's space: centre and width (x), depth (y), height (z). */
    box: {
        center: Vec3;
        size: Vec3;
    };
    sequences: G1PrimitiveSequence[];
}
/** One of NVIDIA's G1 Smart Primitives, such as 'jump_0_5m', 'jump_0_75m', 'sit' or 'avoidance'. */
export declare function loadG1Primitive(name: string, sources?: {
    documents: string;
    /** Clips are in Git LFS, so they come from GitHub's media host; documents name them relative to this folder. */
    clips: string;
}): Promise<G1Primitive>;
/**
 * Load one of NVIDIA's Smart Primitives for the G1 or SOMA, such as 'jump_0_5m', 'jump_0_75m', 'sit' or
 * 'avoidance', with the clip frames its targets use.
 */
export declare function loadPrimitive(name: string, { rig, sources }?: {
    rig?: PrimitiveRig;
    sources?: typeof G1_PRIMITIVE_SOURCES;
}): Promise<G1Primitive>;
/** An obstacle in G1 ground space: where its interaction box sits, and the yaw of the primitive's frame. */
export interface G1Obstacle {
    primitive: G1Primitive;
    x: number;
    y: number;
    yaw: number;
    /** Where each pivot was last aligned, so a following sequence (standing up after sitting) keeps it. */
    pivots?: Map<number, Xform>;
}
/** The near and far ends of the probe ahead of a robot at `p` moving along unit `d` at `speed`. */
export declare function probeSegment(p: Vec3, d: Vec3, speed: number): [Vec3, Vec3];
/** The obstacle's primitive frame in G1 space: its box centre on the obstacle, turned by its yaw. */
export declare function obstacleFrame(o: G1Obstacle): Xform;
/**
 * The sequence's steps placed in G1 space for a robot at (x, y) moving along `angle` at `speed`. A sequence with its
 * own pivot turns it to the approach; one without keeps the placement its pivot was last aligned to.
 */
export declare function placeSequence(o: G1Obstacle, sequence: G1PrimitiveSequence, x: number, y: number, angle: number, speed: number): {
    window: G1Pose[];
    name: string;
    /** Frames before the end of this step's motion where the next step takes over. */
    transitionEarly: number;
    /** Allowed token counts for the generation toward this step. */
    tokens: number[];
}[];
/**
 * Whether the probe of a robot at (x, y) moving along `angle` at `speed` meets the obstacle's interaction box,
 * looking `extra` metres further than the probe itself.
 */
export declare function probeHits(o: G1Obstacle, x: number, y: number, angle: number, speed: number, extra?: number): boolean;
export {};
//# sourceMappingURL=G1Primitives.d.ts.map