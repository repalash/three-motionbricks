import { G1Pose, MotionGenerator } from './MotionBricksG1';
export interface G1Bricks {
    /** Standing pose, one frame, at the ground origin facing +X. */
    rest: number[][];
    /** Four consecutive frames of each gait, starting at the ground origin, travelling along +X. */
    walk: number[][];
    run: number[][];
}
export type G1Gait = 'walk' | 'run';
export declare const GAIT_SPEEDS: Record<G1Gait, number>;
export interface G1ControllerStats {
    generations: number;
    lastInferenceMs: number;
    averageInferenceMs: number;
    lastTokenCount: number;
}
export declare class G1Controller {
    readonly engine: MotionGenerator;
    readonly bricks: G1Bricks;
    /** The pose to show this frame, in G1 space. */
    readonly pose: G1Pose;
    goal: [number, number] | null;
    gait: G1Gait;
    /**
     * Seconds of travel between the robot and its gait target. The gait bricks were baked with 1 s, so the
     * stride's phase at the target matches it; other values make the model squeeze or stretch steps to fit.
     */
    lead: number;
    stats: G1ControllerStats;
    onGenerated?: (motion: G1Pose[], targets: G1Pose[]) => void;
    onError?: (e: unknown) => void;
    private _segment;
    private _frame;
    private _request;
    private _requestId;
    private _dirty;
    private _pending;
    /** Seconds to wait before generating on its own again, after a failed generation. */
    private _retryIn;
    /**
     * The segment being replaced, still playing while the new one fades in. Motion that arrives late starts
     * a few frames in, and the fade hides the difference from the frame shown.
     */
    private _fade;
    private _segmentPose;
    private _fadePose;
    constructor(engine: MotionGenerator, bricks: G1Bricks, x?: number, y?: number, angle?: number);
    get moving(): boolean;
    get generating(): boolean;
    /** The segment being played: its length and transition point in frames, and the playhead. */
    get playback(): {
        frames: number;
        frame: number;
        transition: number;
        stopping: boolean;
    } | null;
    /** Walk or run to a ground point in G1 space, then stop there. */
    moveTo(x: number, y: number, gait?: G1Gait): void;
    /** Come to a stop a short way ahead. */
    stop(): void;
    /**
     * Generate for the new goal now, or once the motion in flight is in place: a request's start frame
     * belongs to the segment it was made from, so none is made while another waits to take over.
     */
    private _regenerate;
    /** Advance playback by dt seconds and update pose. */
    update(dt: number): void;
    private _latencyFrames;
    /**
     * Frames before a gait segment's end where the next one takes over. A run keeps its natural cadence with 10, a
     * walk with 2. Slow inference also gets 2 when running: more time to generate, for a slightly quicker cadence.
     */
    private _transitionEarly;
    private _distanceToGoal;
    /** The frame of the current segment where new motion can start, and the four context poses ending there. */
    private _contextAt;
    private _generate;
    private _switch;
}
//# sourceMappingURL=G1Controller.d.ts.map