import { G1Pose, MotionGenerator } from './MotionBricksG1';
export interface G1Bricks {
    /** Standing pose, one frame, at the ground origin facing +X. */
    rest: number[][];
    /** Four consecutive frames of each gait, starting at the ground origin, travelling along +X. */
    walk: number[][];
    run: number[][];
    /**
     * One steady stride of each gait, footfall to the same foot's next footfall, starting at the ground origin and
     * travelling along +X, with the ground travel over the whole stride. With these, gait targets are taken from the
     * point of the stride the robot will reach when it gets there, so no step is squeezed or stretched to fit.
     */
    cycles?: Partial<Record<G1Gait, G1StrideCycle>>;
}
/** One stride of a gait: its frames from a footfall, and the ground travel over the whole stride. */
export interface G1StrideCycle {
    frames: number[][];
    travel: number[];
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
     * Seconds of travel between the robot and its gait target. With stride cycles in the bricks, the target is
     * the pose the stride reaches by then and the segment length is fixed to match. Without them, the single gait
     * brick was baked with 1 s; other values make the model squeeze or stretch steps to fit.
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
     * Frames before a gait segment's end where the next one takes over. With stride cycles, a run hands over 6
     * frames early and a walk 2, so the hand-overs fall on alternate feet rather than always on the same step.
     * Without them, a run keeps its cadence with 10. Slow inference gets 2: more time to generate.
     */
    private _transitionEarly;
    private _distanceToGoal;
    /** The frame of the current segment where new motion can start, and the four context poses ending there. */
    private _contextAt;
    private _generate;
    private _switch;
}
//# sourceMappingURL=G1Controller.d.ts.map