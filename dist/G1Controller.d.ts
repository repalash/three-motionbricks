import { G1Obstacle, G1PrimitiveStep } from './G1Primitives';
import { G1Pose, MotionGenerator } from './MotionBricksG1';
import { PoseRig } from './PoseRig';
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
    /** Walk and run speeds in m/s, when not GAIT_SPEEDS (another skeleton's clips). */
    speeds?: Record<G1Gait, number>;
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
    /** Obstacles with a Smart Primitive (see G1Primitives), which the robot jumps or vaults on its way. */
    obstacles: G1Obstacle[];
    /** The obstacle the robot sits on, or null. */
    seat: G1Obstacle | null;
    /** An action asked for (sit, stand up), to start at the next generation. */
    private _action;
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
    /** The skeleton's poses: G1_RIG, or SOMA_RIG for the SOMA model. */
    readonly rig: PoseRig;
    constructor(engine: MotionGenerator, bricks: G1Bricks, x?: number, y?: number, angle?: number, rig?: PoseRig);
    /** Walk or run speed in m/s. */
    speedOf(gait: G1Gait): number;
    get moving(): boolean;
    get generating(): boolean;
    /**
     * The sequence being played (a jump, a vault, sitting down, or one given to perform), or null: its name, primitive,
     * the step whose motion is playing, and that step's index.
     */
    get action(): {
        name: string;
        primitive: import('./G1Primitives').G1Primitive | null;
        step: string;
        index: number;
    } | null;
    /** Whether a sequence is asked for, generating or playing, so that new goals wait for it. */
    get busy(): boolean;
    /**
     * Play target steps in controller space, one generation each: poses authored at the origin facing +X and placed
     * with rig.place, such as the human's kick or reach (SOMAActions). With `hold`, the motion ends on the last step
     * and stays there; otherwise the robot goes on to its goal, or stops, after it. The steps start from the next
     * generation, so the motion already planned plays into them.
     */
    perform(steps: G1PrimitiveStep[], { name, hold }?: {
        name?: string;
        hold?: boolean;
    }): void;
    /** The segment being played: its length and transition point in frames, and the playhead. */
    get playback(): {
        frames: number;
        frame: number;
        transition: number;
        stopping: boolean;
    } | null;
    /** Walk or run to a ground point in G1 space, then stop there. */
    moveTo(x: number, y: number, gait?: G1Gait): void;
    /**
     * Sit on a 'sit' obstacle: walk the last metre to it, turn and sit, then stay seated. Start within about two
     * metres of it. Moving again stands up first.
     */
    sit(o: G1Obstacle): void;
    /** Stand up from the seat and stay there. */
    standUp(): void;
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
    /**
     * A jump to start now: the nearest obstacle the probe meets whose first target is no more than `lead` seconds of
     * travel ahead, with its steps placed for this approach. When the robot is already past the first target, it
     * starts at the next one still ahead.
     */
    private _findJump;
    private _distanceToGoal;
    /** The frame of the current segment where new motion can start, and the four context poses ending there. */
    private _contextAt;
    private _generate;
    private _switch;
}
//# sourceMappingURL=G1Controller.d.ts.map