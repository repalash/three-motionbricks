import { Group, Object3D, Vector3 } from 'three';
import { G1Robot } from './G1Robot';
import { CharacterRig, RetargetedCharacter } from './RetargetedCharacter';
import { G1Bricks, G1Controller, G1ControllerStats, G1Gait } from './G1Controller';
import { MotionGenerator, OrtApi } from './MotionBricksG1';
/** Where the model and ONNX Runtime load from when the options do not say, and where the robot model is. */
export declare const MOTIONBRICKS_DEFAULTS: {
    /** NVIDIA's original fp32 model, 730 MB. A pack from tools/pack_model.py is about 190 MB. */
    model: string;
    /** This package's assets/ folder on jsDelivr: the robot and the other characters (see MOTIONBRICKS_CHARACTERS). */
    assets: string;
    /** The robot, for your GLTFLoader: assets/unitree_g1.glb in this package (6 MB, BSD-3-Clause, Unitree). */
    robot: string;
    ort: string;
};
/**
 * The models in assets/: the G1 the motion is made for, and others it is retargeted to with a rig from
 * CHARACTER_RIGS. Each `file` is relative to MOTIONBRICKS_DEFAULTS.assets.
 */
export declare const MOTIONBRICKS_CHARACTERS: readonly [{
    readonly id: "g1";
    readonly label: "Unitree G1";
    readonly file: "unitree_g1.glb";
}, {
    readonly id: "h1";
    readonly label: "Unitree H1";
    readonly file: "robots/unitree_h1.glb";
    readonly rig: "unitreeH1";
}, {
    readonly id: "t1";
    readonly label: "Booster T1";
    readonly file: "robots/booster_t1.glb";
    readonly rig: "boosterT1";
}, {
    readonly id: "op3";
    readonly label: "Robotis OP3";
    readonly file: "robots/robotis_op3.glb";
    readonly rig: "robotisOP3";
}, {
    readonly id: "robot-expressive";
    readonly label: "RobotExpressive";
    readonly file: "characters/robot_expressive.glb";
    readonly rig: "robotExpressive";
}, {
    readonly id: "astronaut";
    readonly label: "Astronaut";
    readonly file: "characters/astronaut.glb";
    readonly rig: "mixamo";
}, {
    readonly id: "polybot";
    readonly label: "Polybot";
    readonly file: "characters/polybot.glb";
    readonly rig: "mixamo";
}];
export interface MotionBricksCharacterOptions {
    /** A folder with pack.json from tools/pack_model.py (ending in /), or an original MotionBricks G1 .onnx file. */
    model?: string;
    /** The onnxruntime-web module to import. */
    ort?: string;
    /** Parts of the ORT .wasm file, joined in order, for hosts that limit file size. */
    wasm?: string[];
    /**
     * 'wasm' (default) or 'webgpu'. With onnxruntime-web 1.30, WebGPU computes this model wrongly: the robot
     * sinks, sprints or walks away from its goal. It is here for testing newer releases.
     */
    backend?: 'wasm' | 'webgpu';
    /** Run the model in a Web Worker (default true). On the page, every generation stalls rendering. */
    worker?: boolean;
    /** Make the robot's meshes cast and receive shadows. Default true. */
    shadows?: boolean;
    /** Standing, walk and run target poses. The default ones are baked from the model. */
    bricks?: G1Bricks;
}
export type LoadStage = 'runtime' | 'model' | 'warmup';
/**
 * A Unitree G1 that walks and runs where you send it, with every frame of motion generated in the
 * browser by NVIDIA MotionBricks. Add `object` to a three.js scene and call `update(dt)` every frame.
 *
 * ```ts
 * const gltf = await new GLTFLoader().loadAsync(MOTIONBRICKS_DEFAULTS.robot)
 * const g1 = new MotionBricksCharacter()
 * g1.setRobot(gltf.scene)
 * scene.add(g1.object)
 * await g1.load()
 * g1.moveTo(new Vector3(3, 0, -2))
 * renderer.setAnimationLoop(() => { g1.update(clock.getDelta()); renderer.render(scene, camera) })
 * ```
 *
 * Positions are in the world space of `object`'s parent, on the ground plane y = 0 of `object`.
 */
export declare class MotionBricksCharacter {
    readonly options: MotionBricksCharacterOptions;
    /** Holds the robot. Move or scale it to place the character in your scene. */
    readonly object: Group<import('three').Object3DEventMap>;
    /** The model being posed: the G1 itself, or another model the motion is retargeted to. */
    robot?: G1Robot | RetargetedCharacter;
    engine?: MotionGenerator;
    controller?: G1Controller;
    gait: G1Gait;
    /** Called when a generation fails. */
    onError?: (e: unknown) => void;
    private _pendingGoal;
    private _loading?;
    /** Where the robot stood when the model was disposed, so a new load continues from there. */
    private _lastPose?;
    private _drive;
    constructor(options?: MotionBricksCharacterOptions);
    /** The model is loaded and the robot responds to moveTo and drive. */
    get ready(): boolean;
    /** 'wasm' or 'webgpu', once the model is loaded. */
    get backend(): string;
    get stats(): G1ControllerStats | undefined;
    /** The segment being played, for a progress display. */
    get playback(): {
        frames: number;
        frame: number;
        transition: number;
        stopping: boolean;
    } | null;
    /** The robot is walking or running. */
    get moving(): boolean;
    /** Standing still, with no goal and no generation running. */
    get idle(): boolean;
    /**
     * Use a loaded G1 model, such as `gltf.scene` of MOTIONBRICKS_DEFAULTS.robot, which then stands in its rest pose.
     * Load it with whatever loader your project uses.
     */
    setRobot(root: Object3D): G1Robot;
    /**
     * Use another model, such as a robot from assets/robots or a character with a Mixamo skeleton: the G1's motion
     * is retargeted to it with a rig from CHARACTER_RIGS (or your own). It replaces the current robot.
     */
    setCharacter(root: Object3D, rig: CharacterRig, options?: {
        matchSize?: boolean;
    }): RetargetedCharacter;
    /**
     * Load ONNX Runtime and the model, in a worker unless `worker: false`, and warm it up. The robot can be set
     * before or after. Calling it again while loading waits for the same load.
     */
    load(onProgress?: (stage: LoadStage, loaded: number, total: number) => void): Promise<this>;
    private _load;
    /** Advance the motion by dt seconds and pose the robot. Call it every frame. */
    update(dt: number): void;
    /** Convert a world point to the character's G1 ground coordinates. */
    private _toG1;
    /** Walk or run to a world point and stop there, facing the way it walked. */
    moveTo(point: Vector3, gait?: G1Gait): void;
    /**
     * Steer with a world direction on the ground (WASD, a joystick), or null to stop. Call it every frame
     * while the input is held. The gait applies while driving; `gait` stays the default for moveTo. A new direction or gait generates at once; small changes steer the goal,
     * kept a few metres ahead, from the next segment.
     */
    drive(direction: Vector3 | null, gait?: G1Gait): void;
    /** Come to a stop a short way ahead. */
    stop(): void;
    /** The robot's ground position, in world space. */
    getPosition(target?: Vector3): Vector3;
    /** The current goal, in world space, or null. */
    getGoal(target?: Vector3): Vector3 | null;
    /** The direction the robot faces, as an angle around Y in the character's space (0 faces +X). */
    get heading(): number;
    /** Stop the worker. The robot stays in the scene, and load() starts the model again from where it stands. */
    dispose(): void;
}
/** Import onnxruntime-web on the page, for when no worker is used. */
export declare function loadOrt(url: string, wasm?: string[]): Promise<OrtApi & {
    env: any;
}>;
//# sourceMappingURL=MotionBricksCharacter.d.ts.map