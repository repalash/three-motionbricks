import { Group, Material, Object3D } from 'three';
/** The body parts a rig maps, each driven by one G1 body. */
export declare const G1_PARTS: {
    readonly pelvis: "pelvis";
    readonly torso: "torso_link";
    readonly leftThigh: "left_hip_yaw_link";
    readonly leftShin: "left_knee_link";
    readonly leftFoot: "left_ankle_roll_link";
    readonly rightThigh: "right_hip_yaw_link";
    readonly rightShin: "right_knee_link";
    readonly rightFoot: "right_ankle_roll_link";
    readonly leftUpperArm: "left_shoulder_yaw_link";
    readonly leftForearm: "left_elbow_link";
    readonly leftHand: "left_wrist_yaw_link";
    readonly rightUpperArm: "right_shoulder_yaw_link";
    readonly rightForearm: "right_elbow_link";
    readonly rightHand: "right_wrist_yaw_link";
};
export type G1Part = keyof typeof G1_PARTS;
/** The SOMA joint that drives each part, when the motion comes from the SOMA (human) model. */
export declare const SOMA_PARTS: Record<G1Part, string>;
/**
 * How a model's nodes follow the G1. The G1's reference is standing straight with the arms hanging; `rest`
 * turns the model's own rest pose (a T-pose, say) into that.
 */
export interface CharacterRig {
    label: string;
    /** The node driven by each part, by name. Unmapped nodes keep their rest rotation and follow their parent. */
    bones: Partial<Record<G1Part, string>>;
    /** The node that moves the whole body; by default the pelvis node. */
    root?: string;
    /** The direction the model faces as authored, so it is turned to face +X like the G1. */
    forward: '+x' | '-x' | '+z' | '-z';
    /** Turns of nodes, in the model's own axes, that bring its rest pose to the reference: name to Euler XYZ radians. */
    rest?: Record<string, [number, number, number]>;
    /** Nodes that keep to another node's position, such as feet driven by IK in the original rig. */
    attach?: Record<string, string>;
    /** Point the limbs like the G1's before taking the rest pose (default true); `rest` turns come first. */
    autoRest?: boolean;
    /** A node at the far end of a part, where the next mapped part is missing or does not start there (IK feet). */
    ends?: Partial<Record<G1Part, string>>;
}
/** Rigs for the models in assets/, and for any character with a Mixamo skeleton. */
export declare const CHARACTER_RIGS: {
    /** Mixamo skeletons in a T-pose facing -Z, such as Polygonal Mind's 100 Avatars (VRM 0). */
    mixamo: {
        label: string;
        bones: Partial<Record<"pelvis" | "torso" | "leftThigh" | "leftShin" | "leftFoot" | "rightThigh" | "rightShin" | "rightFoot" | "leftUpperArm" | "leftForearm" | "leftHand" | "rightUpperArm" | "rightForearm" | "rightHand", string>>;
        forward: "-z";
    };
    /** The same, facing +Z, as Mixamo exports to glTF. */
    mixamoPlusZ: {
        label: string;
        bones: Partial<Record<"pelvis" | "torso" | "leftThigh" | "leftShin" | "leftFoot" | "rightThigh" | "rightShin" | "rightFoot" | "leftUpperArm" | "leftForearm" | "leftHand" | "rightUpperArm" | "rightForearm" | "rightHand", string>>;
        forward: "+z";
    };
    /** Tomás Laulhé's RobotExpressive (Quaternius, CC0), from the three.js examples. */
    robotExpressive: {
        label: string;
        bones: {
            pelvis: string;
            torso: string;
            leftThigh: string;
            leftShin: string;
            leftFoot: string;
            rightThigh: string;
            rightShin: string;
            rightFoot: string;
            leftUpperArm: string;
            leftForearm: string;
            rightUpperArm: string;
            rightForearm: string;
        };
        root: string;
        forward: "+z";
        attach: {
            'Foot.L': string;
            'Foot.R': string;
        };
        ends: {
            leftShin: string;
            rightShin: string;
            leftForearm: string;
            rightForearm: string;
        };
    };
    unitreeH1: {
        label: string;
        bones: {
            pelvis: string;
            torso: string;
            leftThigh: string;
            leftShin: string;
            leftFoot: string;
            rightThigh: string;
            rightShin: string;
            rightFoot: string;
            leftUpperArm: string;
            leftForearm: string;
            rightUpperArm: string;
            rightForearm: string;
        };
        forward: "+x";
        rest: {
            left_elbow_link: [number, number, number];
            right_elbow_link: [number, number, number];
        };
    };
    boosterT1: {
        label: string;
        bones: {
            pelvis: string;
            torso: string;
            leftThigh: string;
            leftShin: string;
            leftFoot: string;
            rightThigh: string;
            rightShin: string;
            rightFoot: string;
            leftUpperArm: string;
            leftForearm: string;
            leftHand: string;
            rightUpperArm: string;
            rightForearm: string;
            rightHand: string;
        };
        root: string;
        forward: "+x";
    };
    robotisOP3: {
        label: string;
        bones: {
            pelvis: string;
            leftThigh: string;
            leftShin: string;
            leftFoot: string;
            rightThigh: string;
            rightShin: string;
            rightFoot: string;
            leftUpperArm: string;
            leftForearm: string;
            rightUpperArm: string;
            rightForearm: string;
        };
        forward: "+x";
    };
};
/**
 * Plays G1 poses, or SOMA poses with `source: 'soma'`, on another model: a robot with different proportions, or a
 * skinned character. Each mapped node takes the world rotation its part has turned through since the source's
 * reference pose; the root moves with the source's pelvis. `setPose` takes the source's poses: the 36 joint
 * coordinates of G1Robot, or SOMA local transforms.
 */
export declare class RetargetedCharacter {
    /** Holds the model: add this to the scene, or to MotionBricksCharacter.object. */
    readonly object: Group<import('three').Object3DEventMap>;
    readonly model: Object3D;
    readonly rig: CharacterRig;
    readonly materials: Material[];
    /** The model's scale, so its legs are as long as the G1's and its steps land. */
    readonly scale: number;
    private _src;
    private _parts;
    private _root;
    private _pelvis;
    private _g1Pelvis;
    private _pelvisRest;
    private _g1PelvisRest;
    private _attach;
    /**
     * @param model the loaded model, such as `gltf.scene`
     * @param rig one of CHARACTER_RIGS, or your own mapping
     * @param matchSize scale the model so its legs match the source's (default true); otherwise its steps slide
     * @param source the skeleton the motion comes from: 'g1' (default) or 'soma'
     */
    constructor(model: Object3D, rig: CharacterRig, { shadows, matchSize, source }?: {
        shadows?: boolean | undefined;
        matchSize?: boolean | undefined;
        source?: "g1" | "soma" | undefined;
    });
    /** A node's rotation in the space of `object`. */
    private _localQuaternion;
    private _localPosition;
    /** Show a G1 pose: root XYZ, root XYZW quaternion, then the 29 joint angles. */
    setPose(pose: ArrayLike<number>): void;
}
//# sourceMappingURL=RetargetedCharacter.d.ts.map