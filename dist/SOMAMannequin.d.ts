import { Group, Material, Object3D } from 'three';
import { SOMAPose } from './MotionBricksSOMA';
export declare class SOMAMannequin {
    readonly object: Group<import('three').Object3DEventMap>;
    readonly joints: Object3D[];
    readonly materials: Material[];
    constructor({ shadows, color, accent }?: {
        shadows?: boolean | undefined;
        color?: string | undefined;
        accent?: string | undefined;
    });
    /** Show a SOMA pose: local transforms of SOMA_SKELETON, the root in world space. */
    setPose(pose: SOMAPose | ArrayLike<number>): void;
}
//# sourceMappingURL=SOMAMannequin.d.ts.map