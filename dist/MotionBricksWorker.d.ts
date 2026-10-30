import { GenerateOptions } from './MotionBricksG1';
export interface WorkerLoadOptions {
    /** Absolute URL of the onnxruntime-web module. */
    ort: string;
    /** Absolute URLs of the parts of the ORT .wasm file, for hosts that limit file size. */
    wasm?: string[];
    /** Absolute URL: a folder with pack.json (ending in /) or a .onnx file. */
    model: string;
    sessionOptions: any;
    /** Cache Storage name for model packs, or false to always download. */
    cache?: string | false;
}
export type WorkerRequest = {
    id: number;
    type: 'load';
    options: WorkerLoadOptions;
} | {
    id: number;
    type: 'generate';
    context: ArrayLike<number>[];
    target: ArrayLike<number>[];
    options?: GenerateOptions;
};
//# sourceMappingURL=MotionBricksWorker.d.ts.map