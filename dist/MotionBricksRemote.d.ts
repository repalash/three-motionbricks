import { GenerateOptions, GeneratedMotion, MotionGenerator, ProgressCallback } from './MotionBricksG1';
import { WorkerLoadOptions } from './MotionBricksWorker';
/**
 * MotionBricksG1 in a Web Worker (MotionBricksWorker.ts), behind the same generate() call.
 * Loading, dequantizing the weights and every inference happen off the main thread.
 */
export declare class MotionBricksRemote implements MotionGenerator {
    readonly worker: Worker;
    backend: string;
    private _id;
    private _pending;
    constructor(worker: Worker);
    private _call;
    /**
     * Start the worker and load the model in it. URLs are resolved against the page.
     * @param worker a worker running MotionBricksWorker; the default is the one bundled with this package
     */
    static Load(options: WorkerLoadOptions, onProgress?: ProgressCallback, worker?: Worker): Promise<MotionBricksRemote>;
    generate(context: ArrayLike<number>[], target: ArrayLike<number>[], options?: GenerateOptions): Promise<GeneratedMotion>;
}
//# sourceMappingURL=MotionBricksRemote.d.ts.map