import { GenerateOptions, GeneratedMotion, MotionGenerator, ProgressCallback } from './MotionBricksG1';
import { WorkerLoadOptions } from './MotionBricksWorker';
/** The worker could not start or crashed, as opposed to an error from the model, which is passed on as it is. */
export declare class MotionBricksWorkerError extends Error {
    name: string;
}
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
     * Start the worker and load the model in it. URLs are resolved against the page. It throws a
     * MotionBricksWorkerError when the worker cannot run, and the model's own error when the model fails to load.
     * @param worker a worker running MotionBricksWorker; the default is the one bundled with this package
     */
    static Load(options: WorkerLoadOptions, onProgress?: ProgressCallback, worker?: Worker): Promise<MotionBricksRemote>;
    generate(context: ArrayLike<number>[], target: ArrayLike<number>[], options?: GenerateOptions): Promise<GeneratedMotion>;
}
//# sourceMappingURL=MotionBricksRemote.d.ts.map