# three-motionbricks

[NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks) for three.js: a Unitree G1 robot that walks and runs where you send it, with every frame of its motion generated in the browser by the model. No animation clips play at runtime.

It needs only `three`, and works with any three.js setup (plain three.js, react-three-fiber, [threepipe](https://github.com/repalash/threepipe)). ONNX Runtime Web is imported at runtime from a URL, and the model runs in a Web Worker, so inference never stalls rendering.

```bash
npm i three-motionbricks
```

Or without a bundler, from jsDelivr in an import map:

```html
<script type="importmap">
{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/",
  "three-motionbricks": "https://cdn.jsdelivr.net/gh/repalash/three-motionbricks@v0.1.0/dist/index.mjs"
}}
</script>
```

```js
import * as THREE from 'three'
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js'
import {MotionBricksCharacter, MOTIONBRICKS_DEFAULTS} from 'three-motionbricks'

const g1 = new MotionBricksCharacter()
// Load the robot with any loader: assets/unitree_g1.glb ships with the package.
g1.setRobot((await new GLTFLoader().loadAsync(MOTIONBRICKS_DEFAULTS.robot)).scene)
scene.add(g1.object)
await g1.load((stage, loaded, total) => console.log(stage, loaded, total))

g1.moveTo(new THREE.Vector3(3, 0, -2))        // walk there and stop
g1.moveTo(new THREE.Vector3(-4, 0, 1), 'run')

// Every frame: steer with a world direction while a key is held (or null), then update.
g1.drive(wasdDirection, shift ? 'run' : 'walk')
g1.update(clock.getDelta())
```

`example/index.html` is a complete plain three.js page: click the floor, or drive with WASD.

## Options

| Option | Default |
| --- | --- |
| `model` | NVIDIA's original `mmm_v6_g1_fp32.onnx` (730 MB). A folder (ending in `/`) with a pack from `tools/pack_model.py` is about 190 MB. |
| `ort` | onnxruntime-web 1.30.0 from jsDelivr. |
| `backend` | `'wasm'`. `'webgpu'` is for testing: with onnxruntime-web 1.30 it computes this model wrongly. |
| `worker` | `true`. With `false` the model runs on the page and each generation stalls rendering. |
| `shadows` | `true`: the robot's meshes cast and receive shadows. |

## The robot

`assets/unitree_g1.glb` (6 MB, 165k triangles) is Unitree's G1 description and meshes in one GLB, built by `tools/build_g1_glb.mjs`. `MOTIONBRICKS_DEFAULTS.robot` points to it on jsDelivr. Its nodes are named after the MJCF bodies, so you can swap its materials, or load your own model with the same node names, and `setRobot` poses it.

## What is inside

| Export | Role |
| --- | --- |
| `MotionBricksCharacter` | The robot, the model and the controller behind `moveTo`, `drive`, `stop` and `update`. |
| `MotionBricksG1` | The engine: a port of NVIDIA's G1 runtime that prepares the ONNX inputs, decodes the output into world space, smooths the root and corrects the endpoints. It has no three.js dependency and runs in Node too. |
| `MotionBricksRemote` | The engine in a Web Worker, behind the same `generate()` call. |
| `G1Controller` | Point-to-point locomotion that generates each segment ahead of playback. |
| `G1Robot` | Poses a loaded G1 model from 36 joint coordinates. |

`tools/` holds the scripts that build the robot GLB (`build_g1_glb.mjs`), pack the model (`pack_model.py`), bake the gait targets (`make_bricks.py`), and the NumPy reference the engine was checked against (`motionbricks_ref.py`, to within 0.001°).

## Development

```bash
npm install
npm run build     # dist/index.mjs, dist/motionbricks.worker.js and the .d.ts files
npm run serve     # opens example/
```

`dist/` is committed so tagged versions load from jsDelivr's GitHub CDN.

## Licenses

The code is Apache 2.0. The MotionBricks ONNX models are under the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/), and the G1 description and meshes are Unitree's, under BSD-3-Clause (`assets/LICENSE-unitree.txt`). The gait targets in `G1Bricks.ts` are the model's own output, seeded from NVIDIA's sample clips.
