import {defineConfig} from 'vite'
import dts from 'vite-plugin-dts'

// A plain three.js library: three stays external, and the model worker is bundled into its own file,
// which the library starts with new Worker(new URL(..., import.meta.url)).
export default defineConfig(({mode}) => ({
    base: '',
    build: {
        sourcemap: true,
        minify: false,
        lib: {
            entry: 'src/index.ts',
            formats: ['es'],
            fileName: () => 'index.mjs',
        },
        outDir: 'dist',
        emptyOutDir: mode !== 'development',
        rollupOptions: {
            external: ['three'],
        },
    },
    worker: {
        format: 'es',
        rollupOptions: {output: {entryFileNames: 'motionbricks.worker.js'}},
    },
    plugins: [
        mode !== 'development' ? dts({tsconfigPath: './tsconfig.json', rollupTypes: false}) : null,
    ],
}))
