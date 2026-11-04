import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: './',
  resolve: {
    alias: [
      { find: '@minilm-source', replacement: fileURLToPath(new URL('./node_modules/@huggingface/transformers/src', import.meta.url)) },
      { find: /^(?:.*\/)?onnx-node\.js$/, replacement: fileURLToPath(new URL('./src/no-node-runtime.js', import.meta.url)) },
    ],
  },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
