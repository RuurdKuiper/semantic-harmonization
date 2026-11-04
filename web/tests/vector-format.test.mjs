import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeVectors } from '../src/vector-format.js';

test('int8 decoding preserves negative components and uses little-endian norms', () => {
  const bytes = new ArrayBuffer(8);
  new Int8Array(bytes, 0, 4).set([127, -64, 32, 0]);
  const norm = Math.fround(1 / Math.hypot(127, -64, 32));
  new DataView(bytes).setFloat32(4, norm, true);
  const result = decodeVectors(bytes, 'int8', 1, 4);
  assert.deepEqual([...result.values], [127, -64, 32, 0]);
  assert.equal(result.normalizers[0], norm);
  assert.ok(Math.abs(Math.hypot(...result.values) * norm - 1) < 1e-6);
});

test('both formats reject truncated chunks and invalid int8 norms', () => {
  assert.throws(() => decodeVectors(new ArrayBuffer(7), 'int8', 1, 4), /do not match/);
  assert.throws(() => decodeVectors(new ArrayBuffer(8), 'int8', 1, 4), /Invalid/);
  assert.throws(() => decodeVectors(new ArrayBuffer(7), 'fp32', 1, 4), /do not match/);
  const floats = new Float32Array([1, -2, 3, 4]);
  assert.deepEqual([...decodeVectors(floats.buffer, 'fp32', 1, 4).values], [...floats]);
});
