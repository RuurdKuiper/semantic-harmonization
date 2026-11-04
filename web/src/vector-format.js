export function decodeVectors(bytes, dtype, count, dimensions) {
  const size = count * dimensions;
  if (dtype === 'fp32' && bytes.byteLength === size * 4) {
    return { values: new Float32Array(bytes), normalizers: null };
  }
  if (dtype === 'int8' && bytes.byteLength === size + count * 4) {
    const normalizers = new Float32Array(count);
    const view = new DataView(bytes);
    for (let i = 0; i < count; i++) {
      normalizers[i] = view.getFloat32(size + i * 4, true);
      if (!Number.isFinite(normalizers[i]) || normalizers[i] <= 0) throw Error('Invalid quantized vector norm.');
    }
    return { values: new Int8Array(bytes, 0, size), normalizers };
  }
  throw Error('Terminology rows and vectors do not match.');
}
