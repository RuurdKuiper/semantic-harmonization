import { bm25, queries, rankScores, standardKey } from './core.js';
import { prefetchOrdered } from './prefetch.js';
import { decodeVectors } from './vector-format.js';

let database;
function db() {
  if (!database) database = new Promise((resolve, reject) => {
    const request = indexedDB.open('semantic-harmonization-minilm-v1', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('assets');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return database;
}
async function readCache(key) {
  const database = await db();
  return new Promise((resolve, reject) => {
    const request = database.transaction('assets').objectStore('assets').get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function writeCache(key, bytes) {
  const database = await db();
  return new Promise((resolve, reject) => {
    const tx = database.transaction('assets', 'readwrite');
    tx.objectStore('assets').put(bytes, key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
async function removeObsoleteVectors(hashes = []) {
  if (!Array.isArray(hashes) || hashes.some(key => !/^[0-9a-f]{64}$/.test(key))) throw Error('Invalid cache upgrade metadata.');
  if (!hashes.length) return;
  try {
    const database = await db();
    await new Promise((resolve, reject) => {
      const tx = database.transaction('assets', 'readwrite');
      for (const key of hashes) tx.objectStore('assets').delete(key);
      tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error);
    });
  } catch { /* Retrieval still works if persistent browser storage is unavailable. */ }
}
async function hash(bytes) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
}
async function asset(descriptor, base, progress) {
  const url = new URL(descriptor.url, base).href;
  const key = descriptor.sha256;
  let bytes;
  try { bytes = await readCache(key); } catch { /* Retrieval can continue without persistent storage. */ }
  if (bytes && (bytes.byteLength !== descriptor.bytes || await hash(bytes) !== key)) bytes = null;
  if (!bytes) {
    progress(`Downloading ${Math.ceil(descriptor.bytes / 1024 / 1024)} MB of terminology data…`);
    const response = await fetch(url);
    if (response.status === 401) throw Error('The terminology service is unavailable. Please contact the tool administrator.');
    if (!response.ok) throw Error(`Terminology download failed (${response.status}). Check the asset host and its CORS settings.`);
    bytes = await response.arrayBuffer();
    if (bytes.byteLength !== descriptor.bytes || await hash(bytes) !== key) throw Error('A terminology download failed its integrity check. Please retry.');
    try { await writeCache(key, bytes); } catch { progress('Browser storage is full or unavailable; continuing without saving this download.'); }
  }
  return bytes;
}
async function jsonAsset(descriptor, base, progress) {
  const bytes = await asset(descriptor, base, progress);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}

let encoder, rows = [], ranking;
function rowAt(position) {
  const i = ranking.order[position];
  return { ...rows[i], rank: position + 1, lexical_score: ranking.lexical[i], embedding_score: ranking.semantic[i], retrieval_score: ranking.scores[i] };
}
async function retrieve({ edf, manifestUrl, vocabularies, weight, provider = 'minilm', embeddingVector }, progress) {
  rows = []; ranking = undefined;
  const response = await fetch(manifestUrl, { cache: 'no-cache' });
  if (!response.ok) throw Error('Cannot load the terminology data. Please retry or contact the tool administrator.');
  const manifest = await response.json();
  const dimensions = provider === 'openai' ? 3072 : 384;
  if (manifest.format !== `semantic-harmonization-${provider}-v1` || manifest.dimensions !== dimensions
    || manifest.model !== (provider === 'openai' ? 'text-embedding-3-large' : 'Xenova/all-MiniLM-L6-v2')
    || !['fp32', 'int8'].includes(manifest.dtype)
    || (manifest.dtype === 'int8' && (provider !== 'minilm' || manifest.quantization !== 'symmetric-int8-unit-vectors-v1'))
    || (provider === 'minilm' && manifest.maxLength !== 256)) throw Error('The terminology pack does not match the selected embedding model.');
  if (manifest.dtype === 'int8') await removeObsoleteVectors(manifest.obsoleteVectorHashes);
  const selected = vocabularies.map(id => {
    const item = manifest.vocabularies.find(item => item.id === id);
    if (!item) throw Error(`The terminology pack does not include ${id}.`);
    return item;
  });
  if (!selected.length) throw Error('Select at least one terminology.');
  const query = queries(edf);
  let vector;
  if (provider === 'openai') {
    if (!Array.isArray(embeddingVector) || embeddingVector.length !== dimensions || embeddingVector.some(value => !Number.isFinite(value))) throw Error('The embedding service returned an invalid query vector.');
    vector = embeddingVector;
  } else {
  progress('Loading MiniLM on your computer. The model is downloaded on first use…');
  if (!encoder) {
    const { createMiniLMEncoder } = await import('./minilm-runtime.js');
    encoder = await createMiniLMEncoder(manifest.model, { dtype: 'fp32', device: 'wasm',
      progress_callback: event => { if (event.status === 'progress') progress(`Loading MiniLM: ${Math.round(event.progress)}%`); } });
  }
  const embedding = await encoder(query.semantic, { pooling: 'mean', normalize: true, truncation: true, max_length: 256 });
  vector = embedding.data;
  if (vector.length !== dimensions) throw Error('Unexpected MiniLM output dimensions.');
  }
  const total = selected.reduce((sum, vocabulary) => sum + vocabulary.count, 0);
  const totalFragments = selected.reduce((sum, vocabulary) => sum + vocabulary.chunks.length, 0);
  const lexical = new Float64Array(total), semantic = new Float64Array(total);
  let cursor = 0, loadedFragments = 0;
  for (const vocabulary of selected) {
    const fragmentProgress = message => progress(
      `${loadedFragments} of ${totalFragments} fragments loaded · ${vocabulary.id} · ${message}`,
      loadedFragments / totalFragments,
    );
    fragmentProgress('Preparing search statistics…');
    const stats = await jsonAsset(vocabulary.lexical, manifestUrl, fragmentProgress);
    let vocabularyCount = 0;
    const fragments = prefetchOrdered(vocabulary.chunks, async chunk => {
      const [metadata, bytes] = await Promise.all([
        jsonAsset(chunk.rows, manifestUrl, fragmentProgress),
        asset(chunk.vectors, manifestUrl, fragmentProgress),
      ]);
      if (metadata.length !== chunk.count) throw Error('Terminology rows and vectors do not match.');
      const decoded = decodeVectors(bytes, manifest.dtype, chunk.count, dimensions);
      loadedFragments++;
      fragmentProgress('Loading up to 4 fragments in parallel…');
      return { chunk, metadata, decoded };
    });
    for await (const { chunk, metadata, decoded } of fragments) {
      if (chunk.start !== vocabularyCount) throw Error('The terminology manifest has missing or overlapping chunks.');
      const { values: vectors, normalizers } = decoded;
      for (let i = 0; i < chunk.count; i++) {
        const [code, description] = metadata[i];
        let cosine = 0;
        for (let d = 0; d < dimensions; d++) cosine += vector[d] * vectors[i * dimensions + d];
        semantic[cursor] = normalizers ? cosine * normalizers[i] : cosine;
        lexical[cursor] = bm25(query.lexical, description, stats);
        rows.push({ code, description, vocabulary: vocabulary.id });
        cursor++;
      }
      vocabularyCount += chunk.count;
      fragmentProgress(`Scored ${cursor.toLocaleString()} of ${total.toLocaleString()} codes`);
    }
    if (vocabularyCount !== vocabulary.count) throw Error('Incomplete terminology pack.');
  }
  ranking = rankScores(lexical, semantic, weight);
  return { count: rows.length, queries: query, vocabularyVersions: selected.map(item => item.version),
    model: manifest.model, provider, dimensions, lexicalWeight: weight, dtype: manifest.dtype, maxLength: manifest.maxLength };
}

self.onmessage = async ({ data: { id, method, args } }) => {
  const progress = (message, fraction) => self.postMessage({ id, progress: { message, fraction } });
  try {
    let result;
    if (method === 'retrieve') result = await retrieve(args, progress);
    else if (method === 'page') {
      if (!ranking) throw Error('Run retrieval first.');
      const items = []; let total = 0;
      for (let position = 0; position < ranking.order.length; position++) {
        const row = rows[ranking.order[position]];
        if (args.filter && !`${row.code} ${row.description} ${row.vocabulary}`.toLowerCase().includes(args.filter.toLowerCase())) continue;
        if (total >= args.offset && items.length < args.limit) items.push(rowAt(position));
        total++;
      }
      result = { items, total };
    } else if (method === 'batch') {
      if (!ranking) throw Error('Run retrieval first.');
      result = Array.from({ length: Math.max(0, Math.min(args.limit, rows.length - args.offset)) }, (_, i) => rowAt(args.offset + i));
    } else if (method === 'available') result = [...new Set(rows.map(standardKey))];
    else if (method === 'clear') {
      const database = await db();
      await new Promise((resolve, reject) => {
        const tx = database.transaction('assets', 'readwrite'); tx.objectStore('assets').clear();
        tx.oncomplete = resolve; tx.onerror = () => reject(tx.error);
      });
      result = true;
    } else throw Error('Unknown retrieval operation.');
    self.postMessage({ id, result });
  } catch (error) { self.postMessage({ id, error: error.message || String(error) }); }
};
