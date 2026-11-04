import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prefetchOrdered } from '../src/prefetch.js';

test('downloads overlap within the memory window and yield in corpus order', async () => {
  let active = 0, peak = 0;
  const items = [0, 1, 2, 3, 4, 5, 6];
  const results = [];
  for await (const value of prefetchOrdered(items, async value => {
    active++; peak = Math.max(peak, active);
    await new Promise(resolve => setTimeout(resolve, (7 - value) * 2));
    active--;
    return value;
  })) results.push(value);
  assert.deepEqual(results, items);
  assert.equal(peak, 4);
});

test('a failed look-ahead download is handled and reported in order', async () => {
  const failure = Error('Download failed');
  const iterator = prefetchOrdered([0, 1], async value => {
    if (value === 1) throw failure;
    await new Promise(resolve => setImmediate(resolve));
    return value;
  });
  assert.equal((await iterator.next()).value, 0);
  await assert.rejects(iterator.next(), failure);
});
