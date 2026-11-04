// Keep a bounded window of downloads in flight, yielding in corpus order.
export async function* prefetchOrdered(items, load, concurrency = 4) {
  const pending = new Map();
  let next = 0;
  function fill() {
    while (next < items.length && pending.size < concurrency) {
      const index = next++;
      // Handle look-ahead failures immediately, even while an earlier download
      // is still running. The error is raised when its turn is reached.
      pending.set(index, Promise.resolve().then(() => load(items[index], index))
        .then(value => ({ value }), error => ({ error })));
    }
  }
  fill();
  for (let index = 0; index < items.length; index++) {
    const result = await pending.get(index);
    pending.delete(index);
    if ('error' in result) throw result.error;
    yield result.value;
    fill();
  }
}
