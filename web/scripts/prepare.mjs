import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = path.join(root, '../data/edfs/yamls');
const target = path.join(root, 'public/examples');
await mkdir(target, { recursive: true });
const examples = [];
for (const file of (await readdir(source)).filter(name => name.endsWith('.yaml')).sort()) {
  const text = await readFile(path.join(source, file), 'utf8');
  await writeFile(path.join(target, file), text);
  examples.push({ file, name: file.replace('.yaml', '').replaceAll('_', ' ') });
}
await writeFile(path.join(target, 'index.json'), JSON.stringify(examples));
if (process.env.PIPELINE_MANIFEST_URL || process.env.PIPELINE_API_URL) {
  await writeFile(path.join(root, 'public/config.json'), JSON.stringify({
    manifestUrl: process.env.PIPELINE_MANIFEST_URL || '',
    apiBaseUrl: process.env.PIPELINE_API_URL || '',
  }, null, 2));
}
