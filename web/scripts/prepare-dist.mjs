import { mkdir, writeFile } from 'node:fs/promises';

const worker = `const INDEX_PATH = '/index.html';

export default {
  async fetch(request, env) {
    if (!env.ASSETS || typeof env.ASSETS.fetch !== 'function') {
      return new Response('Static asset binding unavailable', { status: 503 });
    }

    const response = await env.ASSETS.fetch(request);
    if (response.status !== 404 || request.method !== 'GET') return response;

    const url = new URL(request.url);
    url.pathname = INDEX_PATH;
    return env.ASSETS.fetch(new Request(url, request));
  },
};
`;

await mkdir(new URL('../dist/server/', import.meta.url), { recursive: true });
await writeFile(new URL('../dist/server/index.js', import.meta.url), worker, 'utf8');
