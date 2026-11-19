/**
 * The app's bucket. On Cloud Run: Cloud Storage via the service's own
 * identity (metadata server). Locally (no BUCKET): an in-memory store.
 */
export type BlobStore = {
  put(name: string, body: string | Uint8Array, contentType?: string): Promise<void>;
  get(name: string): Promise<string | null>;
  /** Raw bytes, for binary objects such as images. */
  getBytes(name: string): Promise<Uint8Array | null>;
  list(prefix?: string): Promise<string[]>;
};

export function bucketStore(
  bucket: string | undefined,
  fetchImpl: typeof fetch = fetch,
): BlobStore {
  if (!bucket) return memoryStore();
  let token: { value: string; until: number } | null = null;
  const auth = async () => {
    if (!token || token.until < Date.now()) {
      const r = await fetchImpl(
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
        { headers: { 'metadata-flavor': 'Google' } },
      );
      if (!r.ok) throw new Error(`metadata token: ${r.status}`);
      const j = (await r.json()) as { access_token: string; expires_in: number };
      token = { value: j.access_token, until: Date.now() + (j.expires_in - 60) * 1000 };
    }
    return { authorization: `Bearer ${token.value}` };
  };
  const api = `https://storage.googleapis.com/storage/v1/b/${bucket}/o`;
  return {
    async put(name, body, contentType = 'text/plain; charset=utf-8') {
      const r = await fetchImpl(
        `https://storage.googleapis.com/upload/storage/v1/b/${bucket}/o?uploadType=media&name=${encodeURIComponent(name)}`,
        {
          method: 'POST',
          headers: { ...(await auth()), 'content-type': contentType },
          body: typeof body === 'string' ? body : new Uint8Array(body),
        },
      );
      if (!r.ok) throw new Error(`put ${name}: ${r.status} ${await r.text()}`);
    },
    async get(name) {
      const b = await this.getBytes(name);
      return b && new TextDecoder().decode(b);
    },
    async getBytes(name) {
      const r = await fetchImpl(`${api}/${encodeURIComponent(name)}?alt=media`, {
        headers: await auth(),
      });
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`get ${name}: ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    },
    async list(prefix = '') {
      const r = await fetchImpl(`${api}?prefix=${encodeURIComponent(prefix)}`, {
        headers: await auth(),
      });
      if (!r.ok) throw new Error(`list: ${r.status}`);
      return (((await r.json()) as { items?: { name: string }[] }).items ?? []).map((i) => i.name);
    },
  };
}

export function memoryStore(): BlobStore {
  const m = new Map<string, Uint8Array>();
  return {
    async put(name, body) {
      m.set(name, typeof body === 'string' ? new TextEncoder().encode(body) : new Uint8Array(body));
    },
    async get(name) {
      const b = m.get(name);
      return b ? new TextDecoder().decode(b) : null;
    },
    async getBytes(name) {
      return m.get(name) ?? null;
    },
    async list(prefix = '') {
      return [...m.keys()].filter((k) => k.startsWith(prefix)).sort();
    },
  };
}
