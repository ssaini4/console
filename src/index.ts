/**
 * console: a thin multi-model chat. API providers (Anthropic, OpenAI, Gemini)
 * with encrypted keys, remote MCP servers with OAuth, `@alias` to pick a model,
 * and every tool call confirmed by the user. Owner-only.
 */
import { join } from 'node:path';
import { loadConfig } from '@micro-cloud/sdk/config';
import { createSql } from '@micro-cloud/sdk/db';
import { healthRoutes } from '@micro-cloud/sdk/health';
import { createLogger } from '@micro-cloud/sdk/log';
import { webAuth } from '@micro-cloud/sdk/web-auth';
import { Hono, type MiddlewareHandler } from 'hono';
import { streamSSE } from 'hono/streaming';
import { z } from 'zod';
import { bucketStore } from '@micro-cloud/sdk/storage';
import { agent, type Decision, type Event, IMAGE_EXT } from './agent';
import { secretBox } from './crypto';
import { type McpRow, mcpManager } from './mcp';
import { DEFAULT_MODEL, type Kind, listModels } from './providers';

const cfg = loadConfig({
  MASTER_KEY: z.string().regex(/^[0-9a-f]{64}$/, 'MASTER_KEY must be 64 hex characters'),
  OWNER_EMAIL: z.string().optional(),
});
const log = createLogger({ level: cfg.LOG_LEVEL, base: { service: cfg.APP_NAME } });
if (!cfg.DATABASE_URL) throw new Error('DATABASE_URL is not set');
const sql = createSql(cfg.DATABASE_URL);
const box = secretBox(cfg.MASTER_KEY);
const mcp = mcpManager(sql, box, cfg.PUBLIC_URL);
const store = bucketStore(cfg.BUCKET);
if (!cfg.BUCKET) log.warn('BUCKET is not set: generated images are kept in memory only');
const chat = agent(sql, box, mcp, store);
const issuer = cfg.AUTH_ISSUER ?? 'http://localhost:8090';
const resource = cfg.AUTH_AUDIENCE ?? `${cfg.PUBLIC_URL}/mcp`;
const owner = cfg.DEV_AUTH ? cfg.DEV_USER : cfg.OWNER_EMAIL?.toLowerCase();

const app = new Hono();
healthRoutes(app, { service: cfg.APP_NAME, commit: cfg.GIT_SHA, ping: async () => void (await sql`select 1`) });
const auth = webAuth(app, {
  publicUrl: cfg.PUBLIC_URL,
  issuer,
  resource,
  appName: cfg.APP_NAME,
  devUser: cfg.DEV_AUTH ? cfg.DEV_USER : null,
});

/** Signed in AND the owner. With OWNER_EMAIL unset, nobody gets in. */
const onlyOwner: MiddlewareHandler = async (c, next) => {
  if (!owner || auth.user(c).email !== owner) return c.text('Forbidden', 403);
  // JSON-only writes: a cross-site form can't send application/json without a CORS preflight.
  if (c.req.method !== 'GET' && !(c.req.header('content-type') ?? '').startsWith('application/json')) {
    return c.json({ error: 'expected application/json' }, 415);
  }
  return next();
};
app.use('/api/*', auth.requireUser, onlyOwner);

const page = Bun.file(join(import.meta.dir, '..', 'public', 'index.html'));
app.get('/', auth.requireUser, onlyOwner, (c) => c.html(page.text()));

const bad = (c: { json: (b: unknown, s: 400) => Response }, e: unknown) =>
  c.json({ error: e instanceof z.ZodError ? z.prettifyError(e) : (e as Error).message }, 400);

// ------------------------------------------------------------------ state

app.get('/api/state', async (c) => {
  const [providers, servers, chats] = await Promise.all([
    sql`select id, alias, kind, model, key_hint from providers order by id`,
    sql<McpRow[]>`select * from mcp_servers order by id`,
    sql`select id, title, updated_at from chats order by updated_at desc limit 200`,
  ]);
  return c.json({
    providers,
    servers: servers.map((s) => ({
      id: s.id,
      name: s.name,
      url: s.url,
      enabled: s.enabled,
      hasHeaders: Boolean(s.headers_enc),
      connected: s.oauth_enc ? Boolean(JSON.parse(box.open(s.oauth_enc)).tokens) : false,
      always_allow: s.always_allow,
    })),
    chats,
  });
});

// ------------------------------------------------------------------ providers

const Alias = z.string().regex(/^[a-z0-9_-]{1,32}$/, 'alias: lowercase letters, digits, _ or -');
const Kinds = z.enum(['anthropic', 'openai', 'gemini']);

app.post('/api/providers', async (c) => {
  try {
    const b = z
      .object({ kind: Kinds, alias: Alias, apiKey: z.string().min(8).max(500), model: z.string().max(200).optional() })
      .parse(await c.req.json());
    const models = await listModels(b.kind, b.apiKey).catch((e) => {
      throw new Error(`Key check failed: ${(e as Error).message}`);
    });
    const model = b.model || DEFAULT_MODEL[b.kind] || models[0] || '';
    await sql`insert into providers (alias, kind, model, api_key_enc, key_hint)
              values (${b.alias}, ${b.kind}, ${model}, ${box.seal(b.apiKey)}, ${`…${b.apiKey.slice(-4)}`})`;
    return c.json({ ok: true, models });
  } catch (e) {
    return bad(c, e);
  }
});

app.get('/api/providers/:id/models', async (c) => {
  const [p] = await sql<{ kind: Kind; api_key_enc: string }[]>`
    select kind, api_key_enc from providers where id = ${c.req.param('id')}`;
  if (!p) return c.json({ error: 'not found' }, 404);
  try {
    return c.json({ models: await listModels(p.kind, box.open(p.api_key_enc)) });
  } catch (e) {
    return bad(c, e);
  }
});

app.patch('/api/providers/:id', async (c) => {
  try {
    const b = z
      .object({ model: z.string().min(1).max(200).optional(), alias: Alias.optional(), apiKey: z.string().min(8).max(500).optional() })
      .parse(await c.req.json());
    const id = c.req.param('id');
    if (b.model) await sql`update providers set model = ${b.model} where id = ${id}`;
    if (b.alias) await sql`update providers set alias = ${b.alias} where id = ${id}`;
    if (b.apiKey) {
      await sql`update providers set api_key_enc = ${box.seal(b.apiKey)}, key_hint = ${`…${b.apiKey.slice(-4)}`} where id = ${id}`;
    }
    return c.json({ ok: true });
  } catch (e) {
    return bad(c, e);
  }
});

app.delete('/api/providers/:id', async (c) => {
  await sql`delete from providers where id = ${c.req.param('id')}`;
  return c.json({ ok: true });
});

// ------------------------------------------------------------------ MCP servers

const Url = z.url().refine((u) => u.startsWith('https://') || cfg.NODE_ENV === 'development', 'URL must be https');
const Headers = z.record(z.string().regex(/^[A-Za-z0-9-]+$/), z.string().max(4000));

app.post('/api/mcp', async (c) => {
  try {
    const b = z
      .object({ name: z.string().regex(/^[a-zA-Z0-9_-]{1,32}$/), url: Url, headers: Headers.optional() })
      .parse(await c.req.json());
    const headers = b.headers && Object.keys(b.headers).length ? box.seal(JSON.stringify(b.headers)) : null;
    const [row] = await sql<{ id: string }[]>`
      insert into mcp_servers (name, url, headers_enc) values (${b.name}, ${b.url}, ${headers}) returning id`;
    return c.json({ ok: true, id: row?.id });
  } catch (e) {
    return bad(c, e);
  }
});

app.patch('/api/mcp/:id', async (c) => {
  try {
    const b = z
      .object({ enabled: z.boolean().optional(), headers: Headers.optional(), always_allow: z.array(z.string()).optional() })
      .parse(await c.req.json());
    const id = c.req.param('id');
    if (b.enabled !== undefined) await sql`update mcp_servers set enabled = ${b.enabled} where id = ${id}`;
    if (b.always_allow) await sql`update mcp_servers set always_allow = ${b.always_allow} where id = ${id}`;
    if (b.headers) {
      const h = Object.keys(b.headers).length ? box.seal(JSON.stringify(b.headers)) : null;
      await sql`update mcp_servers set headers_enc = ${h} where id = ${id}`;
    }
    return c.json({ ok: true });
  } catch (e) {
    return bad(c, e);
  }
});

app.delete('/api/mcp/:id', async (c) => {
  await sql`delete from mcp_servers where id = ${c.req.param('id')}`;
  return c.json({ ok: true });
});

app.post('/api/mcp/:id/connect', async (c) => {
  const [row] = await sql<McpRow[]>`select * from mcp_servers where id = ${c.req.param('id')}`;
  if (!row) return c.json({ error: 'not found' }, 404);
  try {
    return c.json({ redirect: await mcp.beginOAuth(row) });
  } catch (e) {
    return bad(c, e);
  }
});

app.post('/api/mcp/:id/disconnect', async (c) => {
  await sql`update mcp_servers set oauth_enc = null where id = ${c.req.param('id')}`;
  return c.json({ ok: true });
});

/** Lists a server's tools: the "test connection" button. */
app.get('/api/mcp/:id/tools', async (c) => {
  const [row] = await sql<McpRow[]>`select * from mcp_servers where id = ${c.req.param('id')}`;
  if (!row) return c.json({ error: 'not found' }, 404);
  try {
    const client = await mcp.connect(row);
    try {
      const { tools } = await client.listTools();
      return c.json({ tools: tools.map((t) => ({ name: t.name, description: t.description ?? '' })) });
    } finally {
      await client.close();
    }
  } catch (e) {
    return bad(c, e);
  }
});

app.get('/mcp-oauth/callback', auth.requireUser, onlyOwner, async (c) => {
  const { code, state, error } = c.req.query();
  if (error || !code || !state) return c.text(`OAuth failed: ${error ?? 'missing code'}`, 400);
  try {
    const name = await mcp.finishOAuth(state, code);
    return c.redirect(`/?connected=${encodeURIComponent(name)}`, 302);
  } catch (e) {
    log.warn('oauth callback failed', { error: (e as Error).message });
    return c.text(`OAuth failed: ${(e as Error).message}`, 400);
  }
});

// ------------------------------------------------------------------ chats

app.post('/api/chats', async (c) => {
  const [row] = await sql`insert into chats default values returning id, title, updated_at`;
  return c.json(row);
});

app.get('/api/chats/:id', async (c) => {
  const messages = await sql`
    select id, role, alias, model, text, tool_calls, tool_results, raw from messages
    where chat_id = ${c.req.param('id')} order by id`;
  // Thinking lives in the provider's raw blocks (Anthropic `thinking`, Gemini `thought` parts).
  type Block = { type?: string; thinking?: string; thought?: boolean; text?: string };
  const thinking = (raw: unknown) =>
    (Array.isArray(raw) ? (raw as Block[]) : [])
      .map((b) => (b.type === 'thinking' ? b.thinking : b.thought ? b.text : '') ?? '')
      .filter(Boolean)
      .join('\n\n');
  return c.json({ messages: messages.map(({ raw, ...m }) => ({ ...m, thinking: thinking(raw) })) });
});

app.get('/api/images/:chat/:file{[0-9a-f-]+\\.(png|jpg|webp)}', async (c) => {
  const file = c.req.param('file');
  const data = await store.getBytes(`images/${c.req.param('chat')}/${file}`);
  if (!data) return c.notFound();
  const mime = Object.keys(IMAGE_EXT).find((m) => file.endsWith(`.${IMAGE_EXT[m]}`)) ?? 'image/png';
  return c.body(new Uint8Array(data), 200, { 'content-type': mime, 'cache-control': 'private, max-age=31536000, immutable' });
});

app.delete('/api/chats/:id', async (c) => {
  await sql`delete from chats where id = ${c.req.param('id')}`;
  return c.json({ ok: true });
});

const sse = (c: Parameters<typeof streamSSE>[0], run: (emit: (e: Event) => void) => Promise<void>) =>
  streamSSE(c, async (s) => {
    const queue: Promise<void>[] = [];
    await run((e) => void queue.push(s.writeSSE({ data: JSON.stringify(e) })));
    await Promise.all(queue);
  });

app.post('/api/chats/:id/send', async (c) => {
  const b = z.object({ text: z.string().min(1).max(100_000) }).safeParse(await c.req.json());
  if (!b.success) return c.json({ error: 'text required' }, 400);
  return sse(c, (emit) => chat.send(c.req.param('id'), b.data.text, emit));
});

app.post('/api/chats/:id/resume', async (c) => {
  const b = z
    .object({ decisions: z.record(z.string(), z.enum(['allow', 'always', 'deny'])) })
    .safeParse(await c.req.json());
  if (!b.success) return c.json({ error: 'decisions required' }, 400);
  return sse(c, (emit) => chat.resume(c.req.param('id'), b.data.decisions as Record<string, Decision>, emit));
});

const server = Bun.serve({ port: cfg.PORT, hostname: '0.0.0.0', fetch: app.fetch, idleTimeout: 0 });
log.info('listening', { port: server.port, commit: cfg.GIT_SHA, ownerSet: Boolean(owner) });
process.on('SIGTERM', async () => {
  await server.stop();
  await sql.end({ timeout: 5 });
  process.exit(0);
});
