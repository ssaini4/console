// Runs against a real Postgres (TEST_DATABASE_URL) and a real in-process MCP
// server; only the model is scripted. `bun test` skips it without a database.
import { afterAll, beforeAll, expect, test } from 'bun:test';
import { join } from 'node:path';
import { createSql, migrate } from '@micro-cloud/sdk/db';
import { memoryStore } from '@micro-cloud/sdk/storage';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { WebStandardStreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js';
import { z } from 'zod';
import { agent, type Event } from './agent';
import { secretBox } from './crypto';
import { mcpManager } from './mcp';
import { type StepIn, type StepOut, steps } from './providers';

const url = process.env.TEST_DATABASE_URL;
const t = url ? test : test.skip;
const sql = createSql(url ?? 'postgres://unused', { max: 2 });
const box = secretBox('11'.repeat(32));
const ran: string[] = [];

// A real MCP server with one tool.
const mcpHttp = Bun.serve({
  port: 0,
  fetch: async (req) => {
    const server = new McpServer({ name: 'test', version: '1' });
    server.registerTool('echo', { inputSchema: { msg: z.string() } }, async ({ msg }) => {
      ran.push(msg);
      return { content: [{ type: 'text', text: `echo:${msg}` }] };
    });
    const transport = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    await server.connect(transport);
    return transport.handleRequest(req);
  },
});

// Scripted model: calls echo once, then answers with what it saw.
const seen: StepIn[] = [];
steps.anthropic = async (p: StepIn): Promise<StepOut> => {
  seen.push(p);
  const last = p.history.at(-1);
  if (last?.role === 'user') {
    return { text: 'calling', calls: [{ id: `c${seen.length}`, name: 'test__echo', args: { msg: 'hi' } }], raw: [{ type: 'text', text: 'calling' }] };
  }
  return { text: `got ${last?.tool_results[0]?.content}`, calls: [], raw: [{ type: 'text', text: 'done' }] };
};
steps.openai = async (p) => {
  seen.push(p);
  return { text: 'from gpt', calls: [], raw: null };
};

const collect = () => {
  const events: Event[] = [];
  return { events, emit: (e: Event) => events.push(e) };
};

beforeAll(async () => {
  if (!url) return;
  await sql`drop table if exists messages, chats, mcp_servers, providers, app_migrations cascade`;
  await migrate(sql, join(import.meta.dir, '..', 'migrations'));
  await sql`insert into providers (alias, kind, model, api_key_enc, key_hint) values
    ('claude', 'anthropic', 'm1', ${box.seal('k1')}, 'x'), ('gpt', 'openai', 'm2', ${box.seal('k2')}, 'x')`;
  await sql`insert into mcp_servers (name, url) values ('test', ${`http://localhost:${mcpHttp.port}/mcp`})`;
});
afterAll(async () => {
  mcpHttp.stop();
  await sql.end();
});

t('tool calls wait for approval, run once allowed, and other models see the chat', async () => {
  const a = agent(sql, box, mcpManager(sql, box, 'http://localhost'), memoryStore());
  const [chat] = await sql<{ id: string }[]>`insert into chats default values returning id`;
  const id = chat!.id;

  // 1. Default model (first provider) asks to run echo: nothing runs yet.
  let r = collect();
  await a.send(id, 'hello', r.emit);
  expect(r.events.find((e) => e.type === 'confirm')).toMatchObject({ calls: [{ server: 'test', tool: 'echo' }] });
  expect(ran).toEqual([]);
  expect(seen[0]?.tools.map((x) => x.name)).toEqual(['generate_image', 'test__echo']);

  // 2. Deny: the model gets an error result, the tool never runs.
  r = collect();
  await a.resume(id, { c1: 'deny' }, r.emit);
  expect(ran).toEqual([]);
  expect(seen.at(-1)?.history.at(-1)?.tool_results[0]).toMatchObject({ isError: true });

  // 3. New message, "always allow": runs now, and next time without asking.
  r = collect();
  await a.send(id, 'again', r.emit);
  await a.resume(id, { c3: 'always' }, collect().emit);
  expect(ran).toEqual(['hi']);
  r = collect();
  await a.send(id, 'third', r.emit);
  expect(r.events.some((e) => e.type === 'confirm')).toBe(false);
  expect(ran).toEqual(['hi', 'hi']);
  expect(r.events.at(-2)).toMatchObject({ type: 'start' });

  // 4. @gpt picks the other model, which receives the whole chat.
  r = collect();
  await a.send(id, '@gpt summarize', r.emit);
  expect(r.events[0]).toMatchObject({ type: 'start', alias: 'gpt' });
  const h = seen.at(-1)!.history;
  expect(h.filter((m) => m.role === 'user').map((m) => m.text)).toEqual(['hello', 'again', 'third', '@gpt summarize']);
  expect(h.some((m) => m.alias === 'claude' && m.text === 'got echo:hi')).toBe(true);

  // 5. With no mention, the chat stays on the last model used.
  await a.send(id, 'and then?', collect().emit);
  expect(seen.at(-1)?.model).toBe('m2');
});

t('attached files are stored and reach the model with the message', async () => {
  const store = memoryStore();
  const a = agent(sql, box, mcpManager(sql, box, 'http://localhost'), store);
  const [chat] = await sql<{ id: string }[]>`insert into chats default values returning id`;
  const file = { name: 'notes.md', mime: 'text/markdown', data: new TextEncoder().encode('# hi') };
  await a.send(chat!.id, '@gpt read this', collect().emit, [file]);
  const user = seen.at(-1)!.history.at(-1)!;
  expect(user.files).toEqual([file]);
  const [row] = await sql<{ attachments: { id: string; name: string }[] }[]>`
    select attachments from messages where chat_id = ${chat!.id} and role = 'user'`;
  expect(row!.attachments).toMatchObject([{ name: 'notes.md', mime: 'text/markdown', size: 4 }]);
  expect(await store.list(`console/files/${chat!.id}/`)).toHaveLength(1);
  const [c] = await sql<{ title: string }[]>`select title from chats where id = ${chat!.id}`;
  expect(c?.title).toBe('read this');
});

t('secret box round-trips and rejects tampering', () => {
  const s = box.seal('sk-secret');
  expect(s).not.toContain('sk-secret');
  expect(box.open(s)).toBe('sk-secret');
  expect(() => box.open(`${s.slice(0, -2)}AA`)).toThrow();
});
