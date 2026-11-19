import type { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type { Sql } from 'postgres';
import type { SecretBox } from './crypto';
import type { McpManager, Tool } from './mcp';
import { type Call, type Kind, type Msg, type Result, steps } from './providers';

export type Event =
  | { type: 'start'; alias: string; model: string }
  | { type: 'text'; delta: string }
  | { type: 'calls'; calls: Call[] }
  | { type: 'results'; results: Result[] }
  | { type: 'confirm'; calls: (Call & { server: string; tool: string })[] }
  | { type: 'note'; text: string }
  | { type: 'error'; message: string }
  | { type: 'done' };
export type Emit = (e: Event) => void;
export type Decision = 'allow' | 'always' | 'deny';

type ProviderRow = { id: string; alias: string; kind: Kind; model: string; api_key_enc: string };
type Row = Msg & { id: string };

const MAX_STEPS = 25;
const MENTION = /(?:^|\s)@([a-z0-9_-]{1,32})\b/g;

function system(alias: string, model: string) {
  return `You are @${alias} (${model}), one of several AI models in a personal chat console. \
Earlier turns may come from other models; they appear as "[@name]: ...". Read them as context from \
another assistant, not as your own words. You may have tools from connected MCP servers. Every \
tool call is shown to the user, who approves or denies it. If a call is denied, do not retry it; \
ask what they want instead. Answer in Markdown.`;
}

export function agent(sql: Sql, box: SecretBox, mcp: McpManager) {
  const providers = () => sql<ProviderRow[]>`select * from providers order by id`;
  const history = (chatId: string) =>
    sql<Row[]>`select id, role, alias, kind, text, tool_calls, tool_results, raw
               from messages where chat_id = ${chatId} order by id`;

  /** The assistant turn whose tool calls still lack results, if any. */
  function pending(rows: Row[]): Row | null {
    const last = rows.at(-1);
    return last?.role === 'assistant' && last.tool_calls.length ? last : null;
  }

  async function addTool(chatId: string, from: Row, results: Result[]) {
    await sql`insert into messages (chat_id, role, alias, kind, tool_results)
              values (${chatId}, 'tool', ${from.alias}, ${from.kind}, ${sql.json(results)})`;
  }

  async function call(t: Tool | undefined, clients: Map<string, Client>, c: Call): Promise<Result> {
    const client = t && clients.get(t.server.id);
    if (!t || !client) return { id: c.id, name: c.name, content: `Tool ${c.name} is not available.`, isError: true };
    try {
      const r = await client.callTool({ name: t.tool, arguments: c.args });
      const parts = (r.content as { type: string; text?: string }[] | undefined) ?? [];
      const content = parts.map((p) => (p.type === 'text' ? p.text : JSON.stringify(p))).join('\n');
      return { id: c.id, name: c.name, content: content || '(empty result)', isError: Boolean(r.isError) };
    } catch (e) {
      return { id: c.id, name: c.name, content: (e as Error).message, isError: true };
    }
  }

  /** Runs model steps until the model stops or a tool call needs the user's approval. */
  async function loop(chatId: string, p: ProviderRow, emit: Emit) {
    const { tools, clients, errors } = await mcp.tools();
    for (const e of errors) emit({ type: 'note', text: `MCP ${e}` });
    try {
      for (let i = 0; i < MAX_STEPS; i++) {
        emit({ type: 'start', alias: p.alias, model: p.model });
        const out = await steps[p.kind]({
          apiKey: box.open(p.api_key_enc),
          model: p.model,
          system: system(p.alias, p.model),
          history: await history(chatId),
          tools,
          onText: (delta) => emit({ type: 'text', delta }),
        });
        const [row] = await sql<Row[]>`
          insert into messages (chat_id, role, alias, kind, model, text, tool_calls, raw)
          values (${chatId}, 'assistant', ${p.alias}, ${p.kind}, ${p.model}, ${out.text},
                  ${sql.json(out.calls)}, ${out.raw === null ? null : sql.json(out.raw as never)})
          returning id, role, alias, kind, text, tool_calls, tool_results, raw`;
        await sql`update chats set updated_at = now() where id = ${chatId}`;
        if (out.note) emit({ type: 'note', text: out.note });
        if (!out.calls.length || !row) return;
        emit({ type: 'calls', calls: out.calls });

        const byName = new Map(tools.map((t) => [t.name, t]));
        const ask = out.calls.filter((c) => {
          const t = byName.get(c.name);
          return t && !t.server.always_allow.includes(t.tool);
        });
        if (ask.length) {
          emit({
            type: 'confirm',
            calls: ask.map((c) => {
              const t = byName.get(c.name) as Tool;
              return { ...c, server: t.server.name, tool: t.tool };
            }),
          });
          return;
        }
        const results = await Promise.all(out.calls.map((c) => call(byName.get(c.name), clients, c)));
        await addTool(chatId, row, results);
        emit({ type: 'results', results });
      }
      emit({ type: 'note', text: `Stopped after ${MAX_STEPS} steps.` });
    } finally {
      await Promise.all([...clients.values()].map((c) => c.close().catch(() => {})));
    }
  }

  async function guard(emit: Emit, f: () => Promise<void>) {
    try {
      await f();
    } catch (e) {
      emit({ type: 'error', message: (e as Error).message });
    }
    emit({ type: 'done' });
  }

  return {
    /** A new user message. `@alias` picks the model; otherwise the chat's last one is used. */
    send: (chatId: string, text: string, emit: Emit) =>
      guard(emit, async () => {
        const all = await providers();
        if (!all.length) throw new Error('Add an API provider in Settings first.');
        const rows = await history(chatId);
        const mentioned = [...text.matchAll(MENTION)].map((m) => m[1]).find((a) => all.some((p) => p.alias === a));
        const lastAlias = rows.findLast((r) => r.role !== 'tool' && r.alias)?.alias;
        const p = all.find((x) => x.alias === mentioned) ?? all.find((x) => x.alias === lastAlias) ?? all[0];
        if (!p) throw new Error('no provider');
        const open = pending(rows);
        if (open) {
          await addTool(chatId, open, open.tool_calls.map((c) => ({
            id: c.id, name: c.name, content: 'Not run: the user sent a new message instead.', isError: true,
          })));
        }
        await sql`insert into messages (chat_id, role, alias, text) values (${chatId}, 'user', ${p.alias}, ${text})`;
        if (!rows.length) await sql`update chats set title = ${text.replace(MENTION, ' ').trim().slice(0, 60) || 'New chat'} where id = ${chatId}`;
        await loop(chatId, p, emit);
      }),

    /** The user's answers to a confirm prompt; runs the approved calls and continues. */
    resume: (chatId: string, decisions: Record<string, Decision>, emit: Emit) =>
      guard(emit, async () => {
        const open = pending(await history(chatId));
        if (!open) throw new Error('Nothing is waiting for approval.');
        const p = (await providers()).find((x) => x.alias === open.alias);
        if (!p) throw new Error(`Provider @${open.alias} was removed.`);
        const { tools, clients } = await mcp.tools();
        try {
          const byName = new Map(tools.map((t) => [t.name, t]));
          const results = await Promise.all(
            open.tool_calls.map(async (c) => {
              const t = byName.get(c.name);
              const d = t?.server.always_allow.includes(t.tool) ? 'allow' : (decisions[c.id] ?? 'deny');
              if (d === 'deny') return { id: c.id, name: c.name, content: 'The user denied this tool call.', isError: true };
              if (d === 'always' && t) {
                await sql`update mcp_servers set always_allow = array_append(always_allow, ${t.tool})
                          where id = ${t.server.id} and not (${t.tool} = any(always_allow))`;
              }
              return call(t, clients, c);
            }),
          );
          await addTool(chatId, open, results);
          emit({ type: 'results', results });
        } finally {
          await Promise.all([...clients.values()].map((c) => c.close().catch(() => {})));
        }
        await loop(chatId, p, emit);
      }),
  };
}
