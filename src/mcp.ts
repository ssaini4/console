import { randomBytes } from 'node:crypto';
import { auth, type OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type {
  OAuthClientInformationMixed,
  OAuthTokens,
} from '@modelcontextprotocol/sdk/shared/auth.js';
import type { Sql } from 'postgres';
import type { SecretBox } from './crypto';

export type McpRow = {
  id: string;
  name: string;
  url: string;
  enabled: boolean;
  headers_enc: string | null;
  oauth_enc: string | null;
  always_allow: string[];
};
type OAuthState = {
  client?: OAuthClientInformationMixed;
  tokens?: OAuthTokens;
  verifier?: string;
  nonce?: string;
};

/** A tool offered to models: `server__tool`, mapped back to the MCP server on call. */
export type Tool = {
  name: string;
  server: McpRow;
  tool: string;
  description: string;
  schema: Record<string, unknown>;
};

export class NeedsAuth extends Error {}

export function mcpManager(sql: Sql, box: SecretBox, publicUrl: string) {
  const redirectUrl = `${publicUrl.replace(/\/$/, '')}/mcp-oauth/callback`;

  async function loadState(id: string): Promise<OAuthState> {
    const [r] = await sql<{ oauth_enc: string | null }[]>`
      select oauth_enc from mcp_servers where id = ${id}`;
    return r?.oauth_enc ? JSON.parse(box.open(r.oauth_enc)) : {};
  }
  async function patchState(id: string, patch: Partial<OAuthState>) {
    const next = { ...(await loadState(id)), ...patch };
    await sql`update mcp_servers set oauth_enc = ${box.seal(JSON.stringify(next))} where id = ${id}`;
  }

  /** OAuth provider backed by the row; `redirect` receives the authorize URL. */
  function provider(row: McpRow, redirect: (u: URL) => void = () => {}): OAuthClientProvider {
    return {
      redirectUrl,
      clientMetadata: {
        client_name: 'console',
        redirect_uris: [redirectUrl],
        grant_types: ['authorization_code', 'refresh_token'],
        response_types: ['code'],
        token_endpoint_auth_method: 'none',
      },
      state: async () => {
        const nonce = randomBytes(16).toString('base64url');
        await patchState(row.id, { nonce });
        return `${row.id}.${nonce}`;
      },
      clientInformation: async () => (await loadState(row.id)).client,
      saveClientInformation: (client) => patchState(row.id, { client }),
      tokens: async () => (await loadState(row.id)).tokens,
      saveTokens: (tokens) => patchState(row.id, { tokens }),
      saveCodeVerifier: (verifier) => patchState(row.id, { verifier }),
      codeVerifier: async () => {
        const v = (await loadState(row.id)).verifier;
        if (!v) throw new Error('no code verifier');
        return v;
      },
      redirectToAuthorization: (u) => redirect(u),
    };
  }

  async function connect(row: McpRow): Promise<Client> {
    const headers = row.headers_enc ? JSON.parse(box.open(row.headers_enc)) : {};
    let needsAuth = false;
    const transport = new StreamableHTTPClientTransport(new URL(row.url), {
      requestInit: { headers },
      authProvider: provider(row, () => {
        needsAuth = true;
      }),
    });
    const client = new Client({ name: 'console', version: '1.0.0' });
    try {
      await client.connect(transport);
    } catch (e) {
      if (needsAuth) throw new NeedsAuth(`${row.name} needs to be connected (OAuth)`);
      throw e;
    }
    return client;
  }

  return {
    connect,

    /** Starts OAuth; returns the URL to send the browser to, or null if already authorized. */
    async beginOAuth(row: McpRow): Promise<string | null> {
      let to: URL | null = null;
      const r = await auth(
        provider(row, (u) => {
          to = u;
        }),
        { serverUrl: row.url },
      );
      return r === 'REDIRECT' && to ? String(to) : null;
    },

    async finishOAuth(state: string, code: string): Promise<string> {
      const [id, nonce] = state.split('.');
      if (!id || !nonce) throw new Error('bad state');
      const [row] = await sql<McpRow[]>`select * from mcp_servers where id = ${id}`;
      if (!row || (await loadState(id)).nonce !== nonce) throw new Error('state mismatch');
      await auth(provider(row), { serverUrl: row.url, authorizationCode: code });
      await patchState(id, { nonce: undefined, verifier: undefined });
      return row.name;
    },

    /** Connects every enabled server and lists its tools. Failures are reported, not thrown. */
    async tools(): Promise<{ tools: Tool[]; clients: Map<string, Client>; errors: string[] }> {
      const rows = await sql<McpRow[]>`select * from mcp_servers where enabled order by id`;
      const tools: Tool[] = [];
      const clients = new Map<string, Client>();
      const errors: string[] = [];
      await Promise.all(
        rows.map(async (row) => {
          try {
            const c = await connect(row);
            clients.set(row.id, c);
            for (const t of (await c.listTools()).tools) {
              tools.push({
                name: toolName(row.name, t.name),
                server: row,
                tool: t.name,
                description: t.description ?? '',
                schema: t.inputSchema as Record<string, unknown>,
              });
            }
          } catch (e) {
            errors.push(`${row.name}: ${(e as Error).message}`);
          }
        }),
      );
      tools.sort((a, b) => a.name.localeCompare(b.name)); // stable order keeps prompt caches warm
      return { tools, clients, errors };
    },
  };
}

/** Provider-safe tool name: [a-zA-Z0-9_-], starts with a letter, at most 64 chars. */
export function toolName(server: string, tool: string): string {
  return `${server}__${tool}`.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/^[^a-zA-Z]/, 't').slice(0, 64);
}

export type McpManager = ReturnType<typeof mcpManager>;
