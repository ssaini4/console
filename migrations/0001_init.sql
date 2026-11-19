-- API providers. The key is AES-256-GCM encrypted with MASTER_KEY (see src/crypto.ts).
create table providers (
  id bigserial primary key,
  alias text not null unique check (alias ~ '^[a-z0-9_-]{1,32}$'),
  kind text not null check (kind in ('anthropic', 'openai', 'gemini')),
  model text not null,
  api_key_enc text not null,
  key_hint text not null,
  created_at timestamptz not null default now()
);

-- Remote MCP servers. Headers and OAuth state are encrypted JSON blobs.
create table mcp_servers (
  id bigserial primary key,
  name text not null unique check (name ~ '^[a-zA-Z0-9_-]{1,32}$'),
  url text not null,
  enabled boolean not null default true,
  headers_enc text,
  oauth_enc text,
  always_allow text[] not null default '{}',
  created_at timestamptz not null default now()
);

create table chats (
  id bigserial primary key,
  title text not null default 'New chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Provider-neutral history. `raw` keeps the provider's own assistant content
-- (thinking blocks, thought signatures) so it can be replayed unchanged to the
-- same provider; other providers see text.
create table messages (
  id bigserial primary key,
  chat_id bigint not null references chats(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'tool')),
  alias text,
  kind text,
  model text,
  text text not null default '',
  tool_calls jsonb not null default '[]',
  tool_results jsonb not null default '[]',
  raw jsonb,
  created_at timestamptz not null default now()
);
create index messages_chat on messages (chat_id, id);
