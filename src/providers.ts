import Anthropic from '@anthropic-ai/sdk';
import { type Content, GoogleGenAI, type Part } from '@google/genai';
import OpenAI from 'openai';
import type { Tool } from './mcp';

export type Kind = 'anthropic' | 'openai' | 'gemini';
export type Call = { id: string; name: string; args: Record<string, unknown> };
export type Result = { id: string; name: string; content: string; isError?: boolean };
export type Msg = {
  role: 'user' | 'assistant' | 'tool';
  alias: string | null;
  kind: string | null;
  text: string;
  tool_calls: Call[];
  tool_results: Result[];
  raw: unknown;
};
export type StepIn = {
  apiKey: string;
  model: string;
  system: string;
  history: Msg[];
  tools: Tool[];
  onText: (delta: string) => void;
};
export type StepOut = { text: string; calls: Call[]; raw: unknown; note?: string };

export const DEFAULT_MODEL: Record<Kind, string> = {
  anthropic: 'claude-opus-5-5',
  openai: '',
  gemini: '',
};

/** Another model's turn, as text this provider can read. */
function foreign(m: Msg): string {
  if (m.role === 'tool') {
    return m.tool_results.map((r) => `(tool ${r.name} returned: ${r.content})`).join('\n');
  }
  const calls = m.tool_calls.map((c) => `(called tool ${c.name} ${JSON.stringify(c.args)})`);
  return [`[@${m.alias}]: ${m.text}`, ...calls].join('\n');
}
const native = (m: Msg, kind: Kind) => m.role !== 'user' && m.kind === kind;

// ---------------------------------------------------------------- Anthropic

// Models that accept server-side refusal fallbacks.
const FALLBACKS = /^claude-(fable-5-[12]|opus-5|opus-5-5|sonnet-5-5)$/;

async function anthropicStep(p: StepIn): Promise<StepOut> {
  const client = new Anthropic({ apiKey: p.apiKey });
  const messages: Anthropic.Beta.BetaMessageParam[] = p.history.map((m) => {
    if (!native(m, 'anthropic')) {
      return { role: m.role === 'user' ? 'user' : m.role === 'assistant' ? 'assistant' : 'user', content: m.role === 'user' ? m.text : foreign(m) };
    }
    if (m.role === 'assistant') {
      // Replay our own content unchanged: thinking and fallback blocks must round-trip.
      return { role: 'assistant', content: m.raw as Anthropic.Beta.BetaContentBlockParam[] };
    }
    return {
      role: 'user',
      content: m.tool_results.map((r) => ({
        type: 'tool_result' as const,
        tool_use_id: r.id,
        content: r.content,
        is_error: r.isError ?? false,
      })),
    };
  });
  const tools: Anthropic.Beta.BetaToolUnion[] = p.tools.map((t) => ({
    name: t.name,
    description: t.description.slice(0, 1024),
    input_schema: t.schema as Anthropic.Beta.BetaTool.InputSchema,
    eager_input_streaming: true,
  }));
  const stream = client.beta.messages.stream({
    model: p.model,
    max_tokens: 64000,
    system: p.system,
    messages,
    ...(tools.length ? { tools } : {}),
    ...(FALLBACKS.test(p.model)
      ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' as const }
      : {}),
  });
  stream.on('text', p.onText);
  const msg = await stream.finalMessage();
  const calls = msg.content.flatMap((b) =>
    b.type === 'tool_use' ? [{ id: b.id, name: b.name, args: b.input as Record<string, unknown> }] : [],
  );
  const text = msg.content.flatMap((b) => (b.type === 'text' ? [b.text] : [])).join('');
  if (msg.stop_reason === 'refusal') return { text, calls: [], raw: msg.content, note: 'The model declined this request.' };
  // A tool input cut off at max_tokens can still look valid; never run it.
  if (msg.stop_reason === 'max_tokens' && calls.length) {
    return { text, calls: [], raw: msg.content.filter((b) => b.type !== 'tool_use'), note: 'Output limit reached mid tool call.' };
  }
  return { text, calls, raw: msg.content };
}

// ---------------------------------------------------------------- OpenAI

async function openaiStep(p: StepIn): Promise<StepOut> {
  const client = new OpenAI({ apiKey: p.apiKey });
  const messages: OpenAI.ChatCompletionMessageParam[] = [{ role: 'system', content: p.system }];
  for (const m of p.history) {
    if (!native(m, 'openai')) {
      messages.push({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.role === 'user' ? m.text : foreign(m) });
    } else if (m.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: m.text || null,
        ...(m.tool_calls.length
          ? {
              tool_calls: m.tool_calls.map((c) => ({
                id: c.id,
                type: 'function' as const,
                function: { name: c.name, arguments: JSON.stringify(c.args) },
              })),
            }
          : {}),
      });
    } else {
      for (const r of m.tool_results) messages.push({ role: 'tool', tool_call_id: r.id, content: r.content });
    }
  }
  const tools: OpenAI.ChatCompletionTool[] = p.tools.map((t) => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.schema },
  }));
  const stream = client.chat.completions.stream({
    model: p.model,
    messages,
    ...(tools.length ? { tools } : {}),
  });
  stream.on('content.delta', (e) => p.onText(e.delta));
  const done = await stream.finalChatCompletion();
  const msg = done.choices[0]?.message;
  const calls = (msg?.tool_calls ?? []).flatMap((c) => {
    if (c.type !== 'function') return [];
    let args: Record<string, unknown> = {};
    try {
      args = JSON.parse(c.function.arguments || '{}');
    } catch {}
    return [{ id: c.id, name: c.function.name, args }];
  });
  return { text: msg?.content ?? '', calls, raw: null, note: msg?.refusal ?? undefined };
}

// ---------------------------------------------------------------- Gemini

async function geminiStep(p: StepIn): Promise<StepOut> {
  const ai = new GoogleGenAI({ apiKey: p.apiKey });
  const contents: Content[] = [];
  const push = (role: 'user' | 'model', parts: Part[]) => {
    const last = contents.at(-1);
    if (last?.role === role) last.parts = [...(last.parts ?? []), ...parts];
    else contents.push({ role, parts });
  };
  for (const m of p.history) {
    if (!native(m, 'gemini')) push(m.role === 'assistant' ? 'model' : 'user', [{ text: m.role === 'user' ? m.text : foreign(m) }]);
    // Our own parts carry thought signatures that must be sent back as-is.
    else if (m.role === 'assistant') push('model', m.raw as Part[]);
    else {
      push(
        'user',
        m.tool_results.map((r) => ({
          functionResponse: {
            id: r.id,
            name: r.name,
            response: r.isError ? { error: r.content } : { output: r.content },
          },
        })),
      );
    }
  }
  const stream = await ai.models.generateContentStream({
    model: p.model,
    contents,
    config: {
      systemInstruction: p.system,
      ...(p.tools.length
        ? {
            tools: [
              {
                functionDeclarations: p.tools.map((t) => ({
                  name: t.name,
                  description: t.description,
                  parametersJsonSchema: t.schema,
                })),
              },
            ],
          }
        : {}),
    },
  });
  const parts: Part[] = [];
  let text = '';
  let note: string | undefined;
  for await (const chunk of stream) {
    const cand = chunk.candidates?.[0];
    for (const part of cand?.content?.parts ?? []) {
      parts.push(part);
      if (part.text && !part.thought) {
        text += part.text;
        p.onText(part.text);
      }
    }
    if (cand?.finishReason && cand.finishReason !== 'STOP') note = `Stopped: ${cand.finishReason}`;
  }
  const calls = parts.flatMap((part, i) =>
    part.functionCall?.name
      ? [{ id: part.functionCall.id ?? `call_${i}`, name: part.functionCall.name, args: part.functionCall.args ?? {} }]
      : [],
  );
  return { text, calls, raw: parts, note: calls.length ? undefined : note };
}

export const steps: Record<Kind, (p: StepIn) => Promise<StepOut>> = {
  anthropic: anthropicStep,
  openai: openaiStep,
  gemini: geminiStep,
};

/** Model ids the key can use; also validates the key. */
export async function listModels(kind: Kind, apiKey: string): Promise<string[]> {
  const ids: string[] = [];
  if (kind === 'anthropic') {
    for await (const m of new Anthropic({ apiKey }).models.list()) ids.push(m.id);
  } else if (kind === 'openai') {
    for await (const m of new OpenAI({ apiKey }).models.list()) ids.push(m.id);
  } else {
    for await (const m of await new GoogleGenAI({ apiKey }).models.list()) {
      if (m.name && m.supportedActions?.includes('generateContent')) ids.push(m.name.replace(/^models\//, ''));
    }
  }
  return ids.sort();
}
