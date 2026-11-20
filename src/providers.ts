import Anthropic from '@anthropic-ai/sdk';
import { type Content, GoogleGenAI, Modality, type Part } from '@google/genai';
import OpenAI from 'openai';

export type Kind = 'anthropic' | 'openai' | 'gemini';
export type Call = { id: string; name: string; args: Record<string, unknown> };
export type Result = { id: string; name: string; content: string; isError?: boolean; image?: string };
/** What a model sees of a tool. */
export type ToolSpec = { name: string; description: string; schema: Record<string, unknown> };
/** A file the user attached, with its bytes loaded. */
export type FileData = { name: string; mime: string; data: Uint8Array };
export type Msg = {
  role: 'user' | 'assistant' | 'tool';
  alias: string | null;
  kind: string | null;
  text: string;
  tool_calls: Call[];
  tool_results: Result[];
  raw: unknown;
  /** User messages only: attached files. */
  files?: FileData[];
};
export type StepIn = {
  apiKey: string;
  model: string;
  system: string;
  history: Msg[];
  tools: ToolSpec[];
  onText: (delta: string) => void;
  onThinking: (delta: string) => void;
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

// ---------------------------------------------------------------- attachments

const IMAGE_MIMES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const MAX_TEXT = 200_000;
/** How a provider can take a file: as an image, a PDF, other media (Gemini only), or decoded text. */
export type Piece =
  | { kind: 'text'; text: string }
  | { kind: 'image' | 'pdf' | 'media'; file: FileData; b64: string };

function asText(f: FileData): string | null {
  try {
    const t = new TextDecoder('utf-8', { fatal: true }).decode(f.data);
    return t.includes('\0') ? null : t;
  } catch {
    return null;
  }
}

/** A user message's files as provider-neutral pieces, followed by its text. */
export function pieces(m: Msg): Piece[] {
  const out: Piece[] = [];
  for (const f of m.files ?? []) {
    const b64 = () => Buffer.from(f.data).toString('base64');
    if (IMAGE_MIMES.includes(f.mime)) out.push({ kind: 'image', file: f, b64: b64() });
    else if (f.mime === 'application/pdf') out.push({ kind: 'pdf', file: f, b64: b64() });
    else if (/^(audio|video)\//.test(f.mime)) out.push({ kind: 'media', file: f, b64: b64() });
    else {
      const t = asText(f);
      const body = t === null
        ? `(${f.mime || 'binary'} file, ${f.data.length} bytes: its contents can't be read as text)`
        : t.length > MAX_TEXT ? `${t.slice(0, MAX_TEXT)}\n… (truncated)` : t;
      out.push({ kind: 'text', text: `<file name="${f.name.replace(/"/g, "'")}">\n${body}\n</file>` });
    }
  }
  if (m.text) out.push({ kind: 'text', text: m.text });
  return out;
}
const unreadable = (p: Piece & { file: FileData }) =>
  `(attached ${p.file.name}, ${p.file.mime}: this model can't read that file type)`;

// ---------------------------------------------------------------- Anthropic

// Models that accept server-side refusal fallbacks.
const FALLBACKS = /^claude-(fable-5-[12]|opus-5|opus-5-5|sonnet-5-5)$/;
// Models that take adaptive thinking; older ones 400 on it.
const ADAPTIVE = /^claude-(fable|mythos|opus-5|opus-4-[678]|sonnet-5|sonnet-4-6|haiku-5)/;

async function anthropicStep(p: StepIn): Promise<StepOut> {
  const client = new Anthropic({ apiKey: p.apiKey });
  const messages: Anthropic.Beta.BetaMessageParam[] = p.history.map((m) => {
    if (m.role === 'user' && m.files?.length) {
      const content: Anthropic.Beta.BetaContentBlockParam[] = pieces(m).map((x) =>
        x.kind === 'image'
          ? { type: 'image', source: { type: 'base64', media_type: x.file.mime as 'image/png', data: x.b64 } }
          : x.kind === 'pdf'
            ? { type: 'document', title: x.file.name, source: { type: 'base64', media_type: 'application/pdf', data: x.b64 } }
            : { type: 'text', text: x.kind === 'text' ? x.text : unreadable(x) },
      );
      return { role: 'user', content };
    }
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
    // Default display is "omitted" (empty text); ask for readable summaries to show the user.
    ...(ADAPTIVE.test(p.model) ? { thinking: { type: 'adaptive' as const, display: 'summarized' as const } } : {}),
    ...(FALLBACKS.test(p.model)
      ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' as const }
      : {}),
  });
  stream.on('text', p.onText);
  stream.on('thinking', (delta) => p.onThinking(delta));
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
    if (m.role === 'user' && m.files?.length) {
      messages.push({
        role: 'user',
        content: pieces(m).map((x): OpenAI.ChatCompletionContentPart =>
          x.kind === 'image'
            ? { type: 'image_url', image_url: { url: `data:${x.file.mime};base64,${x.b64}` } }
            : x.kind === 'pdf'
              ? { type: 'file', file: { filename: x.file.name, file_data: `data:application/pdf;base64,${x.b64}` } }
              : { type: 'text', text: x.kind === 'text' ? x.text : unreadable(x) },
        ),
      });
    } else if (!native(m, 'openai')) {
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
    if (m.role === 'user' && m.files?.length) {
      push(
        'user',
        pieces(m).map((x) => (x.kind === 'text' ? { text: x.text } : { inlineData: { mimeType: x.file.mime, data: x.b64 } })),
      );
    } else if (!native(m, 'gemini')) push(m.role === 'assistant' ? 'model' : 'user', [{ text: m.role === 'user' ? m.text : foreign(m) }]);
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
      // ponytail: name-based guard; older models reject thinkingConfig.
      ...(/gemini-(2\.5|[3-9])/.test(p.model) ? { thinkingConfig: { includeThoughts: true } } : {}),
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
      if (part.text && part.thought) p.onThinking(part.text);
      else if (part.text) {
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

/** One image from the provider's image model. Anthropic has none. */
export async function generateImage(kind: Kind, apiKey: string, prompt: string): Promise<{ mime: string; data: Buffer }> {
  // ponytail: picks the alphabetically last matching model id (≈ newest); add a setting if that picks wrong.
  const pick = async (re: RegExp) => {
    const model = (await cachedModels(kind, apiKey)).filter((m) => re.test(m)).at(-1);
    if (!model) throw new Error(`No image model available for this ${kind} key.`);
    return model;
  };
  if (kind === 'openai') {
    const r = await new OpenAI({ apiKey }).images.generate({ model: await pick(/^gpt-image/), prompt });
    const b64 = r.data?.[0]?.b64_json;
    if (!b64) throw new Error('The image model returned no image.');
    return { mime: 'image/png', data: Buffer.from(b64, 'base64') };
  }
  if (kind === 'gemini') {
    const r = await new GoogleGenAI({ apiKey }).models.generateContent({
      model: await pick(/^gemini-.*image/),
      contents: prompt,
      config: { responseModalities: [Modality.TEXT, Modality.IMAGE] },
    });
    const img = r.candidates?.[0]?.content?.parts?.find((p) => p.inlineData?.data)?.inlineData;
    if (!img?.data) throw new Error(r.text || 'The image model returned no image.');
    return { mime: img.mimeType ?? 'image/png', data: Buffer.from(img.data, 'base64') };
  }
  throw new Error(`${kind} cannot generate images.`);
}

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

const modelCache = new Map<string, { until: number; ids: Promise<string[]> }>();
/** listModels, remembered for 10 minutes per key. */
function cachedModels(kind: Kind, apiKey: string): Promise<string[]> {
  const k = `${kind}:${apiKey}`;
  const hit = modelCache.get(k);
  if (hit && hit.until > Date.now()) return hit.ids;
  const ids = listModels(kind, apiKey);
  modelCache.set(k, { until: Date.now() + 10 * 60_000, ids });
  ids.catch(() => modelCache.delete(k));
  return ids;
}

/** The newest plain Gemini Flash model the key can use: highest version, stable before preview. */
export function pickFlash(ids: string[]): string | undefined {
  const flash = ids.flatMap((id) => {
    const m = /^gemini-(\d+(?:\.\d+)?)-flash(-preview(?:-[\w-]+)?)?$/.exec(id);
    return m ? [{ id, version: Number(m[1]), preview: Boolean(m[2]) }] : [];
  });
  flash.sort((a, b) => b.version - a.version || Number(a.preview) - Number(b.preview));
  return flash[0]?.id;
}

/** Speech to text with a Gemini key. `audio` is base64. */
export async function transcribe(apiKey: string, mime: string, audio: string): Promise<string> {
  const model = pickFlash(await cachedModels('gemini', apiKey));
  if (!model) throw new Error('This Gemini key has no Flash model for transcription.');
  const r = await new GoogleGenAI({ apiKey }).models.generateContent({
    model,
    contents: [
      {
        role: 'user',
        parts: [
          { inlineData: { mimeType: mime, data: audio } },
          {
            text: 'Transcribe this audio verbatim, in the language spoken. Reply with only the transcript: no preamble, labels, timestamps or quotes. If there is no speech, reply with nothing.',
          },
        ],
      },
    ],
  });
  return (r.text ?? '').trim();
}
