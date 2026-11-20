import { expect, test } from 'bun:test';
import { type Msg, pickFlash, pieces } from './providers';

test('pickFlash takes the newest plain Flash model, stable before preview', () => {
  const ids = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-flash-image', 'gemini-2.5-flash-preview-tts', 'gemini-2.5-pro'];
  expect(pickFlash(ids)).toBe('gemini-2.5-flash');
  expect(pickFlash([...ids, 'gemini-3-flash-preview'])).toBe('gemini-3-flash-preview');
  expect(pickFlash([...ids, 'gemini-3-flash-preview', 'gemini-3-flash'])).toBe('gemini-3-flash');
  expect(pickFlash(['gemini-2.5-pro'])).toBeUndefined();
});

test('pieces: images and PDFs stay binary, text files are inlined, binaries are described', () => {
  const enc = (s: string) => new TextEncoder().encode(s);
  const m: Msg = {
    role: 'user', alias: 'a', kind: null, text: 'look', tool_calls: [], tool_results: [], raw: null,
    files: [
      { name: 'a.png', mime: 'image/png', data: new Uint8Array([1, 2]) },
      { name: 'r.pdf', mime: 'application/pdf', data: enc('%PDF') },
      { name: 'x.ts', mime: '', data: enc('const x = 1;') },
      { name: 'b.bin', mime: 'application/octet-stream', data: new Uint8Array([0xff, 0xfe, 0]) },
      { name: 'v.mp3', mime: 'audio/mpeg', data: new Uint8Array([1]) },
    ],
  };
  const p = pieces(m);
  expect(p.map((x) => x.kind)).toEqual(['image', 'pdf', 'text', 'text', 'media', 'text']);
  expect(p[0]).toMatchObject({ b64: 'AQI=' });
  expect(p[2]).toMatchObject({ text: '<file name="x.ts">\nconst x = 1;\n</file>' });
  expect((p[3] as { text: string }).text).toContain("can't be read as text");
  expect(p.at(-1)).toEqual({ kind: 'text', text: 'look' });
});
