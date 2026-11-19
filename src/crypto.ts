import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto';

/** AES-256-GCM with a 32-byte hex MASTER_KEY. Output: iv.tag.ciphertext (base64url). */
export function secretBox(masterKeyHex: string) {
  const key = Buffer.from(masterKeyHex, 'hex');
  if (key.length !== 32) throw new Error('MASTER_KEY must be 64 hex characters');
  return {
    seal(plain: string): string {
      const iv = randomBytes(12);
      const c = createCipheriv('aes-256-gcm', key, iv);
      const ct = Buffer.concat([c.update(plain, 'utf8'), c.final()]);
      return [iv, c.getAuthTag(), ct].map((b) => b.toString('base64url')).join('.');
    },
    open(sealed: string): string {
      const [iv, tag, ct] = sealed.split('.').map((s) => Buffer.from(s, 'base64url'));
      if (!iv || !tag || !ct) throw new Error('bad ciphertext');
      const d = createDecipheriv('aes-256-gcm', key, iv);
      d.setAuthTag(tag);
      return Buffer.concat([d.update(ct), d.final()]).toString('utf8');
    },
  };
}
export type SecretBox = ReturnType<typeof secretBox>;
