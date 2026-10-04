// The server's access token (S2C_ACCESS_TOKEN), asked for once and kept in this browser.
export const TOKEN_KEY = 's2c-token';

export function storedToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}

export function storeToken(token: string): void {
  try { localStorage.setItem(TOKEN_KEY, token); } catch { /* storage blocked: asked again next time */ }
}

export function withToken(init: RequestInit | undefined, token: string | null): RequestInit {
  if (!token) return init ?? {};
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${token}`);
  return { ...init, headers };
}

let asking: Promise<string | null> | null = null;
let declinedAt: number | null = null;
const DECLINE_MS = 15_000; // a dismissed prompt covers the burst of requests behind it, not the rest of the visit

/** The token after asking for it: one prompt for every request refused at the same time (polling never stacks
 * prompts), a dismissal remembered for the burst behind it (asked again after DECLINE_MS, so an Esc never locks
 * the app until a reload), and no prompt at all when the server has no token (`reason` "no_token": only this
 * computer is served, so no token would help). */
export function askToken(message: string, ask: (text: string) => string | null = (text) => window.prompt(text),
  reason?: string, now: number = Date.now()): Promise<string | null> {
  if (reason === 'no_token' || (declinedAt !== null && now - declinedAt < DECLINE_MS)) return Promise.resolve(null);
  asking ??= Promise.resolve().then(() => {
    const token = ask(`${message}\n\nAccess token:`)?.trim() || null;
    if (token) storeToken(token);
    declinedAt = token ? null : now;
    return token;
  }).finally(() => { asking = null; });
  return asking;
}

/** For tests: forget a refusal and any prompt in flight. */
export function resetTokenPrompt(): void {
  asking = null;
  declinedAt = null;
}
