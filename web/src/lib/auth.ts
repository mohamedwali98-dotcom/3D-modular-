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
let declined = false;

/** The token after asking for it: one prompt for every request refused at the same time (polling never stacks
 * prompts), a refusal remembered until the page reloads, and no prompt at all when the server has no token
 * (`reason` "no_token": only this computer is served, so no token would help). */
export function askToken(message: string, ask: (text: string) => string | null = (text) => window.prompt(text),
  reason?: string): Promise<string | null> {
  if (reason === 'no_token' || declined) return Promise.resolve(null);
  asking ??= Promise.resolve().then(() => {
    const token = ask(`${message}\n\nAccess token:`)?.trim() || null;
    if (token) storeToken(token);
    else declined = true;
    return token;
  }).finally(() => { asking = null; });
  return asking;
}

/** For tests: forget a refusal and any prompt in flight. */
export function resetTokenPrompt(): void {
  asking = null;
  declined = false;
}
