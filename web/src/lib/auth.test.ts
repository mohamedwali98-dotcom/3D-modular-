import { describe, expect, it } from 'vitest';
import { askToken, resetTokenPrompt, withToken } from './auth';

describe('access token', () => {
  it('sends the token as a bearer header, keeping the other headers', () => {
    const init = withToken({ method: 'POST', headers: { 'Content-Type': 'application/json' } }, 's3cret');
    const headers = new Headers(init.headers);
    expect(headers.get('Authorization')).toBe('Bearer s3cret');
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(init.method).toBe('POST');
  });
  it('leaves the request as it is without a token', () => {
    const init = { method: 'GET' };
    expect(withToken(init, null)).toBe(init);
  });
});

describe('asking for the access token', () => {
  it('asks once for every request refused at the same time, and remembers a refusal', async () => {
    resetTokenPrompt();
    let asked = 0;
    const ask = () => { asked += 1; return null; };
    const [a, b] = await Promise.all([askToken('Enter the access token.', ask), askToken('Enter the access token.', ask)]);
    expect([a, b, asked]).toEqual([null, null, 1]);
    expect(await askToken('Enter the access token.', ask)).toBeNull();
    expect(asked).toBe(1);
  });
  it('never asks for a token the server does not have', async () => {
    resetTokenPrompt();
    let asked = 0;
    expect(await askToken('Set S2C_ACCESS_TOKEN', () => { asked += 1; return 'x'; }, 'no_token')).toBeNull();
    expect(asked).toBe(0);
  });
});

describe('a dismissed token prompt', () => {
  it('is remembered for a burst of requests, then asked again: an Esc never locks the app until a reload', async () => {
    resetTokenPrompt();
    let asked = 0;
    const ask = () => { asked += 1; return null; };
    await askToken('Enter the access token.', ask, undefined, 0);
    await askToken('Enter the access token.', ask, undefined, 5_000);
    expect(asked).toBe(1);
    await askToken('Enter the access token.', ask, undefined, 20_000);
    expect(asked).toBe(2);
  });
});
