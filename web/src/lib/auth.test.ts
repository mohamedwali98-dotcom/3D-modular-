import { describe, expect, it } from 'vitest';
import { withToken } from './auth';

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
