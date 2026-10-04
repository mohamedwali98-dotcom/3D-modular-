import { describe, expect, it } from 'vitest';
import { GIVE_UP_MS, POLL_MS, nextPoll } from './poll';

describe('nextPoll', () => {
  it('polls quickly while the server answers', () => {
    expect(nextPoll(0, 0)).toBe(POLL_MS);
  });
  it('backs off while the network is down, up to a few seconds', () => {
    const delays = [1, 2, 3, 4, 5, 8].map((fails) => nextPoll(fails, 1000));
    expect(delays).toEqual([...delays].sort((a, b) => (a ?? 0) - (b ?? 0)));
    expect(Math.max(...delays.map((d) => d ?? 0))).toBeLessThanOrEqual(5000);
  });
  it('gives up on time, not on a count: a lift ride of a minute is survived', () => {
    expect(nextPoll(40, 60_000)).not.toBeNull();
    expect(nextPoll(3, GIVE_UP_MS)).toBeNull();
  });
});
