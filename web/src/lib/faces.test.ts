import { describe, expect, it } from 'vitest';
import { canReject, faceInfo } from './faces';

describe('faces', () => {
  it('names every face source the server sends, and any it may send later', () => {
    for (const by of ['observed', 'qwen-image', 'triposr', 'mirrored', 'assumed', 'inferred', 'not-yet-known']) {
      expect(faceInfo(by).badge.length).toBeGreaterThan(0);
    }
    expect(faceInfo('inferred').badge).toMatch(/turned/i);
  });
  it('offers Reject only where rejecting changes the part', () => {
    expect(canReject('qwen-image')).toBe(true);
    expect(canReject('triposr')).toBe(true);
    for (const by of ['observed', 'assumed', 'mirrored', 'inferred']) expect(canReject(by)).toBe(false);
  });
});
