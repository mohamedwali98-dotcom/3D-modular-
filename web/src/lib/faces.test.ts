import { describe, expect, it } from 'vitest';
import { canReject, faceInfo, rejectAction } from './faces';

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

describe('rejectAction', () => {
  it('keeps Undo on a rejected face, whatever the server refilled it with', () => {
    expect(rejectAction('assumed', true)).toBe('undo');
    expect(rejectAction('triposr', true)).toBe('undo');
  });
  it('offers Reject only on a face an AI drew, and nothing on the others', () => {
    expect(rejectAction('qwen-image', false)).toBe('reject');
    expect(rejectAction('assumed', false)).toBeNull();
    expect(rejectAction('observed', false)).toBeNull();
  });
});
