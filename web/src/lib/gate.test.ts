import { describe, expect, it } from 'vitest';
import { reviewGate } from './gate';

const base = { requestId: 'j1', pending: false, merging: false, mergeErr: null, building: false, invalid: false };

describe('reviewGate', () => {
  it('waits for a merge that is due or in flight', () => {
    expect(reviewGate({ ...base, pending: true })).toMatchObject({ updating: true, blocked: true, retry: false });
    expect(reviewGate({ ...base, merging: true })).toMatchObject({ updating: true, blocked: true });
  });
  it('offers a retry when the update failed', () => {
    expect(reviewGate({ ...base, pending: true, mergeErr: 'offline' })).toMatchObject({ retry: true, blocked: false });
  });
  it('never waits on a part described in the chat: it has no analysis to merge with', () => {
    expect(reviewGate({ ...base, requestId: null, pending: true })).toMatchObject({ updating: false, blocked: false });
  });
  it('holds Build while a typed value is invalid', () => {
    expect(reviewGate({ ...base, invalid: true }).blocked).toBe(true);
  });
});
