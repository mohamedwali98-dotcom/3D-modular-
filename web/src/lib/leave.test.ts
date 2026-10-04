import { describe, expect, it } from 'vitest';
import type { Job } from '../api/types';
import { onLeave } from './leave';

const job = (status: Job['status'], result: Job['result'] = null) => ({ status, result }) as Job;

describe('onLeave', () => {
  it('keeps a finished analysis even when Cancel is pressed before its playback ends', () => {
    expect(onLeave(job('done', { request_id: 'j', spec: null, abstain: null, filled_by: {} }))).toBe('hand_over');
  });
  it('cancels a job still running, or one that has not reported yet', () => {
    expect(onLeave(job('running'))).toBe('cancel');
    expect(onLeave(null)).toBe('cancel');
  });
  it('just leaves a job that already stopped', () => {
    expect(onLeave(job('failed'))).toBe('leave');
  });
});
