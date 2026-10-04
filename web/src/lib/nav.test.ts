import { describe, expect, it } from 'vitest';
import type { Job } from '../api/types';
import { initialState, reducer, type State } from '../state/store';
import { buildNav } from './nav';

function job(status: Job['status']): Job {
  return {
    job_id: 'j1', status, stages: [], images: [],
    coverage: { front: 'empty', back: 'empty', left: 'empty', right: 'empty', top: 'empty', bottom: 'empty' },
    result: null, error: status === 'failed' ? 'boom' : null,
  };
}

const analyzing = (patch: Partial<State> = {}): State => ({ ...initialState, screen: 'analyzing', jobId: 'j1', ...patch });
const capture = (s: State) => buildNav(s)[0];

describe('buildNav capture item', () => {
  it('spins while a started job has not reported yet', () => {
    const c = capture(analyzing());
    expect(c.spinning).toBe(true);
    expect(c.sub).toBe('Analyzing…');
  });

  it('spins while the job runs', () => {
    expect(capture(analyzing({ job: job('running') })).spinning).toBe(true);
  });

  it('does not spin on the analyzing screen when no job was started', () => {
    const c = capture(analyzing({ jobId: null }));
    expect(c.spinning).toBe(false);
    expect(c.sub).not.toBe('Analyzing…');
  });

  it('does not spin once the job was lost, even though it never reported', () => {
    const s = reducer(analyzing(), { type: 'JOB_LOST', error: 'Could not reach the server.' });
    const c = capture(s);
    expect(c.spinning).toBe(false);
    expect(c.sub).toBe('Analysis lost');
    expect(c.subC).toBe('var(--stop)');
  });

  it('does not spin after the job failed', () => {
    const c = capture(analyzing({ job: job('failed') }));
    expect(c.spinning).toBe(false);
    expect(c.sub).toBe('Analysis failed');
  });

  it('a new job clears the lost flag', () => {
    const lost = reducer(analyzing(), { type: 'JOB_LOST', error: 'x' });
    const again = reducer(lost, { type: 'START_JOB', jobId: 'j2' });
    expect(capture(again).spinning).toBe(true);
  });
});

describe('buildNav model item', () => {
  it('never says Built for a build that stopped', () => {
    const spec = { version: 'mv1', envelope: { x_mm: 1, y_mm: 1, z_mm: 1 }, features: [], provenance: {} } as unknown as State['modelSpec'];
    const analysis = { request_id: 'j1', spec, abstain: null, filled_by: {} } as unknown as State['analysis'];
    const model = { key: null, abstain: { stage: 'build', reason: 'x', remedy: 'r', partial: null } } as unknown as State['model'];
    const item = buildNav({ ...initialState, screen: 'review', analysis, model })[2];
    expect(item.sub).not.toBe('Built');
  });
});
