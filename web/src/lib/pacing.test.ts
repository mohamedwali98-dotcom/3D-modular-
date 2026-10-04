import { describe, expect, it } from 'vitest';
import type { Job, Stage, StageKey, StageState } from '../api/types';
import { MIN, pace } from './pacing';

const KEYS: StageKey[] = ['label', 'outline', 'read', 'draw', 'fuse'];

function job(states: StageState[], reads = 3): Job {
  const stages: Stage[] = KEYS.map((key, i) => ({
    key, state: states[i], tool: 'x', ai: false, detail: '',
    started: states[i] === 'pending' ? null : 1000, ended: states[i] === 'done' ? 1000 : null,
  }));
  return {
    job_id: 'j', status: states.every((s) => s === 'done') ? 'done' : 'running', stages,
    images: [{
      index: 0, width: 100, height: 100, face: 'front', kind: 'sketch',
      outline: [[0, 0], [10, 0], [10, 10]], circles: [],
      reads: Array.from({ length: reads }, () => ({ text: '50', value_mm: 50, kind: 'linear' as const, bbox: [0, 0, 1, 1] as [number, number, number, number], confidence: 0.9 })),
    }],
    coverage: { front: 'observed', back: 'empty', left: 'empty', right: 'qwen-image', top: 'empty', bottom: 'empty' },
    result: null, error: null,
  };
}

const at = (j: Job, s: number) => pace(j, 0, s * 1000);

describe('pace', () => {
  it('(a) a job done at t=0 plays label first and finishes at the sum of the minimums', () => {
    const j = job(['done', 'done', 'done', 'done', 'done'], 3);
    const p0 = at(j, 0.3);
    expect(p0.stages[0].state).toBe('running');
    expect(p0.stages.slice(1).every((s) => s.state === 'pending')).toBe(true);
    expect(p0.allDone).toBe(false);

    const sum = MIN.label + MIN.outline + Math.max(MIN.readMin, 3 * MIN.readEach) + MIN.draw + MIN.fuse;
    expect(at(j, sum - 0.05).allDone).toBe(false);
    const end = at(j, sum);
    expect(end.allDone).toBe(true);
    expect(end.stages.every((s) => s.state === 'done')).toBe(true);
    expect(end.trace[0]).toBe(1);
    expect(end.readsShown[0]).toBe(3);
    expect(end.faces.right).toBe('ai');
  });

  it('(b) a job still running read never shows draw as started', () => {
    const j = job(['done', 'done', 'running', 'pending', 'pending'], 0);
    for (const s of [0, 1, 5, 30, 600]) {
      const p = at(j, s);
      expect(p.stages[3].state).toBe('pending');
      expect(p.stages[4].state).toBe('pending');
      expect(p.allDone).toBe(false);
    }
    expect(at(j, 30).stages[2].state).toBe('running');
  });
});

describe('pace on a drawing sheet', () => {
  const crop = (index: number, face: 'front' | 'top' | 'right') => ({
    index, width: 100, height: 100, face, kind: 'drawing' as const, outline: [[0, 0], [10, 0], [10, 10]] as [number, number][],
    circles: [], reads: [],
  });
  const sheet: Job = {
    job_id: 's', status: 'done',
    stages: (['views', 'lines', 'values', 'draw', 'fuse'] as StageKey[]).map((key) => ({
      key, state: 'done' as StageState, tool: 'x', ai: false, detail: '', started: 1000, ended: 1000,
    })),
    images: [crop(0, 'front'), crop(1, 'top'), crop(2, 'right')],
    coverage: { front: 'observed', top: 'observed', right: 'observed', back: 'empty', left: 'empty', bottom: 'inferred' },
    result: null, error: null,
  };
  it('shows the views it read as observed, though a sheet has no outline stage', () => {
    const end = pace(sheet, 0, 600_000);
    expect(end.faces.front).toBe('photo');
    expect(end.faces.top).toBe('photo');
    expect(end.trace).toEqual([1, 1, 1]);
  });
  it('names a turned view inferred from the others instead of leaving it empty', () => {
    expect(pace(sheet, 0, 600_000).faces.bottom).toBe('inferred');
  });
});
