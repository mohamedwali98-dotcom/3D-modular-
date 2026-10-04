import { describe, expect, it } from 'vitest';
import type { Analysis, ModelResult, Spec } from '../api/types';
import { initialAi, initialState, reducer, type CaptureItem, type State } from './store';

const item = (id: string, face: CaptureItem['face'] = 'front'): CaptureItem =>
  ({ id, file: new Blob() as File, url: `blob:${id}`, face, kind: 'sketch' });
const analysis = (request_id: string): Analysis => ({ request_id, spec: null, abstain: null, filled_by: {} });

describe('store', () => {
  it('sends no image to a hosted service until the user turns one on', () => {
    expect(initialAi).toMatchObject({ use_qwen_image: false, use_rescue: false, use_triposr: false, use_solaria: false });
  });
  it('freezes the images a job was started with', () => {
    let s: State = { ...initialState, items: [item('a'), item('b', 'top')] };
    s = reducer(s, { type: 'START_JOB', jobId: 'j1' });
    expect(s.jobItems).toEqual([{ url: 'blob:a', face: 'front', kind: 'sketch' }, { url: 'blob:b', face: 'top', kind: 'sketch' }]);
    s = reducer(s, { type: 'REMOVE_ITEM', id: 'a' });
    expect(s.jobItems[0].url).toBe('blob:a');
  });

  it('ignores an analysis for another job', () => {
    let s = reducer({ ...initialState, items: [item('a')] }, { type: 'START_JOB', jobId: 'j2' });
    s = reducer(s, { type: 'ANALYSIS', analysis: analysis('j1') });
    expect(s.analysis).toBeNull();
    s = reducer(s, { type: 'ANALYSIS', analysis: analysis('j2') });
    expect(s.analysis?.request_id).toBe('j2');
  });

  it('forgets typed feature values when the rejected faces change, and keeps the typed sizes', () => {
    // features are numbered per fuse: a reject can renumber them, so features[1] may name another feature next time
    let s: State = { ...initialState, typed: { 'envelope.x_mm': 50, 'features[1].diameter_mm': 6, 'features[2].keep': 0 } };
    s = reducer(s, { type: 'TOGGLE_REJECT', face: 'right' });
    expect(s.rejected).toEqual(['right']);
    expect(s.typed).toEqual({ 'envelope.x_mm': 50 });
  });

  it('ignores a model built for an analysis the user has left', () => {
    const s0: State = { ...initialState, analysis: analysis('j2') };
    const late = reducer(s0, { type: 'MODEL', model: { key: 'old' } as ModelResult, spec: { version: 'mv1' } as Spec, requestId: 'j1' });
    expect(late.model).toBeNull();
    const now = reducer(s0, { type: 'MODEL', model: { key: 'new' } as ModelResult, spec: { version: 'mv1' } as Spec, requestId: 'j2' });
    expect(now.model?.key).toBe('new');
  });

  it('remembers which typed values the analysis on screen already holds, across visits to Review', () => {
    let s: State = reducer({ ...initialState, items: [item('a')] }, { type: 'START_JOB', jobId: 'j1' });
    expect(s.applied.typed).toBe(s.typed);
    s = reducer(s, { type: 'ANALYSIS', analysis: analysis('j1') });
    expect(s.applied.typed).toBe(s.typed);
    s = reducer(s, { type: 'TYPE_VALUE', path: 'envelope.x_mm', value: 50 });
    expect(s.applied.typed).not.toBe(s.typed);  // a merge is due, even if Review was left before it answered
    const sent = { typed: s.typed, rejected: s.rejected };
    s = reducer(s, { type: 'ANALYSIS', analysis: analysis('j1'), applied: sent });
    expect(s.applied).toEqual(sent);
    expect(s.applied.typed).toBe(s.typed);
  });

  it('remembers the spec a model was built from', () => {
    const spec = { version: 'mv1' } as Spec;
    const model = { key: 'k' } as ModelResult;
    const s = reducer(initialState, { type: 'MODEL', model, spec });
    expect(s.modelSpec).toBe(spec);
    expect(reducer(s, { type: 'MODEL', model: null }).modelSpec).toBeNull();
  });
});
