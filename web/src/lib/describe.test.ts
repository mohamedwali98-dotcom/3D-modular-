import { describe, expect, it } from 'vitest';
import type { Spec } from '../api/types';
import { initialState, reducer } from '../state/store';
import { buildNav } from './nav';
import { describedAnalysis, formatValue, frontPreview, partRows } from './describe';

const rect = (w: number, h: number): [number, number][] => [[0, 0], [w, 0], [w, h], [0, h]];
const spec: Spec = {
  version: 'mv1', envelope: { x_mm: 60, y_mm: 40, z_mm: 5 },
  views: {
    front: { outer: rect(60, 40), inner: [], source: 'observed', confidence: 1 },
    top: { outer: rect(60, 5), inner: [], source: 'observed', confidence: 1 },
    right: { outer: rect(5, 40), inner: [], source: 'observed', confidence: 1 },
  },
  features: [{ type: 'hole', face: 'front', a_mm: 10, b_mm: 30, diameter_mm: 6, depth_mm: null }],
  finishes: [], provenance: {}, snapped: [], warnings: [], confidence: 1,
};

describe('describe helpers', () => {
  it('lists written values in the type order, then the missing ones', () => {
    const rows = partRows({ type: 'plate', values: { thickness_mm: 5, width_mm: 60 }, holes: [] }, ['height_mm']);
    expect(rows.map((r) => [r.key, r.value])).toEqual([['width_mm', 60], ['thickness_mm', 5], ['height_mm', null]]);
    expect(rows[0].label).toBe('Width');
  });

  it('never shows a missing value as written', () => {
    const rows = partRows({ type: 'plate', values: { width_mm: 60 }, holes: [] }, ['width_mm']);
    expect(rows).toEqual([{ key: 'width_mm', label: 'Width', value: null }]);
    expect(partRows(null, [])).toEqual([]);
  });

  it('formats millimetres and counts', () => {
    expect(formatValue('width_mm', 60)).toBe('60 mm');
    expect(formatValue('bolt_count', 4)).toBe('4');
  });

  it('builds a job-less analysis the store accepts', () => {
    const a = describedAnalysis(spec);
    expect(a).toEqual({ request_id: '', spec, abstain: null, filled_by: {} });
    const s = reducer({ ...initialState, jobId: 'j1', typed: { x: 1 } }, { type: 'ANALYSIS', analysis: a });
    expect(s.analysis?.spec).toBe(spec);
    expect(s.jobId).toBeNull();
    expect(s.typed).toEqual({});
  });

  it('flips the front outline so up is up', () => {
    const p = frontPreview(spec)!;
    expect(p.path.startsWith('M0 40 L60 40')).toBe(true);
    expect(p.holes).toEqual([{ cx: 10, cy: 10, r: 3 }]);
  });

  it('keeps the chat and clears it on reset', () => {
    let s = reducer(initialState, { type: 'CHAT', messages: [{ role: 'user', content: 'a plate' }] });
    expect(s.chat.messages).toHaveLength(1);
    s = reducer(s, { type: 'RESET' });
    expect(s.chat).toEqual({ messages: [], last: null });
  });

  it('shows Describe in the first nav slot', () => {
    const c = buildNav({ ...initialState, screen: 'describe' })[0];
    expect(c).toMatchObject({ label: 'Describe', sub: 'Chat with the AI', st: 'current', screen: 'describe' });
    const m = buildNav({ ...initialState, screen: 'model', analysis: describedAnalysis(spec) })[0];
    expect(m).toMatchObject({ label: 'Describe', st: 'done', screen: 'describe' });
    expect(buildNav({ ...initialState, screen: 'capture', analysis: describedAnalysis(spec) })[0].label).toBe('Capture');
  });
});
