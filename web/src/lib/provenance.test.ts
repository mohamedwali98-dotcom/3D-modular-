import { describe, expect, it } from 'vitest';
import fixture from './__fixtures__/analyze-success.json';
import type { Spec } from '../api/types';
import { BADGE, CHECK, countChecks, envelopeRows, featureRows } from './provenance';

const spec = fixture.spec as unknown as Spec;

describe('provenance helpers', () => {
  it('lists the envelope with its provenance', () => {
    const rows = envelopeRows(spec);
    expect(rows.map((r) => [r.label, r.value, r.prov])).toEqual([
      ['Width', 50, 'user_written'],
      ['Height', 30, 'user_written'],
      ['Depth', 20, 'user_edited'],
    ]);
    expect(rows[0]).toMatchObject({ key: 'x', axis: 'X', path: 'envelope.x_mm' });
  });

  it('lists every numeric feature field, with snapped marked', () => {
    const rows = featureRows(spec);
    const h2d = rows.find((r) => r.name === 'Hole 2 · Ø');
    expect(h2d).toMatchObject({ path: 'features[1].diameter_mm', face: 'right', value: 5.5, prov: 'default', snapped: true });
    const h1a = rows.find((r) => r.name === 'Hole 1 · a');
    expect(h1a).toMatchObject({ path: 'features[0].a_mm', value: 20, prov: 'estimated', snapped: false });
    expect(rows).toHaveLength(6);
  });

  it('lists pockets and pins read from a drawing', () => {
    const drawn = {
      ...spec,
      features: [
        { type: 'pocket', face: 'front', a_mm: 7.5, b_mm: 50, width_mm: 15, height_mm: 20, depth_mm: 20 },
        { type: 'boss', face: 'top', a_mm: 20, b_mm: 55, diameter_mm: 8, height_mm: 6 },
      ],
      provenance: { ...spec.provenance, 'features[0].width_mm': 'scaled', 'features[1].diameter_mm': 'scaled' },
    } as unknown as Spec;
    const rows = featureRows(drawn);
    expect(rows.find((r) => r.name === 'Pocket 1 · height')).toMatchObject({ path: 'features[0].height_mm', value: 20 });
    expect(rows.find((r) => r.name === 'Pin 1 · Ø')).toMatchObject({ path: 'features[1].diameter_mm', value: 8, groupName: 'Pin 1 · Ø' });
    expect(rows).toHaveLength(9);
  });

  it('counts check groups the way the design does', () => {
    expect(countChecks(spec)).toBe(3);
  });

  it('keeps the design badge map', () => {
    expect(CHECK).toEqual(['inferred', 'estimated', 'default']);
    expect(BADGE.default.label).toBe('Default — check');
    expect(BADGE.required.icon).toBe('•');
  });

  it('says a scaled value comes from the sizes you typed, not a reference object', () => {
    // fuse.py marks a value 'measured' when a coin or card set the scale, 'scaled' when your overall size did.
    expect(BADGE.scaled.label).toBe('Scaled from your sizes');
  });
});
