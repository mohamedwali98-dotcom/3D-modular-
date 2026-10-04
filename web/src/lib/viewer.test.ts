import { describe, expect, it } from 'vitest';
import { viewerModel } from './viewer';

describe('viewerModel', () => {
  const model = { key: 'k' }, spec = { v: 1 }, edited = { v: 2 };
  it('shows the model built from the spec on screen', () => {
    expect(viewerModel(model, spec, spec, false)).toBe(model);
  });
  it('keeps the old part on screen while its rebuild runs', () => {
    expect(viewerModel(model, spec, edited, true)).toBe(model);
  });
  it('never shows a part, its sizes or its volume for a spec it was not built from once the rebuild stopped', () => {
    expect(viewerModel(model, spec, edited, false)).toBeNull();
  });
});

describe('viewerModel and the shape settings', () => {
  const model = { key: 'k' }, spec = { v: 1 };
  it('drops a part built with other shape settings once their rebuild stopped', () => {
    const built = { finish: 'none', finish_mm: 1 }, now = { finish: 'fillet', finish_mm: 6 };
    expect(viewerModel(model, spec, spec, false, built, now)).toBeNull();
    expect(viewerModel(model, spec, spec, true, built, now)).toBe(model);
    expect(viewerModel(model, spec, spec, false, built, { ...built })).toBe(model);
  });
});
