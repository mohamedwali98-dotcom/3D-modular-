import { describe, expect, it } from 'vitest';
import { layerMax, withNozzle } from './print';

describe('print', () => {
  it('caps the layer at three quarters of the nozzle, as the slicer settings require', () => {
    expect(layerMax('0.2')).toBe(0.15);
    expect(layerMax('0.4')).toBe(0.3);
    expect(layerMax('0.8')).toBe(0.32); // the slider's own top
  });
  it('lowers the layer when a smaller nozzle is chosen, and keeps it otherwise', () => {
    expect(withNozzle({ nozzle: '0.4', layer: 0.2 }, '0.2')).toEqual({ nozzle: '0.2', layer: 0.15 });
    expect(withNozzle({ nozzle: '0.4', layer: 0.2 }, '0.6')).toEqual({ nozzle: '0.6', layer: 0.2 });
  });
});
