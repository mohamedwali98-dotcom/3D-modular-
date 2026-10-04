import { describe, expect, it } from 'vitest';
import { zoomed } from './orbit';

describe('zoomed', () => {
  it('spreading two fingers brings the camera closer, pinching moves it away', () => {
    expect(zoomed(200, 50, 100 / 200)).toBe(100);
    expect(zoomed(100, 50, 200 / 100)).toBe(200);
  });
  it('stays between 1.4 and 6.4 part sizes, like the wheel', () => {
    expect(zoomed(100, 50, 0.01)).toBe(70);
    expect(zoomed(100, 50, 100)).toBe(320);
  });
});
