import { describe, expect, it } from 'vitest';
import { photoPlan } from './photo';

const MB = 1024 * 1024;

describe('photoPlan', () => {
  it('leaves a normal JPEG or PNG alone', () => {
    expect(photoPlan('image/jpeg', 3 * MB, 4000, 3000)).toBeNull();
    expect(photoPlan('image/png', 1 * MB, 1600, 1200)).toBeNull();
  });
  it('shrinks a 48 MP photo to 4000 px on its long side before it crosses mobile data', () => {
    expect(photoPlan('image/jpeg', 12 * MB, 8000, 6000)).toEqual({ width: 4000, height: 3000 });
  });
  it('re-encodes a file the server would refuse: HEIC, WebP, or past 10 MB', () => {
    expect(photoPlan('image/heic', 2 * MB, 3000, 4000)).toEqual({ width: 3000, height: 4000 });
    expect(photoPlan('image/webp', 1 * MB, 1000, 800)).toEqual({ width: 1000, height: 800 });
    expect(photoPlan('image/png', 11 * MB, 3000, 3000)).toEqual({ width: 3000, height: 3000 });
  });
});
