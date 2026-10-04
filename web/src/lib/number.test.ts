import { describe, expect, it } from 'vitest';
import { dropFeatureDrafts, fieldProblem, parseMm } from './number';

describe('number', () => {
  it('reads a decimal comma, Arabic-Indic and Persian digits, and refuses everything else', () => {
    expect(parseMm('6,5')).toBe(6.5);
    expect(parseMm(' 12.5 ')).toBe(12.5);
    expect(parseMm('٦٫٥')).toBe(6.5);
    expect(parseMm('۱۲')).toBe(12);
    expect(parseMm('6.')).toBe(6);
    expect(parseMm('-3')).toBe(-3);
    for (const bad of ['', '12.5x', '0x10', '1e3', 'abc', '1.2.3', '.']) expect(parseMm(bad)).toBeNull();
  });
  it('says what is wrong with a typed value, in the field it belongs to', () => {
    expect(fieldProblem('features[0].diameter_mm', '6,5')).toBeNull();
    expect(fieldProblem('features[0].diameter_mm', '0')).toMatch(/above 0/);
    expect(fieldProblem('envelope.x_mm', '20000')).toMatch(/10 000/);
    expect(fieldProblem('features[0].a_mm', '-3')).toBeNull(); // a position may be anywhere
    expect(fieldProblem('features[0].width_mm', '12x')).toMatch(/number like 12\.5/);
  });
  it('drops only the feature drafts when the faces change', () => {
    expect(dropFeatureDrafts({ 'envelope.x_mm': '50', 'features[1].diameter_mm': '6' })).toEqual({ 'envelope.x_mm': '50' });
  });
});
