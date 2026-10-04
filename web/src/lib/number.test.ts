import { describe, expect, it } from 'vitest';
import { dropFeatureDrafts, fieldProblem, parseMm, shownProblems } from './number';

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

describe('shownProblems', () => {
  it('holds Build only for a value still on screen: a removed feature takes its bad draft with it', () => {
    const drafts = { 'features[0].diameter_mm': '0', 'envelope.x_mm': '50' };
    expect(Object.keys(shownProblems(drafts, new Set(['envelope.x_mm', 'features[0].diameter_mm'])))).toEqual(['features[0].diameter_mm']);
    expect(shownProblems(drafts, new Set(['envelope.x_mm']))).toEqual({});
  });
});

describe('parseMm and thousands', () => {
  it('refuses "1,200": 1.2 mm in French, 1200 mm in English, so it must be typed unambiguously', () => {
    expect(parseMm('1,200')).toBeNull();
    expect(parseMm('12,500')).toBeNull();
    expect(parseMm('1,2')).toBe(1.2);
    expect(parseMm('1,25')).toBe(1.25);
    expect(parseMm('1200')).toBe(1200);
  });
});
