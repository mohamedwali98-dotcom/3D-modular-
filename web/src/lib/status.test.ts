import { describe, expect, it } from 'vitest';
import type { Status } from '../api/types';
import { aiReady, aiShownOn, formatReady } from './status';

const status = (p: Partial<Status['providers']>): Status => ({
  providers: { vision: false, reader: false, qwen_image: false, triposr: false, solaria: false, slicer: false, blender: false, ...p },
  ttl_s: 3600,
});

describe('what this server can do', () => {
  it('offers G-code only with a slicer and the Blender scene only with Blender', () => {
    expect(formatReady('gcode', status({}))).toBe(false);
    expect(formatReady('gcode', status({ slicer: true }))).toBe(true);
    expect(formatReady('blend', status({}))).toBe(false);
    expect(formatReady('step', status({}))).toBe(true);
  });
  it('offers a hosted helper only when it is set up', () => {
    expect(aiReady('use_qwen_image', status({}))).toBe(false);
    expect(aiReady('use_rescue', status({ qwen_image: true }))).toBe(true);
    expect(aiReady('use_reader', status({ vision: true }))).toBe(true);
  });
  it('blocks nothing before the status has loaded', () => {
    expect(formatReady('gcode', null)).toBe(true);
    expect(aiReady('use_solaria', null)).toBe(true);
  });
});

describe('aiShownOn', () => {
  it('never shows a helper ticked when this server cannot run it', () => {
    const ai = { use_reader: true, use_qwen_image: true } as Parameters<typeof aiShownOn>[1];
    expect(aiShownOn('use_reader', ai, status({}))).toBe(false);
    expect(aiShownOn('use_reader', ai, status({ reader: true }))).toBe(true);
    expect(aiShownOn('use_qwen_image', ai, null)).toBe(true);
  });
});
