// What this server can do (GET /api/status): options it cannot honour are shown off, not offered and then ignored.
import type { AiSettings, Status } from '../api/types';

type Provider = keyof Status['providers'];

const FORMAT_NEEDS: Record<string, Provider> = { gcode: 'slicer', blend: 'blender' };
const AI_NEEDS: Partial<Record<keyof AiSettings, Provider[]>> = {
  use_reader: ['reader', 'vision'], use_qwen_image: ['qwen_image'], use_rescue: ['qwen_image'],
  use_triposr: ['triposr'], use_solaria: ['solaria'],
};

/** An export format this server can write; true until the status has loaded. */
export function formatReady(format: string, status: Status | null): boolean {
  const need = FORMAT_NEEDS[format];
  return !need || !status || status.providers[need];
}

/** An AI helper this server has set up; true until the status has loaded. */
export function aiReady(setting: keyof AiSettings, status: Status | null): boolean {
  const needs = AI_NEEDS[setting];
  return !needs || !status || needs.some((p) => status.providers[p]);
}

/** A helper's switch as shown: never ticked when this server cannot run it (a locked, ticked box reads as "on"). */
export function aiShownOn(setting: keyof AiSettings, ai: AiSettings, status: Status | null): boolean {
  return !!ai[setting] && aiReady(setting, status);
}
