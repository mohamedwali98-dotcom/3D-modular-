import type { Face, Provenance, Spec } from '../api/types';

/** `found`: a value the analysis kept while it stopped, when the server could not say where it came from. */
export type BadgeKey = Provenance | 'required' | 'found';
export interface BadgeStyle { label: string; icon: string; fg: string; line: 'solid' | 'dotted' | 'dashed'; bg: string }

const TRU = 'var(--trusted)', CHK = 'var(--check)', AI = 'var(--ai)', STOP = 'var(--stop)';
const tint = (c: string) => `color-mix(in oklch, ${c} 12%, transparent)`;

// Copied from Review v2.dc.html
export const BADGE: Record<BadgeKey, BadgeStyle> = {
  user_written: { label: 'Written by you', icon: '✓', fg: TRU, line: 'solid', bg: tint(TRU) },
  measured: { label: 'Measured', icon: '✓', fg: TRU, line: 'solid', bg: tint(TRU) },
  user_edited: { label: 'Edited by you', icon: '✓', fg: TRU, line: 'solid', bg: tint(TRU) },
  scaled: { label: 'Scaled from your sizes', icon: '✓', fg: TRU, line: 'dotted', bg: 'transparent' },
  inferred: { label: 'AI-drawn — check', icon: '◇', fg: AI, line: 'dashed', bg: 'transparent' },
  estimated: { label: 'Estimated — check', icon: '!', fg: CHK, line: 'dashed', bg: 'transparent' },
  default: { label: 'Default — check', icon: '!', fg: CHK, line: 'dashed', bg: 'transparent' },
  required: { label: 'Required', icon: '•', fg: STOP, line: 'dashed', bg: 'transparent' },
  found: { label: 'Found — check', icon: '!', fg: CHK, line: 'dashed', bg: 'transparent' },
};

export const CHECK: Provenance[] = ['inferred', 'estimated', 'default'];

export const isCheck = (p: BadgeKey | undefined | null): boolean => !!p && (CHECK as string[]).includes(p);

export interface EnvelopeRow { key: 'x' | 'y' | 'z'; label: 'Width' | 'Height' | 'Depth'; axis: 'X' | 'Y' | 'Z'; path: string; value: number; prov: Provenance }

const ENV = [
  { key: 'x', label: 'Width', axis: 'X' },
  { key: 'y', label: 'Height', axis: 'Y' },
  { key: 'z', label: 'Depth', axis: 'Z' },
] as const;

/** Provenance for a path; values with no entry are treated as `default` (they need a check). */
export const provOf = (spec: Spec, path: string): Provenance => spec.provenance[path] ?? 'default';

export function envelopeRows(spec: Spec): EnvelopeRow[] {
  return ENV.map((e) => {
    const path = `envelope.${e.key}_mm`;
    return { key: e.key, label: e.label, axis: e.axis, path, value: spec.envelope[`${e.key}_mm`], prov: provOf(spec, path) };
  });
}

export interface FeatureRow {
  path: string;         // e.g. "features[1].diameter_mm"
  name: string;         // e.g. "Hole 2 · Ø"
  face: Face;
  value: number;
  prov: Provenance;
  snapped: boolean;
  index: number;        // index in spec.features
  field: string;        // e.g. "diameter_mm"
  label: string;        // short field label: "Ø", "a", "b", "width", "length", "angle", "depth"
  feature: string;      // "Hole 2"
  group: string;        // check group id, e.g. "features[0].size" or "features[0].position"
  groupName: string;    // "Hole 1 · Ø" / "Hole 1 · position"
}

type FeatureType = 'hole' | 'slot' | 'pocket' | 'boss';
const NAMES: Record<FeatureType, string> = { hole: 'Hole', slot: 'Slot', pocket: 'Pocket', boss: 'Pin' };

const FIELDS: Record<FeatureType, { field: string; label: string; group: 'size' | 'position' | 'angle' | 'depth' }[]> = {
  hole: [
    { field: 'diameter_mm', label: 'Ø', group: 'size' },
    { field: 'a_mm', label: 'a', group: 'position' },
    { field: 'b_mm', label: 'b', group: 'position' },
    { field: 'depth_mm', label: 'depth', group: 'depth' },
  ],
  slot: [
    { field: 'width_mm', label: 'width', group: 'size' },
    { field: 'length_mm', label: 'length', group: 'size' },
    { field: 'a_mm', label: 'a', group: 'position' },
    { field: 'b_mm', label: 'b', group: 'position' },
    { field: 'angle_deg', label: 'angle', group: 'angle' },
    { field: 'depth_mm', label: 'depth', group: 'depth' },
  ],
  pocket: [
    { field: 'width_mm', label: 'width', group: 'size' },
    { field: 'height_mm', label: 'height', group: 'size' },
    { field: 'a_mm', label: 'a', group: 'position' },
    { field: 'b_mm', label: 'b', group: 'position' },
    { field: 'depth_mm', label: 'depth', group: 'depth' },
  ],
  boss: [
    { field: 'diameter_mm', label: 'Ø', group: 'size' },
    { field: 'height_mm', label: 'height', group: 'size' },
    { field: 'a_mm', label: 'a', group: 'position' },
    { field: 'b_mm', label: 'b', group: 'position' },
  ],
};

/**
 * The original index of each feature on screen. The server fuses the features in the same order every time and
 * then drops the ones the user removed (`features[k].keep` = 0), so the spec shows them renumbered; values the
 * user types must keep the original index or they land on the wrong feature after a removal.
 */
function originalIndices(n: number, typed?: Record<string, number>): number[] {
  const removed = new Set<number>();
  for (const [key, value] of Object.entries(typed ?? {})) {
    const m = /^features\[(\d+)\]\.keep$/.exec(key);
    if (m && value === 0) removed.add(Number(m[1]));
  }
  const out: number[] = [];
  for (let k = 0; out.length < n; k += 1) if (!removed.has(k)) out.push(k);
  return out;
}

export function featureRows(spec: Spec, typed?: Record<string, number>): FeatureRow[] {
  const rows: FeatureRow[] = [];
  const count: Record<FeatureType, number> = { hole: 0, slot: 0, pocket: 0, boss: 0 };
  const original = originalIndices(spec.features.length, typed);
  spec.features.forEach((f, shown) => {
    const index = original[shown];
    count[f.type] += 1;
    const feature = `${NAMES[f.type]} ${count[f.type]}`;
    for (const d of FIELDS[f.type]) {
      const value = (f as unknown as Record<string, unknown>)[d.field];
      if (typeof value !== 'number') continue;
      const path = `features[${index}].${d.field}`;
      const shownPath = `features[${shown}].${d.field}`;  // the spec's own provenance keys follow the screen
      const groupLabel = d.group === 'size' ? (f.type === 'hole' || f.type === 'boss' ? 'Ø' : 'size') : d.group;
      rows.push({
        path, name: `${feature} · ${d.label}`, face: f.face, value, prov: provOf(spec, shownPath),
        snapped: spec.snapped.includes(shownPath), index, field: d.field, label: d.label, feature,
        group: `features[${index}].${d.group}`, groupName: `${feature} · ${groupLabel}`,
      });
    }
  });
  return rows;
}

/**
 * Number of feature-level groups (size, position, ...) whose provenance still needs a check.
 * Paths in `confirmed` (values the user typed or confirmed) count as `user_edited`.
 */
export function countChecks(spec: Spec, confirmed?: Record<string, number>): number {
  const groups = new Set<string>();
  for (const r of featureRows(spec, confirmed)) if (isCheck(r.prov) && !(confirmed && r.path in confirmed)) groups.add(r.group);
  return groups.size;
}
