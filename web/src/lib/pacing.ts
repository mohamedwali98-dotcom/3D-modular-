import type { Face, Job, StageKey, StageState } from '../api/types';

/**
 * Display pacing for the Analyzing screen.
 *
 * The pipeline can finish a stage in a few milliseconds (OpenCV) or in many seconds (a VLM call).
 * `pace` turns the job's real events into a display clock so every stage stays legible:
 *
 * - Stages play in order. Stage i starts on screen when stage i-1 ends on screen.
 * - A stage's data (outlines, reads, coverage) is only revealed once the job says the stage is
 *   finished. The reveal starts at max(display start, real end) and lasts at least MIN[key].
 * - A stage is shown finished only when the job says so AND its minimum reveal time has elapsed,
 *   so playback never runs ahead of the real pipeline.
 *
 * Real end times come from the server's `started`/`ended` stamps, taken relative to the first
 * `started` stamp, so client/server clock skew cancels out. `startedAt` and `now` are client ms.
 */

export const MIN = {
  label: 0.9, outline: 1.5, readEach: 0.7, readMin: 0.9, draw: 1.2, fuse: 0.9, skipped: 0.4,
  // "One sheet (all views)": read_sketch's own three checkpoints, in place of label/outline/read.
  views: 0.9, lines: 1.2, values: 0.9,
} as const;

export const FACES: Face[] = ['front', 'top', 'right', 'back', 'left', 'bottom'];
/** The three faces the part is fused from (Spec.views). */
const FUSED: Face[] = ['front', 'top', 'right'];

export type FaceShow = 'empty' | 'photo' | 'pending' | 'drawing' | 'ai' | 'mirror' | 'assumed' | 'inferred';

export interface PacedStage {
  key: StageKey;
  /** Display state: never ahead of the job's own state. */
  state: StageState;
  tool: string;
  ai: boolean;
  detail: string;
  /** Display-clock seconds. `reveal`/`end` are Infinity until the job reports the stage finished. */
  start: number;
  reveal: number;
  end: number;
  /** Reveal progress 0..1 (0 while the real stage is still running). */
  p: number;
}

export interface Playback {
  /** Display clock in seconds. */
  t: number;
  stages: PacedStage[];
  /** Index of the stage to headline (the running one, else the next pending, else the last). */
  cur: number;
  /** Overall progress 0..1 for the ring. */
  progress: number;
  /** Outline reveal 0..1 per image (index matches job.images). */
  trace: number[];
  /** Reads revealed per image, fractional: read k is being read while k < n < k+1 and locked once n >= k+1. */
  readsShown: number[];
  faces: Record<Face, FaceShow>;
  allDone: boolean;
}

const cl = (x: number) => Math.max(0, Math.min(1, x));
const FINISHED: StageState[] = ['done', 'skipped', 'failed'];

function minFor(key: StageKey, state: StageState, reads: number): number {
  if (state === 'skipped') return MIN.skipped;
  if (key === 'read') return Math.max(MIN.readMin, reads * MIN.readEach);
  return MIN[key];
}

export function pace(job: Job, startedAt: number, now: number): Playback {
  const t = Math.max(0, (now - startedAt) / 1000);
  const readCounts = job.images.map((im) => im.reads.length);
  const totalReads = readCounts.reduce((a, b) => a + b, 0);

  const stamps = job.stages.map((s) => s.started).filter((v): v is number => typeof v === 'number');
  const t0 = stamps.length ? Math.min(...stamps) : 0;

  const stages: PacedStage[] = [];
  let prevEnd = 0;
  for (const s of job.stages) {
    const start = prevEnd;
    const finished = FINISHED.includes(s.state);
    const endOff = typeof s.ended === 'number' ? Math.max(0, s.ended - t0) : 0;
    const dur = minFor(s.key, s.state, totalReads);
    const reveal = finished && Number.isFinite(start) ? Math.max(start, endOff) : Infinity;
    const end = reveal + dur;
    let state: StageState;
    if (!Number.isFinite(start) || t < start || s.state === 'pending') state = 'pending';
    else if (t < end) state = 'running';
    else state = s.state;
    const p = Number.isFinite(reveal) ? cl((t - reveal) / dur) : 0;
    stages.push({ key: s.key, state, tool: s.tool, ai: s.ai, detail: s.detail, start, reveal, end, p });
    // The next stage only starts on screen once this one has ended on screen.
    prevEnd = state === 'pending' ? Infinity : end;
  }

  const byKey = (k: StageKey) => stages.find((s) => s.key === k);
  const isOver = (s?: PacedStage) => !!s && s.state !== 'pending' && s.state !== 'running';
  const label = byKey('label'), outline = byKey('outline'), read = byKey('read'), draw = byKey('draw'), fuse = byKey('fuse');

  // Outline reveal: images traced one after the other inside the outline window (a sheet's: its views stage).
  const traced = outline ?? byKey('views');
  const n = job.images.length;
  const op = traced ? traced.p : 0;
  const trace = job.images.map((_, i) => (n ? cl(op * n - i) : 0));

  // Reads locked one after the other, across images in order.
  const g = (read ? read.p : 0) * totalReads;
  let off = 0;
  const readsShown = readCounts.map((c) => {
    const v = Math.max(0, Math.min(c, g - off));
    off += c;
    return v;
  });

  const faces = {} as Record<Face, FaceShow>;
  for (const f of FACES) {
    const cov = job.coverage[f] ?? 'empty';
    const img = job.images.findIndex((im) => im.face === f);
    let show: FaceShow = 'empty';
    if (cov === 'observed' || img >= 0) {
      show = (img >= 0 ? trace[img] >= 1 : isOver(traced)) ? 'photo' : 'empty';
    } else if (isOver(draw)) {
      if (cov === 'qwen-image' || cov === 'triposr') show = 'ai';
      else if (cov === 'assumed') show = 'assumed';
      else if (cov === 'mirrored') show = fuse && fuse.state !== 'pending' ? 'mirror' : 'empty';
      else if (cov === 'inferred') show = 'inferred';  // a turned part's view, following from the others
    } else if (draw?.ai && FUSED.includes(f) && isOver(label)) {
      show = draw.state === 'running' ? 'drawing' : 'pending';
    }
    faces[f] = show;
  }

  const allDone = job.status === 'done' && stages.length > 0 && stages.every(isOver);

  let cur = stages.findIndex((s) => s.state === 'running');
  if (cur < 0) cur = stages.findIndex((s) => s.state === 'pending');
  if (cur < 0 || allDone) cur = Math.max(0, stages.length - 1);

  const frac = stages.map((s) => {
    if (isOver(s)) return 1;
    if (s.state !== 'running') return 0;
    if (Number.isFinite(s.reveal)) return 0.3 + 0.7 * s.p;
    return 0.3 * cl((t - s.start) / 6);
  });
  const progress = stages.length ? frac.reduce((a, b) => a + b, 0) / stages.length : 0;

  return { t, stages, cur, progress, trace, readsShown, faces, allDone };
}
