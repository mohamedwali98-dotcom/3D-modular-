import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { ApiError, cancelJob, getJob } from '../api/client';
import type { Face, Job, JobImage, ReadValue, Spec, StageKey } from '../api/types';
import { StopCard } from '../components/StopCard';
import { analyzingCta } from '../lib/abstain';
import { pace, type FaceShow, type PacedStage, type Playback } from '../lib/pacing';
import { BADGE, countChecks, envelopeRows, featureRows } from '../lib/provenance';
import { useStore } from '../state/store';

// Port of "Analyzing v2.dc.html". The scripted timeline is replaced by `pace()` over the real job:
// every outline, read value and filled face on screen comes from the polled Job, never invented.

const SACC = 'oklch(0.74 0.10 52)', STRU = 'oklch(0.78 0.14 155)', SAI = 'oklch(0.70 0.07 195)';
const MONO = "'Geist Mono', monospace";
const cl = (x: number) => Math.max(0, Math.min(1, x));
const ez = (x: number) => 1 - Math.pow(1 - cl(x), 3);
const back = (x: number) => { x = cl(x); const s = 1.6; return 1 + (s + 1) * Math.pow(x - 1, 3) + s * Math.pow(x - 1, 2); };
const fmt = (v: number) => String(Math.round(v * 10) / 10);
const POLL_MS = 400;
const MAX_FAILS = 15;

/** Fixed UI copy per stage (what the step does). Values shown come from the job. */
const COPY: Record<StageKey, { name: string; run: string }> = {
  label: { name: 'Reading your photos', run: 'Working out which face each photo shows' },
  outline: { name: 'Tracing outlines', run: 'Following your pencil lines' },
  read: { name: 'Reading your numbers', run: 'Reading what you wrote' },
  draw: { name: 'Drawing the missing face', run: 'Filling in the faces you did not send' },
  fuse: { name: 'Putting it together', run: 'Snapping the views into one part' },
  // "One sheet (all views)": find the faces, label and complete them, read the numbers.
  views: { name: 'Finding the faces', run: 'Finding the views drawn on your sheet' },
  lines: { name: 'Labelling and completing the faces', run: 'Naming each face and reading its inner lines' },
  values: { name: 'Reading your numbers', run: 'Reading what you wrote' },
};

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six'];

function useReducedMotion(): boolean {
  const [rm, setRm] = useState(() => typeof window !== 'undefined' && !!window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setRm(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return rm;
}

function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    let raf = 0;
    const loop = () => { setNow(Date.now()); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    // rAF pauses in background tabs; keep the display clock moving there too.
    const iv = window.setInterval(() => setNow(Date.now()), 250);
    return () => { cancelAnimationFrame(raf); window.clearInterval(iv); };
  }, []);
  return now;
}

// ---------------------------------------------------------------- sketch pane

interface PaneProps {
  W: number; H: number; big: boolean;
  url: string | undefined;
  image: JobImage | undefined;
  natural: [number, number] | undefined;
  onNatural: (wh: [number, number]) => void;
  trace: number; reads: number; readBase: number; totalReads: number;
  read: PacedStage | undefined;
  /** The read stage's tool, as the server named it (the configured model, or TrOCR). */
  readerName: string;
  tagO: number; faceTxt: string; tagTxt: string; tagAi: boolean;
  t: number; rm: boolean;
  corner?: string;
  note?: { o: number; text: string };
}

function SketchPane(p: PaneProps) {
  const { W, H, big, image, t, rm } = p;
  const iw = image && image.width > 0 ? image.width : p.natural?.[0] ?? W;
  const ih = image && image.height > 0 ? image.height : p.natural?.[1] ?? H;
  const k = Math.min(W / iw, H / ih), ox = (W - iw * k) / 2, oy = (H - ih * k) / 2;
  const u = (px: number) => px / k; // screen px -> image px

  const tracing = p.trace > 0 && p.trace < 1;
  const scan = {
    scanL: `${p.trace * 100}%`, scanO: rm ? 0 : tracing ? 1 : 0,
    clip: rm ? 'none' : `inset(0 ${100 - p.trace * 100}% 0 0)`,
    bpOp: rm ? ez(p.trace) : p.trace > 0 ? 1 : 0, glow: tracing ? 3 : 1.2,
  };

  const circles = image?.circles ?? [];
  const oP = circles.length ? cl(p.trace / 0.7) : p.trace;
  const cP = circles.length ? cl((p.trace - 0.7) / 0.3) : 0;
  const lw = big ? 2.2 : 1.45;

  let poly: { pts: string; L: number } | null = null;
  if (image?.outline && image.outline.length >= 2) {
    const pts = [...image.outline];
    const [f, l] = [pts[0], pts[pts.length - 1]];
    if (f[0] !== l[0] || f[1] !== l[1]) pts.push(f);
    let L = 0;
    for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    poly = { pts: pts.map((q) => `${q[0]},${q[1]}`).join(' '), L };
  }

  // Read boxes and chips. Read k of this image locks when the global read clock passes it.
  const readDur = p.read && Number.isFinite(p.read.reveal) ? p.read.end - p.read.reveal : 1;
  const reads = (image?.reads ?? []).map((r: ReadValue, i) => {
    const local = p.reads - i; // 0..1 while being read
    const locked = local >= 1;
    const lockAt = (p.read?.reveal ?? 0) + ((p.readBase + i + 1) * readDur) / Math.max(1, p.totalReads);
    const sc = locked && !rm ? 1 + 0.14 * (1 - ez((t - lockAt) / 0.3)) : 1;
    const [bx, by, bw, bh] = r.bbox;
    const pad = u(big ? 6 : 3);
    const cx = ox + (bx + bw / 2) * k;
    const below = oy + (by + bh) * k + (big ? 8 : 4);
    const chipH = big ? 34 : 26;
    const top = below + chipH > H ? oy + by * k - chipH - (big ? 8 : 4) : below;
    const edge = big ? 110 : 70;
    const left = Math.max(edge, Math.min(W - edge, cx));
    const prefix = r.kind === 'diameter' ? 'Ø' : r.kind === 'radius' ? 'R' : '';
    return {
      key: i,
      rect: { x: bx - pad, y: by - pad, w: bw + 2 * pad, h: bh + 2 * pad, c: locked ? STRU : SAI,
        fill: locked ? 'oklch(0.78 0.14 155 / 0.12)' : 'oklch(0.70 0.07 195 / 0.16)',
        da: locked ? 'none' : `${u(7)} ${u(5)}`, dof: -t * u(40), o: local > 0 ? ez(local / 0.35) : 0 },
      chip: { left, top, sc, o: local > 0 ? ez((local - 0.1) / 0.35) : 0, locked, val: locked ? prefix + fmt(r.value_mm) : '··' },
    };
  });

  const holeChipO = circles.length && cP >= 1 ? 1 : 0;
  const c0 = circles[0];

  const box: CSSProperties = { position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block', objectFit: 'contain' };
  return (
    <div style={{ position: 'relative', width: W, height: H, borderRadius: big ? 14 : 12, overflow: 'hidden', background: '#f4f3f0', boxShadow: 'var(--shadow)' }}>
      {p.url && <img src={p.url} alt={`Your ${p.faceTxt.toLowerCase()} image`} style={box}
        onLoad={(e) => p.onNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])} />}
      <div style={{ position: 'absolute', inset: 0, clipPath: scan.clip, opacity: scan.bpOp, background: 'var(--trace-bg)' }}>
        {p.url && <img src={p.url} alt="" style={{ ...box, filter: 'var(--trace-filter)', mixBlendMode: 'var(--trace-blend)' as CSSProperties['mixBlendMode'], opacity: 'var(--trace-img)' }} />}
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(var(--trace-grid) 1px, transparent 1px), linear-gradient(90deg, var(--trace-grid) 1px, transparent 1px)', backgroundSize: big ? '28px 28px' : '14px 14px' }} />
      </div>
      <div style={{ position: 'absolute', top: 0, bottom: 0, left: scan.scanL, width: big ? 140 : 60, marginLeft: big ? -140 : -60, opacity: scan.scanO, background: 'linear-gradient(90deg, transparent, oklch(0.74 0.10 52 / 0.16))', borderRight: '2px solid oklch(0.80 0.09 55)', boxShadow: big ? '6px 0 18px oklch(0.74 0.10 52 / 0.4)' : undefined }} />
      <svg viewBox={`0 0 ${iw} ${ih}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', filter: `drop-shadow(0 0 ${scan.glow}px oklch(0.74 0.10 52 / 0.6))` }}>
        {poly && (
          <polyline points={poly.pts} style={{ fill: 'none', stroke: SACC, strokeWidth: u(lw), strokeLinecap: 'round', strokeLinejoin: 'round',
            strokeDasharray: rm ? 'none' : poly.L, strokeDashoffset: rm ? 0 : poly.L * (1 - oP), opacity: oP > 0 ? 1 : 0 }} />
        )}
        {circles.map((c, i) => {
          const C = Math.PI * c.d;
          return <circle key={i} cx={c.cx} cy={c.cy} r={c.d / 2} style={{ fill: 'none', stroke: SACC, strokeWidth: u(lw), strokeDasharray: rm ? 'none' : C, strokeDashoffset: rm ? 0 : C * (1 - cP), opacity: cP > 0 ? 1 : 0 }} />;
        })}
        {reads.map((r) => (
          <rect key={r.key} x={r.rect.x} y={r.rect.y} width={r.rect.w} height={r.rect.h} rx={u(big ? 5 : 2.5)}
            style={{ fill: r.rect.fill, stroke: r.rect.c, strokeWidth: u(big ? 2.1 : 1.45), strokeDasharray: r.rect.da, strokeDashoffset: r.rect.dof, opacity: r.rect.o }} />
        ))}
      </svg>
      {reads.map((r) => (
        <div key={r.key} style={{ position: 'absolute', left: r.chip.left, top: r.chip.top, transform: `translateX(-50%) scale(${r.chip.sc})`, transformOrigin: '50% 0', opacity: r.chip.o,
          display: 'flex', alignItems: 'center', gap: big ? 8 : 6, height: big ? 34 : 26, padding: big ? '0 11px' : '0 8px', borderRadius: big ? 8 : 6,
          border: `1.5px ${r.chip.locked ? 'solid' : 'dashed'} ${r.chip.locked ? 'var(--trusted)' : 'var(--ai)'}`, background: 'var(--raised)', whiteSpace: 'nowrap', boxShadow: big ? 'var(--shadow)' : undefined }}>
          <span style={{ fontFamily: MONO, fontSize: big ? 18 : 14, fontWeight: 500, fontVariantNumeric: 'tabular-nums', color: 'var(--ink)' }}>
            {r.chip.val}<span style={{ fontWeight: 300, color: 'var(--muted)', fontSize: big ? 14 : 12 }}>{r.chip.locked ? ' mm' : ''}</span>
          </span>
          <span style={{ fontSize: big ? 13 : 12, fontWeight: 500, color: r.chip.locked ? 'var(--trusted)' : 'var(--ai)' }}>{r.chip.locked ? '✓ Written by you' : `${p.readerName} reading`}</span>
        </div>
      ))}
      {c0 && (
        <div style={{ position: 'absolute', left: Math.min(W - 150, ox + (c0.cx + c0.d / 2) * k + 8), top: Math.max(4, oy + c0.cy * k - 13), opacity: holeChipO,
          display: 'flex', alignItems: 'center', gap: 6, height: 26, padding: '0 8px', borderRadius: 6, border: '1.5px solid var(--accent)', background: 'var(--raised)', whiteSpace: 'nowrap' }}>
          <span style={{ fontFamily: MONO, fontSize: 14, fontWeight: 500, color: 'var(--ink)' }}>{circles.length}</span>
          <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--accent-ink)' }}>{circles.length === 1 ? 'hole' : 'holes'} · traced</span>
        </div>
      )}
      <div style={{ position: 'absolute', left: big ? 16 : 10, top: big ? 16 : 10, opacity: p.tagO, transform: `translateY(${rm ? 0 : (big ? -24 : -18) * (1 - p.tagO)}px)`,
        display: 'flex', alignItems: 'center', height: big ? 30 : 24, borderRadius: big ? 8 : 6, overflow: 'hidden',
        border: `1.5px ${p.tagAi ? 'dashed var(--ai)' : 'solid var(--trusted)'}`, background: 'var(--raised)', boxShadow: big ? 'var(--shadow)' : undefined }}>
        <span style={{ padding: big ? '0 9px' : '0 7px', height: '100%', display: 'flex', alignItems: 'center', background: p.tagAi ? 'var(--ai)' : 'var(--trusted)', color: 'var(--raised)', fontFamily: MONO, fontSize: big ? 12 : 11, fontWeight: 500, letterSpacing: '0.06em' }}>{p.faceTxt}</span>
        <span style={{ padding: big ? '0 10px' : '0 7px', fontSize: big ? 13 : 12, fontWeight: 500, color: p.tagAi ? 'var(--ai)' : 'var(--trusted)' }}>{p.tagTxt}</span>
      </div>
      {p.corner && (
        <span style={{ position: 'absolute', right: 8, bottom: 8, fontFamily: MONO, fontSize: 11, padding: '3px 6px', borderRadius: 6, background: 'var(--raised)', color: 'var(--muted)' }}>{p.corner}</span>
      )}
      {p.note && (
        <div style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', opacity: p.note.o, display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', borderRadius: 10, background: 'var(--raised)', boxShadow: 'var(--shadow)', fontSize: 14, color: 'var(--ink)', whiteSpace: 'nowrap', pointerEvents: 'none' }}>
          <span style={{ width: 14, height: 14, border: '1.5px solid var(--accent)', borderRadius: '50%', boxSizing: 'border-box' }} />{p.note.text}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- views mini-drawing

interface Rect { x: number; y: number; w: number; h: number }

/** Map points into a slot, stretching their bounding box to the slot. `flipY` for y-up (mm) data. */
function fitPts(pts: [number, number][], slot: Rect, flipY: boolean): { pts: string; map: (x: number, y: number) => [number, number]; sx: number } | null {
  if (!pts || pts.length < 2) return null;
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = slot.w / Math.max(1e-6, x1 - x0), sy = slot.h / Math.max(1e-6, y1 - y0);
  const map = (x: number, y: number): [number, number] => [slot.x + (x - x0) * sx, flipY ? slot.y + slot.h - (y - y0) * sy : slot.y + (y - y0) * sy];
  return { pts: pts.map((q) => map(q[0], q[1]).join(',')).join(' '), map, sx };
}

const FACE_STYLE: Record<'top' | 'front' | 'right', { stroke: string; fill: string; dash?: string }> = {
  top: { stroke: 'var(--accent)', fill: 'color-mix(in oklch, var(--accent) 12%, transparent)' },
  front: { stroke: 'var(--accent)', fill: 'color-mix(in oklch, var(--accent) 12%, transparent)' },
  right: { stroke: 'var(--ai)', fill: 'color-mix(in oklch, var(--ai) 14%, transparent)', dash: '2 1.4' },
};

function ViewsDrawing({ job, pb, spec, rm }: { job: Job; pb: Playback; spec: Spec | null; rm: boolean }) {
  const st = (k: StageKey) => pb.stages.find((s) => s.key === k);
  const fuse = st('fuse'), draw = st('draw');
  const fp = fuse ? fuse.p : 0;
  const sp = rm ? (fp > 0 ? 1 : 0) : back(fp / 0.55);
  const off = { tx: 0, ty: -7 * (1 - sp), fx: -4 * (1 - sp), fy: 5 * (1 - sp), rx: 8 * (1 - sp), ry: 0 };
  const projO = ez((fp - 0.45) / 0.4);

  // Slots: design layout (third-angle: top above front, right beside front), or real envelope once fused.
  let slots: Record<'top' | 'front' | 'right', Rect> = { top: { x: 0, y: 0, w: 50, h: 20 }, front: { x: 0, y: 28, w: 50, h: 30 }, right: { x: 58, y: 28, w: 20, h: 30 } };
  const useSpec = !!spec && !!fuse && fuse.state !== 'pending';
  if (useSpec && spec) {
    const { x_mm: x, y_mm: y, z_mm: z } = spec.envelope;
    const s = Math.min(70 / Math.max(1e-6, x + z), 50 / Math.max(1e-6, y + z));
    slots = { top: { x: 0, y: 0, w: x * s, h: z * s }, front: { x: 0, y: z * s + 8, w: x * s, h: y * s }, right: { x: x * s + 8, y: z * s + 8, w: z * s, h: y * s } };
  }

  const faceO = (f: 'top' | 'front' | 'right') => {
    const i = job.images.findIndex((im) => im.face === f);
    if (i >= 0) return ez((pb.trace[i] - 0.6) / 0.4);
    const show = pb.faces[f];
    if (show === 'drawing') return 0.35;
    if (show === 'ai' || show === 'assumed' || show === 'mirror') return draw ? Math.max(0.35, ez(draw.p / 0.4)) : 1;
    return 0;
  };

  const shape = (f: 'top' | 'front' | 'right') => {
    const slot = slots[f];
    const obs = job.images.find((im) => im.face === f);
    const aiFace = !obs;
    const sty = aiFace ? FACE_STYLE.right : FACE_STYLE.front;
    const base = { fill: sty.fill, stroke: sty.stroke, strokeWidth: 0.8, strokeDasharray: sty.dash, strokeLinejoin: 'round' as const };
    const hole = { fill: 'none', stroke: sty.stroke, strokeWidth: 0.7, strokeDasharray: aiFace ? '1.2 0.9' : undefined };
    if (useSpec && spec) {
      const o = spec.views[f];
      const fit = o ? fitPts(o.outer, slot, true) : null;
      const s = slot.w / Math.max(1e-6, f === 'right' ? spec.envelope.z_mm : spec.envelope.x_mm);
      const holes = spec.features.filter((h) => h.face === f && h.type === 'hole');
      return (
        <>
          {fit ? <polygon points={fit.pts} style={base} /> : <rect x={slot.x} y={slot.y} width={slot.w} height={slot.h} style={base} />}
          {holes.map((h, i) => h.type === 'hole' && (
            <circle key={i} cx={slot.x + h.a_mm * s} cy={slot.y + slot.h - h.b_mm * s} r={(h.diameter_mm / 2) * s} style={hole} />
          ))}
        </>
      );
    }
    if (obs?.outline) {
      const fit = fitPts(obs.outline, slot, false);
      if (fit) {
        return (
          <>
            <polygon points={fit.pts} style={base} />
            {obs.circles.map((c, i) => { const [cx, cy] = fit.map(c.cx, c.cy); return <circle key={i} cx={cx} cy={cy} r={Math.max(0.8, (c.d / 2) * fit.sx)} style={hole} />; })}
          </>
        );
      }
    }
    return <rect x={slot.x} y={slot.y} width={slot.w} height={slot.h} style={base} />;
  };

  const T = slots.top, F = slots.front, R = slots.right;
  return (
    <svg viewBox="-8 -8 94 76" style={{ width: 156, height: 130, marginTop: 6, overflow: 'visible' }}>
      <g style={{ transform: `translate(${off.tx}px, ${off.ty}px)`, opacity: faceO('top') }}>{shape('top')}</g>
      <g style={{ transform: `translate(${off.fx}px, ${off.fy}px)`, opacity: faceO('front') }}>{shape('front')}</g>
      <g style={{ transform: `translate(${off.rx}px, ${off.ry}px)`, opacity: faceO('right') }}>{shape('right')}</g>
      <g style={{ opacity: projO, stroke: 'var(--muted)', strokeWidth: 0.4, strokeDasharray: '1 1' }}>
        <line x1={T.x} y1={T.y + T.h} x2={F.x} y2={F.y} />
        <line x1={T.x + T.w} y1={T.y + T.h} x2={F.x + F.w} y2={F.y} />
        <line x1={F.x + F.w} y1={F.y} x2={R.x} y2={R.y} />
        <line x1={F.x + F.w} y1={F.y + F.h} x2={R.x} y2={R.y + R.h} />
      </g>
    </svg>
  );
}

// ---------------------------------------------------------------- coverage cube

type FaceStyle = { bs: string; bc: string; bg: string; tc: string; label: string };
const FS: Record<FaceShow, FaceStyle> = {
  empty: { bs: 'solid', bc: 'var(--line)', bg: 'var(--inset)', tc: 'var(--muted)', label: '—' },
  photo: { bs: 'solid', bc: 'var(--trusted)', bg: 'color-mix(in oklch, var(--trusted) 18%, var(--surface))', tc: 'var(--trusted)', label: '✓' },
  pending: { bs: 'dashed', bc: 'var(--ai)', bg: 'color-mix(in oklch, var(--ai) 8%, var(--surface))', tc: 'var(--ai)', label: 'AI draws' },
  drawing: { bs: 'dashed', bc: 'var(--ai)', bg: 'color-mix(in oklch, var(--ai) 20%, var(--surface))', tc: 'var(--ink)', label: 'Drawing…' },
  ai: { bs: 'dashed', bc: 'var(--ai)', bg: 'color-mix(in oklch, var(--ai) 18%, var(--surface))', tc: 'var(--ai)', label: 'AI-drawn' },
  mirror: { bs: 'dotted', bc: 'var(--muted)', bg: 'var(--surface)', tc: 'var(--muted)', label: 'mirrored' },
  assumed: { bs: 'dotted', bc: 'var(--check)', bg: 'var(--surface)', tc: 'var(--check)', label: 'assumed' },
};

function CubeFace({ name, f, bi, pos }: { name: string; f: FaceStyle; bi?: string; pos?: string }) {
  return (
    <div style={{ position: 'absolute', inset: 0, boxSizing: 'border-box', border: `1.5px ${f.bs} ${f.bc}`, backgroundColor: f.bg, backgroundImage: bi ?? 'none', backgroundPosition: pos, borderRadius: 4, padding: '5px 6px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: '0.05em', color: 'var(--muted)' }}>{name}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: f.tc }}>{f.label}</span>
    </div>
  );
}

// ---------------------------------------------------------------- screen

export function Analyzing() {
  const { state, dispatch } = useStore();
  const { jobId, job, jobItems: items, analysis } = state;
  const rm = useReducedMotion();
  const now = useNow();
  const startedAt = useRef(Date.now());
  const [pipe, setPipe] = useState(false);
  const [netErr, setNetErr] = useState<string | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [natural, setNatural] = useState<Record<number, [number, number]>>({});
  const jobRef = useRef(job);
  jobRef.current = job;

  // Poll the job every 400 ms while it runs.
  useEffect(() => {
    if (!jobId) return;
    if (jobRef.current && jobRef.current.job_id === jobId && jobRef.current.status !== 'running') return;
    let alive = true, timer: number | undefined, fails = 0;
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (!alive) return;
        fails = 0;
        setNetErr(null);
        dispatch({ type: 'JOB_UPDATE', job: j });
        if (j.status !== 'running') return;
      } catch (e) {
        if (!alive) return;
        fails += 1;
        const msg = e instanceof ApiError ? e.message : 'Could not reach the server.';
        if ((e instanceof ApiError && (e.status === 404 || e.status === 410)) || fails >= MAX_FAILS) {
          setFatal(msg);
          dispatch({ type: 'JOB_LOST', error: msg });
          return;
        }
        setNetErr(msg);
      }
      timer = window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [jobId, dispatch]);

  const pb = useMemo(() => (job ? pace(job, startedAt.current, now) : null), [job, now]);
  const result = job?.result ?? null;
  const spec = result?.spec ?? null;
  const abstain = result?.abstain ?? null;
  const done = !!pb?.allDone;

  // Hand the analysis over only once playback has caught up, so Review never unlocks ahead of the screen.
  useEffect(() => {
    if (done && result && !analysis) dispatch({ type: 'ANALYSIS', analysis: result });
  }, [done, result, analysis, dispatch]);

  const toCapture = () => dispatch({ type: 'GOTO', screen: 'capture' });
  const onCancel = async () => {
    if (jobId && job?.status !== 'done') {
      try { await cancelJob(jobId); } catch { /* the job may already be gone; leaving is still right */ }
    }
    toCapture();
  };

  const wrap = (children: ReactNode) => (
    <div data-screen="Analyzing" style={{ flex: 1, minWidth: 0, boxSizing: 'border-box', padding: 28, display: 'flex', flexDirection: 'column', gap: 20 }}>{children}</div>
  );

  if (!jobId) {
    return wrap(<StopCard kicker="[ NO ANALYSIS ]" title="Nothing is being analyzed" remedy="Add your sketches or photos on the Capture screen, then press Analyze." actionLabel="Back to capture" onAction={toCapture} />);
  }
  if (job && (job.status === 'failed' || job.status === 'cancelled')) {
    return wrap(<StopCard title={job.status === 'failed' ? 'The analysis stopped' : 'The analysis was cancelled'} remedy={job.error ?? 'Try again or use different photos.'} actionLabel="Back to capture" onAction={toCapture} />);
  }
  if (fatal) {
    return wrap(<StopCard title="Lost the analysis" remedy={fatal} actionLabel="Back to capture" onAction={toCapture} />);
  }
  if (job && job.status === 'done' && !result) {
    return wrap(<StopCard title="The analysis finished without a result" remedy="Try again or use different photos." actionLabel="Back to capture" onAction={toCapture} />);
  }

  const t = pb?.t ?? 0;
  const S = (k: StageKey) => pb?.stages.find((s) => s.key === k);
  const label = S('label'), outline = S('outline'), read = S('read'), draw = S('draw'), fuse = S('fuse');
  const spin = Math.round((t * 420) % 360);
  const images = job?.images ?? [];
  const n = Math.max(images.length, items.length);

  // Header copy from what the user actually sent.
  const kinds = new Set(images.map((im, i) => im.kind ?? (items[i]?.kind !== 'auto' ? items[i]?.kind : null)));
  const kind = kinds.size === 1 ? [...kinds][0] : null;
  const noun = kind === 'sketch' ? 'sketch' : kind === 'photo' ? 'photo' : kind === 'drawing' ? 'drawing' : 'image';
  const plural = n === 1 ? noun : noun === 'sketch' ? 'sketches' : `${noun}s`;
  const noRef = !state.reference || state.reference === 'none';
  const intro = `Reading your ${WORDS[n] ?? n} ${plural}. ${noRef ? 'Every size comes from your writing, never guessed.' : 'Every size is measured or comes from your writing, never guessed.'}`;

  // Per-image playback.
  let base = 0;
  const readBase = images.map((im) => { const b = base; base += im.reads.length; return b; });
  const totalReads = base;
  const tagFor = (i: number) => {
    const im = images[i];
    const item = items[i];
    const face = im?.face && im.face !== 'unknown' ? im.face : item && item.face !== 'auto' ? item.face : null;
    const byUser = !!item && item.face !== 'auto';
    const ai = !byUser && !!label?.ai;
    const lp = label ? label.p : 0;
    const start = n > 1 ? (i / n) * 0.5 : 0;
    return { faceTxt: face ? face.toUpperCase() : '?', tagTxt: ai ? 'labelled by AI' : 'tagged by you', tagAi: ai, tagO: ez((lp - start) / 0.5) };
  };
  const pane = (i: number, W: number, H: number, big: boolean, extra?: Partial<PaneProps>) => (
    <SketchPane key={i} W={W} H={H} big={big} url={items[i]?.url} image={images[i]} natural={natural[i]}
      onNatural={(wh) => setNatural((m) => (m[i] && m[i][0] === wh[0] && m[i][1] === wh[1] ? m : { ...m, [i]: wh }))}
      trace={pb?.trace[i] ?? 0} reads={pb?.readsShown[i] ?? 0} readBase={readBase[i] ?? 0} totalReads={totalReads}
      read={read} readerName={read?.tool ?? 'the reader'} t={t} rm={rm} {...tagFor(i)} {...extra} />
  );

  // The thumbnail follows whichever later image is being traced or read.
  let thumb = 1;
  if (pb) {
    for (let i = 1; i < images.length; i++) if (pb.trace[i] > 0) thumb = i;
    for (let i = 1; i < images.length; i++) if (pb.readsShown[i] > 0) thumb = i;
  }

  const noteO = outline && Number.isFinite(outline.start) && noRef ? Math.min(ez((t - outline.start) / 0.2), 1 - ez((t - outline.start - 0.8) / 0.2)) : 0;

  // Stage list and headline.
  const stages = (pb?.stages ?? []).map((s) => {
    const st = s.state, run = st === 'running';
    const fin = st === 'done', skip = st === 'skipped', bad = st === 'failed';
    return {
      key: s.key, name: COPY[s.key].name, tool: s.tool, kind: s.ai ? 'AI' : 'CODE', detail: s.detail,
      kindBg: s.ai ? 'var(--ai)' : 'var(--line)', kindC: s.ai ? 'var(--raised)' : 'var(--ink)',
      toolC: s.ai ? 'var(--ai)' : 'var(--ink)', toolBd: s.ai ? 'var(--ai)' : 'var(--line)', toolLine: s.ai ? 'dashed' : 'solid',
      bg: run ? 'var(--raised)' : 'transparent', sh: run ? 'var(--shadow)' : 'none',
      nameC: st === 'pending' || skip ? 'var(--muted)' : 'var(--ink)',
      glyph: fin ? '✓' : skip ? '–' : bad ? '!' : '', glyphC: skip ? 'var(--muted)' : 'var(--raised)',
      nodeC: fin ? (s.ai ? 'var(--ai)' : 'var(--ink)') : bad ? 'var(--stop)' : run ? 'transparent' : skip ? 'var(--muted)' : 'var(--line)',
      nodeBg: fin ? (s.ai ? 'var(--ai)' : 'var(--ink)') : bad ? 'var(--stop)' : 'transparent', nodeLine: skip ? 'dashed' : 'solid',
      spinO: run && !rm ? 1 : 0,
    };
  });
  const cur = pb ? pb.stages[pb.cur] : undefined;
  const nStages = pb?.stages.length ?? 5;
  const nCheck = spec ? countChecks(spec, state.typed) : 0;
  const nValues = spec ? envelopeRows(spec).length + featureRows(spec).length : 0;
  const curNote = (() => {
    if (done) return abstain ? abstain.remedy : nCheck ? `${nCheck} ${nCheck === 1 ? 'value needs' : 'values need'} a quick look` : 'Nothing to check — every value is yours';
    if (!cur) return 'Starting…';
    const revealed = Number.isFinite(cur.reveal) && cur.detail;
    return (cur.ai ? `AI · ${cur.tool} — ` : '') + (revealed ? cur.detail : COPY[cur.key].run);
  })();
  const curC = done ? (abstain ? 'var(--stop)' : nCheck ? 'var(--check)' : 'var(--trusted)') : cur?.ai ? 'var(--ai)' : 'var(--muted)';

  // Views panel tag.
  const over = (s?: PacedStage) => !!s && s.state !== 'pending' && s.state !== 'running';
  const fuseTag = over(fuse) ? (abstain ? 'STOPPED' : 'FUSED ✓') : fuse?.state === 'running' ? 'FUSING' : draw?.state === 'running' && draw.ai ? 'AI DRAWING' : over(outline) ? 'FROM SKETCH' : 'WAITING';
  const fuseTagColor = over(fuse) ? (abstain ? 'var(--stop)' : 'var(--trusted)') : draw?.state === 'running' && draw.ai ? 'var(--ai)' : 'var(--muted)';

  // Coverage cube.
  const faces = pb?.faces ?? { front: 'empty', top: 'empty', right: 'empty', back: 'empty', left: 'empty', bottom: 'empty' };
  const faceStyle = (f: Face): FaceStyle => {
    const s = FS[faces[f]];
    if (faces[f] !== 'photo') return s;
    const c = images.filter((im) => im.face === f).length || 1;
    return { ...s, label: `✓ ×${c}` };
  };
  const covered = (Object.values(faces) as FaceShow[]).filter((v) => v !== 'empty' && v !== 'pending' && v !== 'drawing').length;
  const anyAssumed = (Object.values(faces) as FaceShow[]).includes('assumed');
  const fp = rm ? (fuse && fuse.p >= 1 ? 1 : 0) : ez(((fuse?.p ?? 0) - 0.3) / 0.7);
  const cube = {
    root: `translateX(${27 * fp}px) translateZ(${-27 * fp}px) rotateX(${-24 * fp}deg) rotateY(${38 * fp}deg) translateZ(${27 * fp}px)`,
    top: `rotateX(${90 * fp}deg)`, bottom: `rotateX(${-90 * fp}deg)`, left: `rotateY(${-90 * fp}deg)`, right: `rotateY(${90 * fp}deg)`, back: `rotateY(${90 * fp}deg)`,
  };
  const shimmer = faces.right === 'drawing' && !rm ? 'repeating-linear-gradient(135deg, color-mix(in oklch, var(--ai) 30%, transparent) 0 5px, transparent 5px 11px)' : 'none';
  const shimmerPos = `${(t * 24) % 22}px 0`;

  // Ledger: read values while reading, the fused envelope once the part exists.
  type Row = { key: string; name: string; val: string; o: number; x: number; badge: string; line: string; bc: string; bg: string };
  let ledger: Row[] = [];
  let ledgerNote: string | null = null;
  let more = 0;
  if (spec && over(fuse)) {
    ledger = envelopeRows(spec).map((r) => {
      const b = BADGE[r.prov];
      return { key: r.path, name: r.label, val: fmt(r.value), o: 1, x: 0, badge: `${b.icon} ${b.label}`, line: b.line, bc: b.fg, bg: b.bg };
    });
  } else if (read && read.state === 'skipped') {
    ledgerNote = 'No reader switched on — you will type the sizes in Review.';
  } else if (read && Number.isFinite(read.reveal)) {
    const all = images.flatMap((im, i) => im.reads.map((r, k) => ({ r, i, k, face: im.face })));
    if (!all.length) ledgerNote = 'No written sizes found — you will add them in Review.';
    const shown = all.slice(0, 4);
    more = all.length - shown.length;
    ledger = shown.map(({ r, i, k, face }) => {
      const local = (pb?.readsShown[i] ?? 0) - k;
      const on = local >= 1;
      const lockAt = read.reveal + ((readBase[i] + k + 1) * (read.end - read.reveal)) / Math.max(1, totalReads);
      const e = on ? ez((t - lockAt) / 0.4) : 0;
      const name = r.kind === 'diameter' ? 'Ø' : r.kind === 'radius' ? 'R' : face && face !== 'unknown' ? face[0].toUpperCase() + face.slice(1) : 'Size';
      return { key: `${i}-${k}`, name, val: on ? fmt(r.value_mm) : '—', o: 0.35 + 0.65 * e, x: rm ? 0 : -8 * (1 - e),
        badge: on ? '✓ Written by you' : 'reading…', line: on ? 'solid' : 'dashed', bc: on ? 'var(--trusted)' : 'var(--muted)', bg: on ? 'color-mix(in oklch, var(--trusted) 12%, transparent)' : 'transparent' };
    });
  } else if (read && read.state === 'running') {
    ledger = [{ key: 'wait', name: '…', val: '—', o: 0.35, x: 0, badge: 'reading…', line: 'dashed', bc: 'var(--muted)', bg: 'transparent' }];
  } else {
    ledgerNote = 'Waiting for your numbers';
  }

  const stopCta = abstain ? analyzingCta(abstain) : null;
  const ctaLabel = !done ? 'Review opens when ready' : stopCta ? stopCta.label : `Review ${nCheck || nValues} ${(nCheck || nValues) === 1 ? 'value' : 'values'} →`;
  const readyO = done ? 1 : 0;
  const ready = abstain
    ? { c: 'var(--stop)', icon: '!', text: `Stopped — ${abstain.remedy}` }
    : nCheck ? { c: 'var(--check)', icon: '!', text: `Ready — ${nCheck} ${nCheck === 1 ? 'value' : 'values'} to check` }
      : { c: 'var(--trusted)', icon: '✓', text: 'Ready — every value is yours' };

  return wrap(
    <>
      <header style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'flex-end', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 13, letterSpacing: '0.08em', color: 'var(--muted)' }}>[1/3] CAPTURE</span>
          <h1 style={{ margin: 0, fontFamily: "'Silkscreen', monospace", fontWeight: 400, fontSize: 40, lineHeight: 1, letterSpacing: '0.01em' }}>Analyzing</h1>
        </div>
        <p style={{ margin: '0 0 4px', maxWidth: 380, fontSize: 15, lineHeight: 1.4, color: 'var(--muted)', textWrap: 'pretty' } as CSSProperties}>{intro}</p>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span title="Real pipeline events, shown at a readable pace" style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.08em', padding: '5px 8px', border: '1px solid var(--line)', borderRadius: 6, color: netErr ? 'var(--stop)' : 'var(--muted)', whiteSpace: 'nowrap' }}>
            {netErr ? 'RECONNECTING…' : 'LIVE · PACED FOR READING'}
          </span>
          <button type="button" onClick={onCancel} style={{ height: 44, padding: '0 20px', borderRadius: 10, border: '1px solid var(--line)', background: 'var(--raised)', color: 'var(--ink)', font: 'inherit', fontSize: 15, cursor: 'pointer', boxShadow: 'var(--shadow)' }}>Cancel</button>
        </div>
      </header>

      {netErr && <div role="status" style={{ fontSize: 13, color: 'var(--stop)', marginTop: -8 }}>{netErr} Retrying…</div>}

      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: '740px minmax(340px,1fr)', gap: 24 }}>
        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          {pane(0, 740, 529, true, { note: { o: noteO, text: 'No coin or card chosen — sizes will come from your writing' } })}

          <div style={{ display: 'grid', gridTemplateColumns: '252px 280px 176px', gap: 16, height: 190 }}>
            {n > 1
              ? pane(thumb, 252, 190, false, { corner: n > 2 ? `${thumb + 1}/${n}` : undefined })
              : <div style={{ width: 252, height: 190, borderRadius: 12, background: 'var(--inset)', display: 'grid', placeItems: 'center', fontSize: 13, color: 'var(--muted)' }}>One view sent</div>}

            <div style={{ borderRadius: 12, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '14px 16px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', gap: 10, overflow: 'hidden' }}>
              <span style={{ fontSize: 14, fontWeight: 600 }}>{spec && over(fuse) ? 'Part size' : 'Sizes read'}</span>
              {ledger.map((r) => (
                <div key={r.key} style={{ display: 'grid', gridTemplateColumns: '52px minmax(0,1fr) auto', alignItems: 'center', gap: 6, opacity: r.o, transform: `translateX(${r.x}px)` }}>
                  <span style={{ fontSize: 13, color: 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}</span>
                  <span style={{ fontFamily: MONO, fontSize: 18, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{r.val}<span style={{ fontSize: 12, fontWeight: 300, color: 'var(--muted)' }}> mm</span></span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, height: 24, padding: '0 8px', borderRadius: 12, border: `1.5px ${r.line} ${r.bc}`, background: r.bg, color: r.bc, fontSize: 12, fontWeight: 500, whiteSpace: 'nowrap' }}>{r.badge}</span>
                </div>
              ))}
              {more > 0 && <span style={{ fontSize: 12, color: 'var(--muted)' }}>+{more} more in Review</span>}
              {ledgerNote && <span style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.4 }}>{ledgerNote}</span>}
            </div>

            <div style={{ borderRadius: 12, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '12px 14px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 14, fontWeight: 600 }}>Views</span>
                <span style={{ fontFamily: MONO, fontSize: 10, letterSpacing: '0.06em', color: fuseTagColor }}>{fuseTag}</span>
              </div>
              {job && pb && <ViewsDrawing job={job} pb={pb} spec={spec} rm={rm} />}
            </div>
          </div>
        </section>

        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '60px minmax(0,1fr) auto', gap: 14, alignItems: 'center' }}>
              <div style={{ position: 'relative', width: 60, height: 60 }}>
                <svg viewBox="0 0 60 60" style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}>
                  <circle cx="30" cy="30" r="25" style={{ fill: 'none', stroke: 'var(--inset)', strokeWidth: 6 }} />
                  <circle cx="30" cy="30" r="25" style={{ fill: 'none', stroke: done ? 'var(--trusted)' : 'var(--accent)', strokeWidth: 6, strokeLinecap: 'round', strokeDasharray: 157.1, strokeDashoffset: 157.1 * (1 - (pb?.progress ?? 0)) }} />
                </svg>
                <div style={{ position: 'absolute', inset: -5, borderRadius: '50%', border: '2px solid transparent', borderTopColor: 'var(--accent)', opacity: done || rm ? 0 : 1, transform: `rotate(${spin}deg)` }} />
                <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 14, fontWeight: 500, color: done ? 'var(--trusted)' : 'var(--ink)' }}>{done ? '✓' : `${(pb?.cur ?? 0) + 1}/${nStages}`}</div>
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.1em', color: 'var(--muted)' }}>{done ? `ALL ${nStages} STEPS DONE` : `STEP ${(pb?.cur ?? 0) + 1} OF ${nStages}`}</div>
                <div style={{ fontSize: 17, fontWeight: 600, marginTop: 3 }}>{done ? (abstain ? 'Needs your input' : 'Ready for your review') : cur ? COPY[cur.key].name : 'Starting the analysis'}</div>
                <div style={{ fontSize: 13, color: curC, marginTop: 3, textWrap: 'pretty' } as CSSProperties}>{curNote}</div>
              </div>
              <button type="button" onClick={() => setPipe((v) => !v)} aria-expanded={pipe} aria-label="Show all steps" style={{ width: 44, height: 44, borderRadius: 10, border: '1px solid var(--line)', background: 'var(--raised)', color: 'var(--ink)', fontFamily: MONO, fontSize: 16, cursor: 'pointer' }}>{pipe ? '−' : '+'}</button>
            </div>
            {pipe && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginTop: 14, padding: 4, borderRadius: 12, background: 'var(--inset)' }}>
                {stages.map((s) => (
                  <div key={s.key} title={s.detail || undefined} style={{ display: 'grid', gridTemplateColumns: '22px minmax(0,1fr) auto', alignItems: 'center', gap: 10, height: 42, padding: '0 10px', borderRadius: 9, background: s.bg, boxShadow: s.sh }}>
                    <span style={{ position: 'relative', width: 20, height: 20, borderRadius: '50%', boxSizing: 'border-box', border: `1.5px ${s.nodeLine} ${s.nodeC}`, background: s.nodeBg, color: s.glyphC, display: 'grid', placeItems: 'center', fontSize: 11, fontWeight: 700 }}>
                      {s.glyph}
                      <span style={{ position: 'absolute', inset: -1.5, borderRadius: '50%', border: '2px solid transparent', borderTopColor: 'var(--accent)', opacity: s.spinO, transform: `rotate(${spin}deg)` }} />
                    </span>
                    <span style={{ fontSize: 14, fontWeight: 500, color: s.nameC, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.name}</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', height: 22, borderRadius: 6, border: `1px ${s.toolLine} ${s.toolBd}`, overflow: 'hidden', fontFamily: MONO, fontSize: 11, whiteSpace: 'nowrap' }}>
                      <span style={{ padding: '0 5px', height: '100%', display: 'flex', alignItems: 'center', background: s.kindBg, color: s.kindC, fontWeight: 500 }}>{s.kind}</span>
                      <span style={{ padding: '0 6px', color: s.toolC }}>{s.tool}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Coverage</span>
              <span style={{ fontSize: 13, color: 'var(--muted)', whiteSpace: 'nowrap' }}><span style={{ fontFamily: MONO, fontSize: 14, color: 'var(--ink)' }}>{covered}</span> of 6 faces</span>
            </div>
            <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
              <div style={{ position: 'relative', width: 216, height: 162, flex: 'none', perspective: 800 }}>
                <div style={{ position: 'absolute', left: 54, top: 54, width: 54, height: 54, transformStyle: 'preserve-3d', transform: cube.root }}>
                  <CubeFace name="FRONT" f={faceStyle('front')} />
                  <div style={{ position: 'absolute', left: 0, top: -54, width: 54, height: 54, transformOrigin: '50% 100%', transform: cube.top }}><CubeFace name="TOP" f={faceStyle('top')} /></div>
                  <div style={{ position: 'absolute', left: 0, top: 54, width: 54, height: 54, transformOrigin: '50% 0', transform: cube.bottom }}><CubeFace name="BOTTOM" f={faceStyle('bottom')} /></div>
                  <div style={{ position: 'absolute', left: -54, top: 0, width: 54, height: 54, transformOrigin: '100% 50%', transform: cube.left }}><CubeFace name="LEFT" f={faceStyle('left')} /></div>
                  <div style={{ position: 'absolute', left: 54, top: 0, width: 54, height: 54, transformOrigin: '0 50%', transformStyle: 'preserve-3d', transform: cube.right }}>
                    <CubeFace name="RIGHT" f={faceStyle('right')} bi={shimmer} pos={shimmerPos} />
                    <div style={{ position: 'absolute', left: 54, top: 0, width: 54, height: 54, transformOrigin: '0 50%', transform: cube.back }}><CubeFace name="BACK" f={faceStyle('back')} /></div>
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 9, fontSize: 12, color: 'var(--muted)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><span style={{ width: 14, height: 10, border: '1.5px solid var(--trusted)', background: 'color-mix(in oklch, var(--trusted) 18%, transparent)', borderRadius: 2 }} />Photographed</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><span style={{ width: 14, height: 10, border: '1.5px dashed var(--ai)', background: 'color-mix(in oklch, var(--ai) 14%, transparent)', borderRadius: 2 }} />AI draws it</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><span style={{ width: 14, height: 10, border: '1.5px dotted var(--muted)', borderRadius: 2 }} />Mirrored</span>
                {anyAssumed && <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><span style={{ width: 14, height: 10, border: '1.5px dotted var(--check)', borderRadius: 2 }} />Assumed</span>}
                <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><span style={{ width: 14, height: 10, border: '1.5px solid var(--line)', background: 'var(--inset)', borderRadius: 2 }} />Empty</span>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ opacity: readyO, transition: rm ? undefined : 'opacity 400ms', display: 'flex', alignItems: 'center', gap: 10, padding: '11px 14px', borderRadius: 10, border: `1.5px ${ready.c === 'var(--trusted)' ? 'solid' : 'dashed'} ${ready.c}`, background: `color-mix(in oklch, ${ready.c} 12%, transparent)`, fontSize: 14, fontWeight: 500, color: 'var(--ink)' }}>
              <span style={{ width: 20, height: 20, flex: 'none', borderRadius: '50%', border: `1.5px dashed ${ready.c}`, color: ready.c, display: 'grid', placeItems: 'center', fontSize: 11, fontWeight: 700, boxSizing: 'border-box' }}>{ready.icon}</span>{ready.text}
            </div>
            <button type="button" aria-disabled={!done} disabled={!done} onClick={() => dispatch({ type: 'GOTO', screen: stopCta?.to ?? 'review' })}
              style={{ height: 52, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, border: 'none', background: done ? 'var(--accent)' : 'var(--inset)', color: done ? 'var(--on-accent)' : 'var(--muted)', boxShadow: done ? 'var(--shadow)' : 'none', font: 'inherit', fontSize: 16, fontWeight: 600, cursor: done ? 'pointer' : 'default' }}>
              {ctaLabel}
            </button>
          </div>
        </section>
      </div>
    </>,
  );
}
