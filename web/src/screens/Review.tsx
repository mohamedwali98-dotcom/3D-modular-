import { useCallback, useEffect, useRef, useState, type ChangeEvent, type CSSProperties } from 'react';
import { ApiError, buildModel, merge } from '../api/client';
import type { Abstain, Face, FilledBy, JobImage, ReadValue, Spec } from '../api/types';
import { FaceCard } from '../components/FaceCard';
import { StopCard } from '../components/StopCard';
import { EXPIRED, isDimensionAbstain } from '../lib/abstain';
import { snappedLabel } from '../lib/snap';
import { listPhrase } from '../lib/words';
import { BADGE, countChecks, featureRows, isCheck, provOf, type BadgeKey } from '../lib/provenance';
import { useStore } from '../state/store';

const MONO = "'Geist Mono', monospace";
const TRU = 'var(--trusted)', CHK = 'var(--check)', AI = 'var(--ai)', STOP = 'var(--stop)';

// Same three axes as `envelopeRows`, kept local so the required/abstain box can be built without a spec.
const ENV_META = [
  { key: 'x', label: 'Width', axis: 'X' },
  { key: 'y', label: 'Height', axis: 'Y' },
  { key: 'z', label: 'Depth', axis: 'Z' },
] as const;

const EXPLAIN: Record<BadgeKey, string> = {
  user_written: 'You wrote this value.',
  measured: 'Measured on your drawing (its dimensions give the scale) or from the coin or reference in your photo.',
  user_edited: 'You typed this value.',
  scaled: 'Scaled from the overall size you entered.',
  inferred: 'Drawn by the AI from your sketches. Check it against your part.',
  estimated: 'Estimated from the sketch. Type the real value if you know it.',
  default: 'No value was given, so a standard size was used. Confirm or change it.',
  required: 'We need this value before we can build.',
  found: 'Kept from your images while the analysis waits for the missing size. Check it.',
};

const errText = (e: unknown) => (e instanceof ApiError ? e.message : 'Something went wrong. Try again.');

const CSS = `
@media (max-width:1023px){
  .s2c-review-grid{grid-template-columns:minmax(0,1fr)!important}
  .s2c-review-header{height:auto!important;flex-wrap:wrap}
}`;

interface LedgerGroup {
  group: string;
  name: string;
  face: Face;
  feature: string;
  prov: BadgeKey;
  snapped: boolean;
  /** "M5" when the snapped size is exactly a standard clearance hole, otherwise "standard size". */
  snapLabel: string;
  fields: { path: string; label: string; aria: string; value: number }[];
  index: number;        // the feature's index in spec.features
  removable: boolean;   // a pocket or a pin read from a drawing's lines: the user can take it out
}

/** Groups `featureRows` by their check-group (size/position/angle/depth) in first-seen order. */
function groupFeatures(spec: Spec, typed: Record<string, number>): LedgerGroup[] {
  const rows = featureRows(spec, typed);
  const order: string[] = [];
  const byGroup = new Map<string, typeof rows>();
  for (const r of rows) {
    if (!byGroup.has(r.group)) { byGroup.set(r.group, []); order.push(r.group); }
    byGroup.get(r.group)!.push(r);
  }
  return order.map((group) => {
    const rs = byGroup.get(group)!;
    const anyTyped = rs.some((r) => r.path in typed);
    const prov: BadgeKey = anyTyped ? 'user_edited' : rs[0].prov;
    const snappedRow = rs.find((r) => r.snapped);
    const snapped = !!snappedRow && rs[0].prov === 'default';
    return {
      group, name: rs[0].groupName, face: rs[0].face, feature: rs[0].feature, prov, snapped,
      snapLabel: snappedLabel(snappedRow && (typed[snappedRow.path] ?? snappedRow.value)),
      fields: rs.map((r) => ({ path: r.path, label: r.label, aria: `${r.groupName} ${r.label}`, value: typed[r.path] ?? r.value })),
      index: rs[0].index, removable: /^(Pocket|Pin) /.test(rs[0].feature),
    };
  });
}

/** Icon/colour for a warning string, and whether the feature it names has been fully resolved. */
function warningMeta(text: string, spec: Spec, typed: Record<string, number>): { icon: string; color: string; line: 'solid' | 'dashed' } {
  const isAiText = /qwen-image|drawn/i.test(text);
  const m = /^(hole|slot)\s+(\d+)\s*:/i.exec(text.trim());
  let done = false;
  if (m) {
    const featureLabel = `${m[1][0].toUpperCase()}${m[1].slice(1).toLowerCase()} ${m[2]}`;
    const rows = featureRows(spec, typed).filter((r) => r.feature === featureLabel);
    if (rows.length) done = rows.every((r) => !isCheck(r.path in typed ? 'user_edited' : r.prov));
  }
  if (done) return { icon: '✓', color: TRU, line: 'solid' };
  if (isAiText) return { icon: '◇', color: AI, line: 'dashed' };
  return { icon: '!', color: CHK, line: 'dashed' };
}

interface CropInfo { image: JobImage; url: string; read: ReadValue; fieldLabel: string }
interface Sent { typed: Record<string, number>; rejected: Face[] }

/** Draws the local image, cropped to the read's bbox (scaled from the job image's natural size) into a small canvas. */
function HandwritingCrop({ image, url, read, fieldLabel }: CropInfo) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const sx = image.width > 0 ? img.naturalWidth / image.width : 1;
      const sy = image.height > 0 ? img.naturalHeight / image.height : 1;
      const [bx, by, bw, bh] = read.bbox;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, bx * sx, by * sy, Math.max(1, bw * sx), Math.max(1, bh * sy), 0, 0, canvas.width, canvas.height);
    };
    img.src = url;
    return () => { cancelled = true; };
  }, [url, image.width, image.height, read.bbox]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '96px 18px minmax(0,1fr)', alignItems: 'center', gap: 10 }}>
      <canvas ref={canvasRef} width={96} height={56} style={{ width: 96, height: 56, borderRadius: 8, border: `1.5px dashed ${AI}`, boxSizing: 'border-box', background: '#f4f3f0' }} />
      <span style={{ color: 'var(--muted)', fontFamily: MONO }}>→</span>
      <div>
        <div style={{ fontFamily: MONO, fontSize: 20, fontWeight: 500 }}>{read.value_mm}<span style={{ fontSize: 13, fontWeight: 300, color: 'var(--muted)' }}> mm</span></div>
        <div style={{ fontSize: 13, color: 'var(--muted)' }}>{fieldLabel}</div>
      </div>
    </div>
  );
}

/** Review screen: every number the pipeline produced, with its provenance, editable and re-merged live. */
export function Review() {
  const { state, dispatch } = useStore();
  const { analysis, job, jobItems, typed, rejected, geometry } = state;
  const spec = analysis?.spec ?? null;
  const abstain = analysis?.abstain ?? null;
  const requestId = analysis?.request_id || null;

  const [open, setOpen] = useState<string | null>(null);
  const [hwOpen, setHwOpen] = useState(true);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [merging, setMerging] = useState(false);
  const [mergeErr, setMergeErr] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);
  // The typed values and rejects the current analysis already reflects. Anything newer means a merge is due.
  const [applied, setApplied] = useState<Sent>({ typed, rejected });
  const [building, setBuilding] = useState(false);
  const [buildErr, setBuildErr] = useState<string | null>(null);
  const [buildAbstain, setBuildAbstain] = useState<Abstain | null>(null);

  const envRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const mergeTimer = useRef<number | undefined>(undefined);
  const pending = useRef(false);
  const mergeSeq = useRef(0);
  const alive = useRef(true);
  const latest = useRef({ typed, rejected, requestId });
  latest.current = { typed, rejected, requestId };

  // One re-merge with the latest values. Only the newest request may apply its result or clear "updating…";
  // the store drops a result whose request_id is no longer the current job.
  const runMerge = useCallback(() => {
    pending.current = false;
    window.clearTimeout(mergeTimer.current);
    const sent: Sent = { typed: latest.current.typed, rejected: latest.current.rejected };
    const id = latest.current.requestId;
    if (!id) return;
    const my = ++mergeSeq.current;
    if (alive.current) { setMerging(true); setMergeErr(null); }
    merge({ request_id: id, user_values: sent.typed, accepted: [], rejected: sent.rejected })
      .then((a) => {
        if (my !== mergeSeq.current) return;
        dispatch({ type: 'ANALYSIS', analysis: a });
        if (alive.current) setApplied(sent);
      })
      .catch((e: unknown) => {
        if (my !== mergeSeq.current || !alive.current) return;
        if (e instanceof ApiError && e.status === 404) setExpired(true);
        else setMergeErr(errText(e));
      })
      .finally(() => { if (my === mergeSeq.current && alive.current) setMerging(false); });
  }, [dispatch]);

  // Debounced re-merge: 500 ms after the last edit or reject toggle.
  useEffect(() => {
    if (!requestId || expired) return;
    if (typed === applied.typed && rejected === applied.rejected) return;
    pending.current = true;
    window.clearTimeout(mergeTimer.current);
    mergeTimer.current = window.setTimeout(runMerge, 500);
    return () => window.clearTimeout(mergeTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [typed, rejected]);

  // Leaving Review with an edit still waiting: send it now so the typed value is not lost.
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (pending.current) runMerge();
    };
  }, [runMerge]);

  const toCapture = () => dispatch({ type: 'GOTO', screen: 'capture' });

  if (!analysis) {
    return (
      <div style={{ flex: 1, minWidth: 0, boxSizing: 'border-box', padding: 28 }}>
        <StopCard title="Nothing to review yet" remedy="Go back to Capture and analyze your sketches first." actionLabel="Go to Capture" onAction={() => dispatch({ type: 'GOTO', screen: 'capture' })} />
      </div>
    );
  }

  // `missing` lists every path still required; `suggested` holds placeholder values.
  const partial = abstain?.partial ?? {};
  const suggested = abstain?.suggested ?? {};
  const missingPaths = abstain?.missing ?? [];
  const missingEnvKeys = missingPaths
    .map((p) => /^envelope\.(x|y|z)_mm$/.exec(p)?.[1] as 'x' | 'y' | 'z' | undefined)
    .filter((k): k is 'x' | 'y' | 'z' => !!k);
  const firstMissingKey = missingEnvKeys[0] ?? null;
  const missingPhrase = missingEnvKeys.length
    ? listPhrase(missingEnvKeys.map((k) => ENV_META.find((e) => e.key === k)!.label.toLowerCase()))
    : null;

  const envRows = ENV_META.map((e) => {
    const path = `envelope.${e.key}_mm`;
    const required = missingEnvKeys.includes(e.key);
    const typedVal = typed[path];
    const specVal = spec ? spec.envelope[`${e.key}_mm`] : undefined;
    const partialVal = partial[path];
    const value = typedVal ?? specVal ?? partialVal;
    const placeholder = suggested[path] !== undefined ? String(suggested[path]) : '—';
    // During an abstain there is no spec: kept values carry the provenance the server sent, or a neutral "Found".
    const prov: BadgeKey = path in typed ? 'user_edited' : required ? 'required' : spec ? provOf(spec, path)
      : abstain?.partial_provenance?.[path] ?? 'found';
    return { ...e, path, required, value, prov, placeholder };
  });

  const groups = spec ? groupFeatures(spec, typed) : [];
  const nCheck = spec ? countChecks(spec, typed) : 0;
  const touched = spec ? featureRows(spec, typed).some((r) => r.path in typed && isCheck(r.prov)) : false;
  const buildLabel = nCheck > 0 && !touched ? `Build anyway (${nCheck} unchecked) →` : 'Build part →';
  const canBuild = !abstain && !!spec && !expired;
  // A merge is waiting (debounce) or in flight: the spec on screen does not hold every typed value yet.
  const updating = merging || typed !== applied.typed || rejected !== applied.rejected;
  const retry = updating && !merging && !!mergeErr;
  const buildBlocked = building || (updating && !retry);

  const effEnv = (k: 'x' | 'y' | 'z') => typed[`envelope.${k}_mm`] ?? (spec ? spec.envelope[`${k}_mm`] : 0);
  const filledByOf = (f: Face): FilledBy => analysis.filled_by[f] ?? 'observed';

  const keptText = missingEnvKeys.length
    ? (() => {
        const others = ENV_META.filter((e) => !missingEnvKeys.includes(e.key) && partial[`envelope.${e.key}_mm`] !== undefined);
        if (!others.length) return null;
        const parts = others.map((e) => `${e.label.toLowerCase()} ${partial[`envelope.${e.key}_mm`]} mm`);
        return `We kept what we read: ${parts.join(', ')}. We never guess a size.`;
      })()
    : null;

  const readerName = job?.stages.find((s) => s.key === 'read')?.tool ?? 'the reader';
  const crops: CropInfo[] = (job?.images ?? []).flatMap((image, i) => {
    const url = jobItems[i]?.url;
    if (!url) return [];
    const faceLabel = image.face && image.face !== 'unknown' ? image.face : image.kind ?? `image ${i + 1}`;
    return image.reads.map((read) => ({ image, url, read, fieldLabel: `${faceLabel} sketch` }));
  });

  const displayValue = (path: string, computed: number | '' | undefined) =>
    drafts[path] ?? (computed === undefined || computed === '' ? '' : String(computed));

  const onFieldChange = (path: string) => (e: ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setDrafts((d) => ({ ...d, [path]: raw }));
    const num = raw.trim() === '' ? NaN : Number(raw);
    if (Number.isFinite(num)) dispatch({ type: 'TYPE_VALUE', path, value: num });
  };

  const confirmGroup = (g: LedgerGroup) => {
    for (const f of g.fields) dispatch({ type: 'TYPE_VALUE', path: f.path, value: f.value });
    setOpen(null);
  };

  const onBuild = async () => {
    if (!spec || abstain || building || updating || expired) return;
    setBuilding(true);
    setBuildErr(null);
    setBuildAbstain(null);
    try {
      const result = await buildModel({ request_id: requestId, spec, geometry });
      dispatch({ type: 'MODEL', model: result, spec });
      if (result.abstain) setBuildAbstain(result.abstain);
      else dispatch({ type: 'GOTO', screen: 'model' });
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) setExpired(true);
      else setBuildErr(errText(e));
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div data-screen="Review" style={{ flex: 1, minWidth: 0, minHeight: 0, boxSizing: 'border-box', padding: 28, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <style>{CSS}</style>
      <header className="s2c-review-header" style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'flex-end', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 13, letterSpacing: '0.08em', color: 'var(--muted)' }}>[2/3] REVIEW</span>
          <h1 style={{ margin: 0, fontFamily: "'Silkscreen', monospace", fontWeight: 400, fontSize: 40, lineHeight: 1 }}>Review</h1>
        </div>
        <p style={{ margin: '0 0 4px', maxWidth: 400, fontSize: 15, lineHeight: 1.4, color: 'var(--muted)', textWrap: 'pretty' } as CSSProperties}>
          Check every number before we build. Tap a badge to see where it came from.
        </p>
        {merging && <span style={{ marginLeft: 'auto', fontFamily: MONO, fontSize: 12, color: 'var(--muted)' }}>updating…</span>}
      </header>

      <div className="s2c-review-grid" style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: '500px minmax(0,1fr)', gap: 24 }}>
        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          {expired && (
            <StopCard kicker="[ EXPIRED ]" title={EXPIRED.title} remedy={EXPIRED.remedy} actionLabel={EXPIRED.action} onAction={toCapture} />
          )}
          {!expired && abstain && (
            <StopCard
              kicker={`[ STOPPED AT · ${abstain.stage.toUpperCase()} ]`}
              title={missingPhrase ? `We need the ${missingPhrase}.` : 'We need more information.'}
              remedy={keptText ? `${abstain.remedy} ${keptText}` : abstain.remedy}
              {...(isDimensionAbstain(abstain)
                ? { actionLabel: missingPhrase ? `Enter ${missingPhrase}` : 'Check the sizes', onAction: () => envRefs.current[firstMissingKey ?? 'x']?.focus() }
                : { actionLabel: 'Back to capture', onAction: toCapture })}
            />
          )}
          {!expired && !abstain && nCheck > 0 && (
            <div style={{ borderRadius: 14, border: `1.5px dashed ${CHK}`, background: 'color-mix(in oklch, var(--check) 10%, var(--surface))', padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 14 }}>
              <span style={{ width: 32, height: 32, flex: 'none', borderRadius: '50%', border: `1.5px dashed ${CHK}`, color: CHK, display: 'grid', placeItems: 'center', fontWeight: 700, boxSizing: 'border-box' }}>!</span>
              <div>
                <div style={{ fontSize: 18, fontWeight: 600 }}>Ready — <span style={{ fontFamily: MONO }}>{nCheck}</span> {nCheck === 1 ? 'value' : 'values'} to check</div>
                <div style={{ fontSize: 14, color: 'var(--muted)', marginTop: 2 }}>Your written sizes are in. Confirm or fix the ones marked "check".</div>
              </div>
            </div>
          )}
          {!expired && !abstain && nCheck === 0 && (
            <div style={{ borderRadius: 14, border: `1.5px solid ${TRU}`, background: 'color-mix(in oklch, var(--trusted) 10%, var(--surface))', padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 14 }}>
              <span style={{ width: 32, height: 32, flex: 'none', borderRadius: '50%', background: TRU, color: 'var(--raised)', display: 'grid', placeItems: 'center', fontWeight: 700 }}>✓</span>
              <div>
                <div style={{ fontSize: 18, fontWeight: 600 }}>Ready to build</div>
                <div style={{ fontSize: 14, color: 'var(--muted)', marginTop: 2 }}>Every number is written, measured or checked by you.</div>
              </div>
            </div>
          )}

          <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Overall size</span>
              <button type="button" disabled title="Only sizes read from your sketch or a reference are suggested" style={{ border: 'none', background: 'none', padding: 0, font: 'inherit', fontSize: 13, color: 'var(--muted)', cursor: 'default' }}>Use suggested sizes</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 10 }}>
              {envRows.map((e) => {
                const b = BADGE[e.prov];
                const isOpen = open === e.key && !e.required;
                return (
                  <div key={e.key} style={{
                    borderRadius: 12, border: `1.5px ${e.required ? 'dashed' : 'solid'} ${e.required ? STOP : 'var(--line)'}`,
                    background: e.required ? 'color-mix(in oklch, var(--stop) 8%, var(--inset))' : 'var(--inset)',
                    padding: '10px 12px 12px', display: 'flex', flexDirection: 'column', gap: 8,
                    boxShadow: e.required ? '0 0 0 4px color-mix(in oklch, var(--stop) 18%, transparent)' : 'none',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: 13, color: 'var(--muted)' }}>
                      <span>{e.label} <span style={{ fontFamily: MONO }}>· {e.axis}</span></span>
                      <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.06em', color: STOP }}>{e.required ? 'REQUIRED' : ''}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                      <input
                        ref={(el) => { envRefs.current[e.key] = el; }}
                        value={displayValue(e.path, e.value)}
                        onChange={onFieldChange(e.path)}
                        placeholder={e.placeholder} inputMode="decimal" aria-label={`${e.label} in millimetres`}
                        style={{ width: 84, minWidth: 0, border: 'none', outline: 'none', background: 'transparent', color: 'var(--ink)', fontFamily: MONO, fontSize: 28, fontWeight: 500, fontVariantNumeric: 'tabular-nums', padding: 0 }}
                      />
                      <span style={{ fontFamily: MONO, fontSize: 15, fontWeight: 300, color: 'var(--muted)' }}>mm</span>
                    </div>
                    <button type="button" onClick={() => setOpen((o) => (o === e.key ? null : e.key))} style={{
                      alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: 6, height: 26, padding: '0 10px 0 8px',
                      borderRadius: 13, border: `1.5px ${b.line} ${b.fg}`, background: b.bg, color: b.fg, font: 'inherit', fontSize: 12, fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap',
                    }}><span aria-hidden="true">{b.icon}</span>{b.label}</button>
                    {isOpen && (
                      <div style={{ fontSize: 13, lineHeight: 1.4, borderTop: '1px dashed var(--line)', paddingTop: 8, textWrap: 'pretty' } as CSSProperties}>
                        {e.prov === 'user_edited' && e.key !== 'z' ? 'You typed this value.' : EXPLAIN[e.prov]}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 4, overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Every other number</span>
              <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.08em', color: 'var(--muted)' }}>[ FEATURES ]</span>
            </div>
            {abstain ? (
              <div style={{ flex: 1, display: 'grid', placeItems: 'center', textAlign: 'center', fontSize: 14, color: 'var(--muted)', border: '1.5px dashed var(--line)', borderRadius: 12, padding: 24 }}>
                Holes and positions are listed once we have the {missingPhrase ?? 'missing value'}.
              </div>
            ) : groups.length === 0 ? (
              <div style={{ flex: 1, display: 'grid', placeItems: 'center', textAlign: 'center', fontSize: 14, color: 'var(--muted)' }}>No holes, slots, pockets or pins.</div>
            ) : (
              groups.map((g) => {
                const b = BADGE[g.prov];
                const isOpen = open === g.group;
                return (
                  <div key={g.group} style={{ borderRadius: 10, padding: 10, background: isOpen ? 'var(--raised)' : 'transparent' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '120px minmax(0,1fr) auto', alignItems: 'center', gap: 10 }}>
                      <div>
                        <div style={{ fontSize: 15, fontWeight: 500 }}>{g.name}</div>
                        <div style={{ fontSize: 13, color: 'var(--muted)' }}>{g.face} face</div>
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        {g.fields.map((f) => (
                          <label key={f.path} style={{ display: 'flex', alignItems: 'center', gap: 5, height: 40, boxSizing: 'border-box', padding: '0 10px', borderRadius: 9, border: `1.5px ${b.line} ${isCheck(g.prov) ? b.fg : 'var(--line)'}`, background: 'var(--inset)' }}>
                            <span style={{ fontFamily: MONO, fontSize: 13, color: 'var(--muted)' }}>{f.label}</span>
                            <input value={displayValue(f.path, f.value)} onChange={onFieldChange(f.path)} inputMode="decimal" aria-label={f.aria} style={{ width: 40, border: 'none', outline: 'none', background: 'transparent', color: 'var(--ink)', fontFamily: MONO, fontSize: 17, fontWeight: 500, fontVariantNumeric: 'tabular-nums', padding: 0 }} />
                          </label>
                        ))}
                        {g.snapped && <span title="Snapped to a standard size" style={{ fontFamily: MONO, fontSize: 11, color: CHK, border: `1px dashed ${CHK}`, borderRadius: 4, padding: '2px 5px', whiteSpace: 'nowrap' }}>{g.snapLabel}</span>}
                      </div>
                      <button type="button" onClick={() => setOpen((o) => (o === g.group ? null : g.group))} style={{ justifySelf: 'end', display: 'inline-flex', alignItems: 'center', gap: 6, height: 28, padding: '0 11px 0 9px', borderRadius: 14, border: `1.5px ${b.line} ${b.fg}`, background: b.bg, color: b.fg, font: 'inherit', fontSize: 12, fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap' }}><span aria-hidden="true">{b.icon}</span>{b.label}</button>
                    </div>
                    {isOpen && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10, padding: '10px 12px', borderRadius: 9, border: '1px dashed var(--line)', fontSize: 13, lineHeight: 1.4 }}>
                        <span style={{ flex: 1, textWrap: 'pretty' } as CSSProperties}>{g.prov === 'user_edited' ? 'You checked or changed this value.' : EXPLAIN[g.prov]}</span>
                        {g.removable && (
                          <button type="button" onClick={() => dispatch({ type: 'TYPE_VALUE', path: `features[${g.index}].keep`, value: 0 })} style={{ height: 36, padding: '0 12px', borderRadius: 9, border: `1.5px solid ${CHK}`, background: 'transparent', color: CHK, font: 'inherit', fontSize: 13, fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap' }}>Remove {g.feature.toLowerCase()}</button>
                        )}
                        {isCheck(g.prov) && (
                          <button type="button" onClick={() => confirmGroup(g)} style={{ height: 36, padding: '0 12px', borderRadius: 9, border: `1.5px solid ${TRU}`, background: 'transparent', color: TRU, font: 'inherit', fontSize: 13, fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap' }}>✓ Looks right</button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </section>

        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <div style={{ position: 'relative', display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0,1fr))', gap: 14 }}>
            {spec ? (
              <>
                <FaceCard label="FRONT" outline={spec.views.front} width={effEnv('x')} height={effEnv('y')} filledBy={filledByOf('front')} rejected={rejected.includes('front')} onToggleReject={() => dispatch({ type: 'TOGGLE_REJECT', face: 'front' })} />
                <FaceCard label="TOP" outline={spec.views.top} width={effEnv('x')} height={effEnv('z')} filledBy={filledByOf('top')} rejected={rejected.includes('top')} onToggleReject={() => dispatch({ type: 'TOGGLE_REJECT', face: 'top' })} />
                <FaceCard label="RIGHT" outline={spec.views.right} width={effEnv('z')} height={effEnv('y')} filledBy={filledByOf('right')} rejected={rejected.includes('right')} onToggleReject={() => dispatch({ type: 'TOGGLE_REJECT', face: 'right' })} />
              </>
            ) : (
              <div style={{ gridColumn: '1 / -1', borderRadius: 14, border: '1.5px dashed var(--line)', background: 'var(--surface)', padding: 24, display: 'grid', placeItems: 'center', textAlign: 'center', fontSize: 15, fontWeight: 500, minHeight: 130 }}>
                Faces are traced — we&rsquo;ll fuse them once {missingPhrase ? `the ${missingPhrase}` : 'the missing value'} {missingEnvKeys.length > 1 ? 'are' : 'is'} in.
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: 14, flex: 1, minHeight: 0 }}>
            <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0, overflow: 'auto' }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Warnings</span>
              {spec && spec.warnings.length > 0 ? (
                <>
                  <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.1em', color: CHK }}>[ CHECK ]</span>
                  {spec.warnings.map((w, i) => {
                    const meta = warningMeta(w, spec, typed);
                    return (
                      <div key={i} style={{ display: 'grid', gridTemplateColumns: '20px minmax(0,1fr)', gap: 8, fontSize: 14, lineHeight: 1.4, textWrap: 'pretty' } as CSSProperties}>
                        <span style={{ width: 18, height: 18, marginTop: 1, borderRadius: '50%', border: `1.5px ${meta.line} ${meta.color}`, color: meta.color, display: 'grid', placeItems: 'center', fontSize: 11, fontWeight: 700, boxSizing: 'border-box' }}>{meta.icon}</span>
                        <span>{w}</span>
                      </div>
                    );
                  })}
                </>
              ) : (
                <span style={{ fontSize: 14, color: 'var(--muted)' }}>No warnings.</span>
              )}
              <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.1em', color: 'var(--muted)', marginTop: 4 }}>[ INFO ]</span>
              <div style={{ display: 'grid', gridTemplateColumns: '20px minmax(0,1fr)', gap: 8, fontSize: 14, lineHeight: 1.4 }}>
                <span style={{ width: 18, height: 18, marginTop: 1, borderRadius: '50%', border: '1.5px solid var(--muted)', color: 'var(--muted)', display: 'grid', placeItems: 'center', fontSize: 11, boxSizing: 'border-box', fontFamily: MONO }}>i</span>
                <span>{state.reference ? `Sizes checked against your reference: ${state.reference}.` : 'No coin or card in the photos — sizes come from your writing.'}</span>
              </div>
            </div>

            <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 10, minHeight: 0, overflow: 'auto' }}>
              <button type="button" onClick={() => setHwOpen((o) => !o)} aria-expanded={hwOpen} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: 'none', background: 'none', padding: 0, color: 'inherit', font: 'inherit', cursor: 'pointer', textAlign: 'left' }}>
                <span style={{ fontSize: 15, fontWeight: 600 }}>Handwriting read <span style={{ fontWeight: 500, fontSize: 13, color: AI }}>· {readerName}</span></span>
                <span style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--line)', display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 14 }}>{hwOpen ? '−' : '+'}</span>
              </button>
              {hwOpen && (
                <>
                  {crops.length > 0
                    ? crops.map((c, i) => <HandwritingCrop key={i} image={c.image} url={c.url} read={c.read} fieldLabel={c.fieldLabel} />)
                    : <span style={{ fontSize: 13, color: 'var(--muted)' }}>No handwriting read yet.</span>}
                  <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.4 }}>The AI only reads what you wrote. The number is yours.</div>
                </>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <button type="button" onClick={toCapture} style={{ marginRight: 'auto', height: 52, padding: '0 20px', display: 'flex', alignItems: 'center', borderRadius: 12, border: 'none', background: 'var(--ink)', color: 'var(--bg)', font: 'inherit', fontSize: 15, fontWeight: 500, cursor: 'pointer' }}>← Back to capture</button>
            {canBuild ? (
              <button type="button" onClick={retry ? runMerge : onBuild} disabled={buildBlocked} style={{ height: 52, padding: '0 26px', display: 'flex', alignItems: 'center', borderRadius: 12, border: 'none', background: 'var(--accent)', color: 'var(--on-accent)', boxShadow: 'var(--shadow)', font: 'inherit', fontSize: 16, fontWeight: 600, cursor: buildBlocked ? 'default' : 'pointer', opacity: buildBlocked ? 0.7 : 1 }}>
                {building ? 'Building…' : retry ? 'Retry the update' : updating ? 'Updating…' : buildLabel}
              </button>
            ) : (
              <span style={{ height: 52, padding: '0 24px', display: 'flex', alignItems: 'center', borderRadius: 12, background: 'var(--inset)', color: 'var(--muted)', fontSize: 15 }}>
                {expired ? 'Analyze again to build' : missingPhrase ? `Enter the ${missingPhrase} to build` : 'Resolve the issue above to build'}
              </span>
            )}
          </div>
          {mergeErr && <div role="alert" style={{ fontSize: 13, color: STOP }}>{mergeErr}</div>}
          {buildErr && <div role="alert" style={{ fontSize: 13, color: STOP }}>{buildErr}</div>}
          {buildAbstain && (
            <StopCard title="We could not build this part" remedy={buildAbstain.remedy}
              {...(isDimensionAbstain(buildAbstain)
                ? { actionLabel: 'Check the sizes', onAction: () => envRefs.current.x?.focus() }
                : { actionLabel: 'Back to capture', onAction: toCapture })} />
          )}
        </section>
      </div>
    </div>
  );
}
