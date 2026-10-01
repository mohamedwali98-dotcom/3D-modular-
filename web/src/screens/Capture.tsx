import { useCallback, useRef, useState, type CSSProperties } from 'react';
import { ApiError, getExamples, startAnalysis } from '../api/client';
import type { AiSettings, Face } from '../api/types';
import { resetDeadline } from '../components/Shell';
import { CoverageCube } from '../components/CoverageCube';
import { StopCard } from '../components/StopCard';
import { MAX_ITEMS, toCaptureItem, useStore, type CaptureItem, type CaptureKind, type CaptureMode, type Projection } from '../state/store';

const MONO = "'Geist Mono', monospace";
const SILK = "'Silkscreen', monospace";

const FACES: Face[] = ['front', 'back', 'left', 'right', 'top', 'bottom'];
const FACE_LABELS: Record<Face, string> = { front: 'Front', back: 'Back', left: 'Left', right: 'Right', top: 'Top', bottom: 'Bottom' };
const KINDS: CaptureKind[] = ['auto', 'sketch', 'photo', 'drawing'];
const MODES: CaptureMode[] = ['photos', 'sheet'];
const MODE_LABELS: Record<CaptureMode, string> = { photos: 'Per-face photos', sheet: 'One sheet (all views)' };
const PROJECTIONS: Projection[] = ['auto', 'first', 'third'];
const PROJECTION_LABELS: Record<Projection, string> = { auto: 'Auto', first: 'First-angle', third: 'Third-angle' };

const panel: CSSProperties = {
  borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px',
  display: 'flex', flexDirection: 'column', gap: 12, boxSizing: 'border-box',
};
const sectionTitle: CSSProperties = { fontSize: 15, fontWeight: 600 };
function Toggle({ label, hint, checked, onChange, disabled }: {
  label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, opacity: disabled ? 0.5 : 1 }}>
      <span style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
        <span style={{ fontSize: 14 }}>{label}</span>
        {hint && <span style={{ fontSize: 12, color: 'var(--muted)' }}>{hint}</span>}
      </span>
      <span style={{ position: 'relative', width: 40, height: 24, flex: 'none', borderRadius: 12, border: '1px solid var(--line)', background: checked ? 'var(--accent)' : 'var(--inset)', transition: 'background 150ms' }}>
        <input
          type="checkbox" checked={checked} disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          aria-label={label}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', margin: 0, opacity: 0, cursor: disabled ? 'not-allowed' : 'pointer' }}
        />
        <span aria-hidden="true" style={{ position: 'absolute', top: 2, left: checked ? 18 : 2, width: 20, height: 20, borderRadius: '50%', background: 'var(--raised)', boxShadow: 'var(--shadow)', transition: 'left 150ms' }} />
      </span>
    </label>
  );
}

function Segmented<T extends string | number>({ options, labels, value, onChange, ariaLabel, silk }: {
  options: readonly T[]; labels: Record<T, string>; value: T; onChange: (v: T) => void; ariaLabel: string; silk?: boolean;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} style={{ display: 'flex', gap: 4, padding: 3, borderRadius: 8, background: 'var(--inset)' }}>
      {options.map((opt) => {
        const active = opt === value;
        return (
          <button
            key={opt} type="button" role="radio" aria-checked={active} onClick={() => onChange(opt)}
            style={{
              flex: 1, height: 26, minWidth: 0, borderRadius: 6, border: 'none',
              background: active ? 'var(--raised)' : 'transparent', boxShadow: active ? 'var(--shadow)' : 'none',
              color: 'var(--ink)', font: 'inherit', fontFamily: silk ? SILK : MONO,
              fontSize: silk ? 10 : 12, letterSpacing: silk ? '0.03em' : 'normal', textTransform: silk ? 'uppercase' : 'none',
              cursor: 'pointer', whiteSpace: 'nowrap',
            }}
          >
            {labels[opt]}
          </button>
        );
      })}
    </div>
  );
}

function Card({ item, onPatch, onRemove, tags = true }: {
  item: CaptureItem; onPatch: (patch: Partial<Omit<CaptureItem, 'id'>>) => void; onRemove: () => void; tags?: boolean;
}) {
  const kindLabels: Record<CaptureKind, string> = { auto: 'Auto', sketch: 'Sketch', photo: 'Photo', drawing: 'Drawing' };
  return (
    <div style={{ borderRadius: 12, background: 'var(--surface)', boxShadow: 'var(--shadow)', border: '1px solid var(--line)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div style={{ position: 'relative', aspectRatio: '4 / 3', background: '#f4f3f0' }}>
        <img src={item.url} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        <button
          type="button" onClick={onRemove} aria-label="Remove image"
          style={{ position: 'absolute', top: 6, right: 6, width: 26, height: 26, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--raised)', color: 'var(--ink)', fontFamily: MONO, fontSize: 14, lineHeight: 1, cursor: 'pointer', display: 'grid', placeItems: 'center', boxShadow: 'var(--shadow)' }}
        >
          ×
        </button>
      </div>
      {tags && (
        <div style={{ padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, color: 'var(--muted)' }}>
            Face
            <select
              value={item.face} onChange={(e) => onPatch({ face: e.target.value as Face | 'auto' })}
              style={{ height: 32, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--inset)', color: 'var(--ink)', font: 'inherit', fontSize: 13, padding: '0 8px' }}
            >
              <option value="auto">Auto</option>
              {FACES.map((f) => <option key={f} value={f}>{FACE_LABELS[f]}</option>)}
            </select>
          </label>
          <Segmented options={KINDS} labels={kindLabels} value={item.kind} onChange={(kind) => onPatch({ kind })} ariaLabel="Image kind" silk />
        </div>
      )}
    </div>
  );
}

/** The app's first screen: drop sketches/photos, tag each with its face, then Analyze. */
export function Capture() {
  const { state, dispatch } = useStore();
  const [dragOver, setDragOver] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [exampleBusy, setExampleBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const sheet = state.mode === 'sheet';
  const maxItems = sheet ? 1 : MAX_ITEMS;
  const full = state.items.length >= maxItems;
  const room = maxItems - state.items.length;

  const addFiles = useCallback((files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) => f.type.startsWith('image/'));
    if (!arr.length) return;
    if (sheet) {
      // One sheet at a time: a new drop replaces whatever was there.
      for (const it of state.items) dispatch({ type: 'REMOVE_ITEM', id: it.id });
      dispatch({ type: 'ADD_FILES', items: arr.slice(0, 1).map((f) => toCaptureItem(f)) });
      return;
    }
    if (room <= 0) return;
    dispatch({ type: 'ADD_FILES', items: arr.slice(0, room).map((f) => toCaptureItem(f)) });
  }, [dispatch, room, sheet, state.items]);

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files);
    e.target.value = '';
  };

  const setAi = (patch: Partial<AiSettings>) => dispatch({ type: 'SET_AI', patch });

  const tryExample = async () => {
    setExampleBusy(true);
    setError(null);
    try {
      const examples = await getExamples();
      const picked = examples.slice(0, Math.max(room, 0));
      const items: CaptureItem[] = [];
      for (const ex of picked) {
        const res = await fetch(ex.url);
        if (!res.ok) continue;
        const blob = await res.blob();
        const name = ex.url.split('/').pop() || ex.name;
        const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
        items.push(toCaptureItem(file, ex.face, ex.kind));
      }
      if (items.length) dispatch({ type: 'ADD_FILES', items });
      else if (!picked.length) setError('No examples are available right now.');
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not load the examples.');
    } finally {
      setExampleBusy(false);
    }
  };

  const analyze = async () => {
    if (!state.items.length || busy) return;
    setBusy(true);
    setError(null);
    try {
      const files = state.items.map((i) => i.file);
      const faces = state.items.map((i) => i.face);
      const kinds = state.items.map((i) => i.kind);
      const jobId = await startAnalysis(files, faces, kinds, state.reference, state.ai, state.mode, state.projection);
      resetDeadline();
      dispatch({ type: 'START_JOB', jobId });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong. Try again.');
    } finally {
      setBusy(false);
    }
  };

  const canAnalyze = state.items.length > 0 && !busy;
  const locked = !sheet && full; // a sheet always accepts a new drop, replacing the one it has
  const dropTitle = sheet ? 'Drop your sketch sheet, or click to choose' : 'Drop sketches or photos, or click to choose';

  return (
    <section data-screen="Capture" style={{ padding: 28, display: 'flex', flexDirection: 'column', gap: 20, flex: 1, minHeight: 0, boxSizing: 'border-box' }}>
      <header style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'flex-end', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 13, letterSpacing: '0.08em', color: 'var(--muted)' }}>[1/3] CAPTURE</span>
          <h1 style={{ margin: 0, fontFamily: SILK, fontWeight: 400, fontSize: 40, lineHeight: 1, letterSpacing: '0.01em' }}>Capture</h1>
        </div>
        <p style={{ margin: '0 0 4px', maxWidth: 420, fontSize: 15, lineHeight: 1.4, color: 'var(--muted)' }}>
          {sheet
            ? 'Drop one photo of your sheet with every view drawn on it. We find the views, the lines and the numbers you wrote.'
            : 'Drop a sketch or photo of each face. Tag what it shows — the AI only fills in what is missing.'}
        </p>
        <button
          type="button" onClick={() => dispatch({ type: 'GOTO', screen: 'describe' })}
          style={{ marginLeft: 'auto', marginBottom: 4, height: 40, padding: '0 16px', borderRadius: 10, border: '1.5px dashed var(--ai)', background: 'color-mix(in oklch, var(--ai) 8%, var(--surface))', color: 'var(--ink)', font: 'inherit', fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap', cursor: 'pointer' }}
        >
          Describe it instead →
        </button>
      </header>

      <div style={{ flex: 'none' }}>
        <Segmented options={MODES} labels={MODE_LABELS} value={state.mode} onChange={(mode) => dispatch({ type: 'SET_MODE', mode })} ariaLabel="Capture mode" />
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(0,3fr) minmax(0,2fr)', gap: 24 }}>
        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <div
            onDragOver={(e) => { e.preventDefault(); if (!locked) setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            style={{
              position: 'relative', minHeight: 130, borderRadius: 14, boxSizing: 'border-box',
              border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--line)'}`,
              background: dragOver ? 'color-mix(in oklch, var(--accent) 8%, var(--surface))' : 'var(--surface)',
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6,
              padding: 20, textAlign: 'center', transition: 'border-color 150ms, background 150ms', flex: 'none',
            }}
          >
            <span style={{ fontSize: 15, fontWeight: 500 }}>{dropTitle}</span>
            <span style={{ fontSize: 13, color: 'var(--muted)' }}>
              {sheet
                ? state.items.length ? 'One sheet — drop another to replace it' : 'One image, with every view drawn on it'
                : full ? 'Maximum 6 images — remove one to add another' : `${state.items.length} of ${MAX_ITEMS} images`}
            </span>
            <input
              ref={inputRef} type="file" accept="image/*" multiple={!sheet} disabled={locked} onChange={onPick}
              aria-label={sheet ? 'Add your sketch sheet' : 'Add sketches or photos'} title={dropTitle}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0, cursor: locked ? 'not-allowed' : 'pointer' }}
            />
          </div>

          <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            {state.items.length === 0 ? (
              <div style={{ height: '100%', display: 'grid', placeItems: 'center', color: 'var(--muted)', fontSize: 14, textAlign: 'center', padding: 24 }}>
                {sheet ? 'No sheet yet. Add a photo of it above.' : 'No images yet. Add your sketches or photos above.'}
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 14 }}>
                {state.items.map((item) => (
                  <Card
                    key={item.id} item={item} tags={!sheet}
                    onPatch={(patch) => dispatch({ type: 'SET_ITEM', id: item.id, patch })}
                    onRemove={() => dispatch({ type: 'REMOVE_ITEM', id: item.id })}
                  />
                ))}
              </div>
            )}
          </div>
        </section>

        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0, overflow: 'auto' }}>
          {sheet ? (
            <div style={panel}>
              <span style={sectionTitle}>One sheet, several views</span>
              <span style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.4 }}>
                We find the views drawn on your sheet, classify the lines and read the numbers you wrote — then the
                same review and export screens a per-face photo uses.
              </span>
              <span style={sectionTitle}>Projection</span>
              <Segmented options={PROJECTIONS} labels={PROJECTION_LABELS} value={state.projection}
                onChange={(projection) => dispatch({ type: 'SET_PROJECTION', projection })} ariaLabel="Projection" />
              <span style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.4 }}>
                Auto reads it from how the views agree (ISO first-angle when they agree both ways). First-angle puts the top view below the front, the left view on its right; third-angle (US) puts the top view above. A projection symbol or the
                view labels on your sheet take precedence.
              </span>
            </div>
          ) : (
            <div style={panel}>
              <CoverageCube items={state.items} />
            </div>
          )}


          <div style={panel}>
            <button
              type="button" onClick={() => setAiOpen((v) => !v)} aria-expanded={aiOpen}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: 'none', background: 'none', padding: 0, color: 'inherit', font: 'inherit', cursor: 'pointer', textAlign: 'left' }}
            >
              <span style={sectionTitle}>Reading &amp; AI</span>
              <span style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--line)', display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 14 }}>{aiOpen ? '−' : '+'}</span>
            </button>
            {aiOpen && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <Toggle label="Read handwriting with the vision model" checked={state.ai.use_reader} onChange={(v) => setAi({ use_reader: v })} />
                <Toggle label="Draw missing faces with Qwen-Image" checked={state.ai.use_qwen_image} onChange={(v) => setAi({ use_qwen_image: v })} />
                <Toggle label="Rescue sketches with an open outline" checked={state.ai.use_rescue} onChange={(v) => setAi({ use_rescue: v })} />
                <Toggle label="TripoSR fallback for missing faces" checked={state.ai.use_triposr} onChange={(v) => setAi({ use_triposr: v })} />
                <Toggle label="Hole depth from photos (Solaria)" hint="Photos only; adds 60–180 s" checked={state.ai.use_solaria} onChange={(v) => setAi({ use_solaria: v })} />
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, color: 'var(--muted)', flex: 1 }}>
                    Seed
                    <input
                      type="number" min={0} value={state.ai.seed} disabled={state.ai.randomize_seed}
                      onChange={(e) => setAi({ seed: Number(e.target.value) || 0 })}
                      style={{ height: 36, borderRadius: 8, border: '1px solid var(--line)', background: 'var(--inset)', color: 'var(--ink)', fontFamily: MONO, fontSize: 15, padding: '0 10px', width: '100%', boxSizing: 'border-box', opacity: state.ai.randomize_seed ? 0.5 : 1 }}
                    />
                  </label>
                  <div style={{ paddingTop: 18 }}>
                    <Toggle label="Randomize" checked={state.ai.randomize_seed} onChange={(v) => setAi({ randomize_seed: v })} />
                  </div>
                </div>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: 'var(--muted)' }}>
                  Attempts per face
                  <Segmented
                    options={[1, 2, 3, 4] as const} labels={{ 1: '1', 2: '2', 3: '3', 4: '4' }}
                    value={state.ai.attempts as 1 | 2 | 3 | 4} onChange={(n) => setAi({ attempts: n })}
                    ariaLabel="Attempts per face"
                  />
                </label>
              </div>
            )}
          </div>

          {!sheet && (
            <div style={panel}>
              <button
                type="button" onClick={tryExample} disabled={exampleBusy || full}
                style={{ height: 44, padding: '0 20px', borderRadius: 10, border: '1px solid var(--line)', background: 'var(--raised)', color: 'var(--ink)', font: 'inherit', fontSize: 15, cursor: exampleBusy || full ? 'not-allowed' : 'pointer', boxShadow: 'var(--shadow)', opacity: exampleBusy || full ? 0.6 : 1 }}
              >
                {exampleBusy ? 'Loading example…' : 'Try an example'}
              </button>
            </div>
          )}

          {error && <StopCard title="Could not start analysis" remedy={error} actionLabel="Try again" onAction={analyze} />}

          <div style={{ marginTop: 'auto' }}>
            <button
              type="button" onClick={analyze} disabled={!canAnalyze}
              style={{
                width: '100%', height: 52, borderRadius: 12, border: 'none',
                background: canAnalyze ? 'var(--accent)' : 'var(--inset)', color: canAnalyze ? 'var(--on-accent)' : 'var(--muted)',
                boxShadow: canAnalyze ? 'var(--shadow)' : 'none', fontSize: 16, fontWeight: 600,
                cursor: canAnalyze ? 'pointer' : 'not-allowed',
              }}
            >
              {busy ? 'Starting…' : 'Analyze →'}
            </button>
          </div>
        </section>
      </div>
    </section>
  );
}
