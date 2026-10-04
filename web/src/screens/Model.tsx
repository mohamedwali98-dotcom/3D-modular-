import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import { ApiError, buildModel, exportFiles, type ExportSettings } from '../api/client';
import type { ExportResult, GeometrySettings, PrintSettings } from '../api/types';
import { MatchRing } from '../components/MatchRing';
import { StopCard } from '../components/StopCard';
import { Viewer, type DimLabel } from '../components/Viewer';
import { EXPIRED } from '../lib/abstain';
import { layerMax, withNozzle } from '../lib/print';
import { BADGE, envelopeRows, isCheck } from '../lib/provenance';
import { useStore } from '../state/store';

// From Model v2.dc.html
const FORMATS: [string, string, string][] = [
  ['stl', 'STL', 'print'], ['step', 'STEP', 'CAD'], ['3mf', '3MF', 'print, colour'], ['gcode', 'G-code', 'ready to print'],
  ['obj', 'OBJ', 'any 3D app'], ['glb', 'GLB', 'web / AR'], ['ply', 'PLY', 'mesh'], ['brep', 'BREP', 'exact solid'],
  ['blend', 'Blender', '.blend scene'], ['dxf', 'DXF', '2D drawing'], ['svg', 'SVG', '2D drawing'], ['pdf', 'PDF', 'dimensioned'],
];
const NOTES: Record<string, string> = { stl: 'for slicers', step: 'open in any CAD tool', '3mf': 'print, colour', pdf: 'dimensioned drawing' };
const EDGES: [string, GeometrySettings['finish_edges']][] = [
  ['outline corners', 'all_vertical'], ['front face', 'top'], ['back face', 'bottom'], ['all', 'all'],
];
const SUPPORTS: [string, PrintSettings['supports']][] = [['off', 'off'], ['build plate', 'buildplate'], ['everywhere', 'everywhere']];
const MONO = "'Geist Mono', monospace";

type Mesh = 'draft' | 'normal' | 'fine';
interface Print { material: PrintSettings['material']; nozzle: string; layer: number; infill: number; pattern: PrintSettings['infill_pattern']; perims: number; supports: PrintSettings['supports']; brim: number; scale: number }
const PRINT0: Print = { material: 'PLA', nozzle: '0.4', layer: 0.2, infill: 20, pattern: 'gyroid', perims: 2, supports: 'off', brim: 0, scale: 100 };

const fmtMm = (v: number) => String(+v.toFixed(1));
const fmtSize = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1000))} KB`);
const baseName = (u: string) => decodeURIComponent(u.split('?')[0].split('/').pop() || 'part_bundle.zip');
const errText = (e: unknown) => (e instanceof ApiError ? e.message : 'Something went wrong. Try again.');

// Segmented control and pill styles from the design's seg()
const segBtn = (on: boolean, extra?: CSSProperties): CSSProperties => ({
  flex: 1, height: 34, borderRadius: 8, border: 'none', background: on ? 'var(--raised)' : 'transparent', boxShadow: on ? 'var(--shadow)' : 'none',
  color: on ? 'var(--ink)' : 'var(--muted)', font: 'inherit', fontSize: 14, cursor: 'pointer', ...extra,
});
const pillBtn = (on: boolean): CSSProperties => ({
  height: 30, padding: '0 10px', borderRadius: 15, border: `1.5px solid ${on ? 'var(--accent)' : 'var(--line)'}`,
  background: on ? 'color-mix(in oklch, var(--accent) 14%, transparent)' : 'transparent', color: on ? 'var(--ink)' : 'var(--muted)',
  font: 'inherit', fontSize: 13, whiteSpace: 'nowrap', cursor: 'pointer',
});
const segWrap = (gap = 4): CSSProperties => ({ display: 'flex', gap, padding: 4, borderRadius: 11, background: 'var(--inset)' });
const label: CSSProperties = { fontSize: 14, color: 'var(--muted)' };
const chip: CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 6, minHeight: 28, padding: '4px 10px', borderRadius: 14, boxSizing: 'border-box',
  border: '1.5px dashed var(--check)', background: 'var(--raised)', color: 'var(--check)', fontSize: 12, fontWeight: 500, lineHeight: 1.3,
};

const CSS = `
@media (max-width:1023px){
  .s2c-model-grid{grid-template-columns:minmax(0,1fr)!important}
  .s2c-model-viewer{min-height:460px}
  .s2c-model-header{height:auto!important;flex-wrap:wrap}
}`;

export function Model() {
  const { state, dispatch } = useStore();
  const spec = state.analysis?.spec ?? null;
  const model = state.model;
  const g = state.geometry;

  const [dims, setDims] = useState(true);
  const [section, setSection] = useState(false);
  const [fit, setFit] = useState(0);
  const [rebuilt, setRebuilt] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildErr, setBuildErr] = useState<string | null>(null);
  const [expired, setExpired] = useState(false);

  const [sel, setSel] = useState<Record<string, boolean>>({ stl: true, step: true, '3mf': true, pdf: true, gcode: true });
  const [mesh, setMesh] = useState<Mesh>('normal');
  const [print, setPrint] = useState<Print>(PRINT0);
  const [drawer, setDrawer] = useState(false);
  const [dp, setDp] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportErr, setExportErr] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);
  const [showResult, setShowResult] = useState(false);

  // Debounced rebuild: 300 ms, last one wins, stale responses are ignored.
  const seq = useRef(0);
  const timer = useRef<number | undefined>(undefined);
  const rbTimer = useRef<number | undefined>(undefined);
  const exportSeq = useRef(0); // bumped whenever the geometry or spec changes: an older export is dropped
  const live = useRef({ requestId: state.analysis?.request_id || null, spec });
  live.current = { requestId: state.analysis?.request_id || null, spec };

  const runBuild = useCallback(async (geometry: GeometrySettings, flash: boolean) => {
    const s = live.current.spec;
    if (!s) return;
    const my = ++seq.current;
    setBuilding(true); setBuildErr(null);
    try {
      const m = await buildModel({ request_id: live.current.requestId, spec: s, geometry });
      if (my !== seq.current) return;
      dispatch({ type: 'MODEL', model: m, spec: s });
      if (flash) {
        setRebuilt(true);
        window.clearTimeout(rbTimer.current);
        rbTimer.current = window.setTimeout(() => setRebuilt(false), 900);
      }
    } catch (e) {
      if (my !== seq.current) return;
      if (e instanceof ApiError && e.status === 404) setExpired(true);
      else setBuildErr(errText(e));
    } finally {
      if (my === seq.current) setBuilding(false);
    }
  }, [dispatch]);

  const setShape = (patch: Partial<GeometrySettings>) => {
    dispatch({ type: 'SET_GEOMETRY', patch });
    const geometry = { ...g, ...patch };
    setResult(null); setShowResult(false); // exported files no longer match the shape
    exportSeq.current += 1; setExporting(false);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => { void runBuild(geometry, true); }, 300);
  };

  // Build when there is no model yet, or when the spec changed after the model was built (a late merge).
  useEffect(() => {
    if (spec && (!state.model || state.modelSpec !== spec)) {
      exportSeq.current += 1;
      setResult(null); setShowResult(false); setExporting(false);
      void runBuild(state.geometry, !!state.model);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec]);

  useEffect(() => () => {
    seq.current += 1;
    exportSeq.current += 1;
    window.clearTimeout(timer.current);
    window.clearTimeout(rbTimer.current);
  }, []);

  const goReview = () => dispatch({ type: 'GOTO', screen: 'review' });
  const goCapture = () => dispatch({ type: 'GOTO', screen: 'capture' });

  if (!spec) {
    return (
      <div style={{ padding: 28 }}>
        <StopCard title="No part to show yet" remedy="Analyze your sketches, then review the numbers to build the part." actionLabel="Go to Capture" onAction={() => dispatch({ type: 'GOTO', screen: 'capture' })} />
      </div>
    );
  }

  // Stats
  const env = envelopeRows(spec);
  const provs = Array.from(new Set(env.map((r) => r.prov)));
  const worst = env.find((r) => isCheck(r.prov));
  const sizeSrc = worst ? `${BADGE[worst.prov].icon} ${BADGE[worst.prov].label}`
    : provs.length === 1 ? `${BADGE[provs[0]].icon} ${BADGE[provs[0]].label}`
      : `✓ ${provs.map((p) => BADGE[p].label.toLowerCase()).join(' & ')}`;
  const bbox = model?.bbox_mm ?? null;
  const labels: [DimLabel, DimLabel, DimLabel] | null = bbox
    ? [{ value: fmtMm(bbox[0]), prov: env[0].prov }, { value: fmtMm(bbox[1]), prov: env[1].prov }, { value: fmtMm(bbox[2]), prov: env[2].prov }]
    : null;
  const ptime = result?.print_time_s ?? null, fil = result?.filament_g ?? null;
  const stats = [
    { label: 'Size', val: bbox ? bbox.map(fmtMm).join(' × ') : '—', unit: bbox ? ' mm' : '', src: sizeSrc },
    { label: 'Volume', val: model?.volume_cm3 != null ? model.volume_cm3.toFixed(2) : '—', unit: model?.volume_cm3 != null ? ' cm³' : '', src: 'computed from the solid' },
    { label: 'Print time', val: ptime != null ? String(Math.round(ptime / 60)) : '—', unit: ptime != null ? ' min' : '', src: ptime != null ? 'from the slicer' : 'after export' },
    { label: 'Filament', val: fil != null ? fil.toFixed(1) : '—', unit: fil != null ? ' g' : '', src: fil != null ? 'from the slicer · ' + print.material : 'after export' },
  ];

  const tools = [
    { label: 'Auto-fit', on: false, onClick: () => setFit((n) => n + 1) },
    { label: 'Dimensions', on: dims, onClick: () => setDims(!dims) },
    { label: 'Section cut', on: section, onClick: () => setSection(!section) },
  ];
  const viewerTag = rebuilt ? 'REBUILT ✓' : 'LIVE 3D · part.step';
  const rebuildTag = rebuilt ? 'REBUILT ✓' : building ? '[ BUILDING… ]' : '[ LIVE ]';
  const finishOff = g.finish === 'none';
  const nSel = FORMATS.filter(([k]) => sel[k]).length;

  const openDrawer = () => { setDrawer(true); setDp(false); requestAnimationFrame(() => requestAnimationFrame(() => setDp(true))); };
  const setP = (patch: Partial<Print>) => setPrint((p) => ({ ...p, ...patch }));

  const doExport = async () => {
    if (!nSel || exporting || expired) return;
    const my = ++exportSeq.current;
    setExporting(true); setExportErr(null);
    const settings: ExportSettings = {
      geometry: g,
      mesh: { quality: mesh },
      printing: {
        material: print.material, nozzle_mm: +print.nozzle, layer_mm: print.layer, infill_pct: print.infill, infill_pattern: print.pattern,
        perimeters: print.perims, supports: print.supports, brim_mm: print.brim, scale_pct: print.scale,
      },
      export: { formats: FORMATS.filter(([k]) => sel[k]).map(([k]) => k) },
    };
    try {
      const r = await exportFiles({ spec, settings });
      if (my !== exportSeq.current) return; // the shape changed while exporting: these files are stale
      setResult(r); setShowResult(!r.abstain);
    } catch (e) {
      if (my !== exportSeq.current) return;
      if (e instanceof ApiError && e.status === 404) setExpired(true);
      else setExportErr(errText(e));
    } finally {
      if (my === exportSeq.current) setExporting(false);
    }
  };

  const files = result ? Object.entries(result.files) : [];
  const gcodeNote = ptime != null || fil != null
    ? [ptime != null ? `${Math.round(ptime / 60)} min` : null, fil != null ? `${fil.toFixed(1)} g` : null].filter(Boolean).join(' · ')
    : 'ready to print';
  const sliders = [
    { label: 'Layer', min: 0.05, max: layerMax(print.nozzle), step: 0.01, value: print.layer, text: print.layer.toFixed(2), unit: ' mm', set: (v: number) => setP({ layer: v }) },
    { label: 'Infill', min: 0, max: 100, step: 5, value: print.infill, text: String(print.infill), unit: ' %', set: (v: number) => setP({ infill: v }) },
    { label: 'Perimeters', min: 1, max: 8, step: 1, value: print.perims, text: String(print.perims), unit: '', set: (v: number) => setP({ perims: v }) },
    { label: 'Brim', min: 0, max: 10, step: 1, value: print.brim, text: String(print.brim), unit: ' mm', set: (v: number) => setP({ brim: v }) },
    { label: 'Scale', min: 50, max: 200, step: 5, value: print.scale, text: String(print.scale), unit: ' %', set: (v: number) => setP({ scale: v }) },
  ];

  const warnings = model?.warnings ?? [];
  const showRing = !!model && !model.abstain && (model.iou_mean != null || Object.keys(model.views).length > 0);

  return (
    <div data-screen="Model" style={{ flex: 1, minWidth: 0, minHeight: 0, boxSizing: 'border-box', padding: 28, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <style>{CSS}</style>
      <header className="s2c-model-header" style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'flex-end', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 13, letterSpacing: '0.08em', color: 'var(--muted)' }}>[3/3] MODEL</span>
          <h1 style={{ margin: 0, fontFamily: "'Silkscreen', monospace", fontWeight: 400, fontSize: 40, lineHeight: 1 }}>Your part</h1>
        </div>
        <p style={{ margin: '0 0 4px', maxWidth: 420, fontSize: 15, lineHeight: 1.4, color: 'var(--muted)', textWrap: 'pretty' } as CSSProperties}>Built from your numbers. Not quite right? Reshape it here, or go back and fix a number.</p>
        <button type="button" onClick={goReview} style={{ marginLeft: 'auto', height: 44, padding: '0 18px', display: 'flex', alignItems: 'center', borderRadius: 10, border: 'none', background: 'var(--ink)', color: 'var(--bg)', font: 'inherit', fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap', cursor: 'pointer' }}>← Fix a number</button>
      </header>

      <div className="s2c-model-grid" style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 404px', gap: 24 }}>
        {/* Viewer */}
        <section className="s2c-model-viewer" style={{ position: 'relative', minHeight: 0, borderRadius: 16, overflow: 'hidden', background: 'var(--surface)', boxShadow: 'var(--shadow)' }}>
          <Viewer glbUrl={model && !model.abstain ? model.glb_url : null} labels={labels} showDims={dims} section={section} fitSignal={fit}
            placeholder={building ? 'Building your part…' : buildErr ? '3D preview unavailable' : model?.abstain ? '' : '3D preview unavailable'} />

          <div style={{ position: 'absolute', left: 18, top: 18, display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 8, maxWidth: 'calc(100% - 420px)', minWidth: 200 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, height: 32, padding: '0 12px', borderRadius: 16, background: 'var(--raised)', boxShadow: 'var(--shadow)', fontFamily: MONO, fontSize: 12, letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent)' }} />{viewerTag}
            </div>
            {warnings.map((w, i) => <span key={i} style={chip}><span aria-hidden="true">!</span>{w}</span>)}
            {buildErr && <span role="alert" style={{ ...chip, border: '1.5px dashed var(--stop)', color: 'var(--stop)' }}>{buildErr}</span>}
          </div>

          <div style={{ position: 'absolute', right: 18, top: 18, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10 }}>
            <div style={segWrap()}>
              {tools.map((b) => (
                <button key={b.label} type="button" onClick={b.onClick} aria-pressed={b.on} style={{ height: 36, padding: '0 12px', borderRadius: 9, border: 'none', background: b.on ? 'var(--raised)' : 'transparent', boxShadow: b.on ? 'var(--shadow)' : 'none', color: 'var(--ink)', font: 'inherit', fontSize: 13, whiteSpace: 'nowrap', cursor: 'pointer' }}>{b.label}</button>
              ))}
            </div>
            {showRing && model && <MatchRing iouMean={model.iou_mean} iou={model.iou} views={model.views} />}
          </div>

          {(expired || model?.abstain) && (
            <div style={{ position: 'absolute', left: 18, right: 18, top: '50%', transform: 'translateY(-50%)' }}>
              {expired
                ? <StopCard kicker="[ EXPIRED ]" title={EXPIRED.title} remedy={EXPIRED.remedy} actionLabel={EXPIRED.action} onAction={goCapture} />
                : <StopCard title="We could not build this part" remedy={model!.abstain!.remedy} actionLabel="← Fix a number" onAction={goReview} />}
            </div>
          )}

          <div style={{ position: 'absolute', left: 18, bottom: 18, right: 18, display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            {stats.map((s) => (
              <div key={s.label} style={{ padding: '10px 14px', borderRadius: 12, background: 'var(--raised)', boxShadow: 'var(--shadow)' }}>
                <div style={{ fontSize: 12, color: 'var(--muted)' }}>{s.label}</div>
                <div style={{ fontFamily: MONO, fontSize: 19, fontWeight: 500, fontVariantNumeric: 'tabular-nums', marginTop: 2, whiteSpace: 'nowrap' }}>{s.val}<span style={{ fontSize: 13, fontWeight: 300, color: 'var(--muted)' }}>{s.unit}</span></div>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2, whiteSpace: 'nowrap' }}>{s.src}</div>
              </div>
            ))}
            <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--muted)', whiteSpace: 'nowrap' }}>Drag to orbit · scroll to zoom</span>
          </div>
        </section>

        {/* Controls */}
        <section style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <div style={{ borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Shape</span>
              <span aria-live="polite" style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.08em', color: 'var(--accent-ink)' }}>{rebuildTag}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '84px minmax(0,1fr)', gap: '10px 12px', alignItems: 'center' }}>
              <span style={label}>Edge finish</span>
              <div style={segWrap()}>
                {(['none', 'fillet', 'chamfer'] as const).map((v) => (
                  <button key={v} type="button" aria-pressed={g.finish === v} onClick={() => setShape({ finish: v })} style={segBtn(g.finish === v)}>{v}</button>
                ))}
              </div>
              <span style={label}>Size</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, opacity: finishOff ? 0.45 : 1 }}>
                <input type="range" min={0.2} max={10} step={0.1} value={g.finish_mm} disabled={finishOff} aria-label="Edge finish size"
                  onChange={(e) => setShape({ finish_mm: +e.target.value })} style={{ flex: 1, accentColor: 'var(--accent)' }} />
                <span style={{ width: 62, textAlign: 'right', fontFamily: MONO, fontSize: 15, fontVariantNumeric: 'tabular-nums' }}>{g.finish_mm.toFixed(1)}<span style={{ fontWeight: 300, fontSize: 12, color: 'var(--muted)' }}> mm</span></span>
              </div>
              <span style={label}>Edges</span>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', opacity: finishOff ? 0.45 : 1 }}>
                {EDGES.map(([l, v]) => (
                  <button key={v} type="button" aria-pressed={g.finish_edges === v} disabled={finishOff} onClick={() => setShape({ finish_edges: v })} style={pillBtn(g.finish_edges === v)}>{l}</button>
                ))}
              </div>
              <span style={label}>Hole fit</span>
              <div style={segWrap()}>
                {(['fine', 'medium', 'coarse'] as const).map((v) => (
                  <button key={v} type="button" aria-pressed={g.clearance === v} onClick={() => setShape({ clearance: v })} style={segBtn(g.clearance === v)}>{v}</button>
                ))}
              </div>
            </div>
            <button type="button" onClick={() => setShape({ snap: !g.snap })} role="switch" aria-checked={g.snap} style={{ display: 'flex', alignItems: 'center', gap: 12, border: 'none', background: 'none', padding: '2px 0 0', color: 'inherit', font: 'inherit', textAlign: 'left', cursor: 'pointer' }}>
              <span style={{ width: 38, height: 22, borderRadius: 11, background: g.snap ? 'var(--accent)' : 'var(--line)', position: 'relative', flex: 'none' }}><span style={{ position: 'absolute', top: 3, left: g.snap ? 19 : 3, width: 16, height: 16, borderRadius: '50%', background: 'var(--raised)' }} /></span>
              <span><span style={{ fontSize: 14 }}>Snap <span style={{ color: 'var(--ai)', fontWeight: 500 }}>AI-suggested</span> values to standard sizes</span><br /><span style={{ fontSize: 12, color: 'var(--muted)' }}>Your numbers are never snapped.</span></span>
            </button>
          </div>

          {!showResult && (
            <div style={{ flex: 1, minHeight: 0, borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 15, fontWeight: 600 }}>Export</span>
                <span style={{ fontSize: 13, color: 'var(--muted)' }}><span style={{ fontFamily: MONO, color: 'var(--ink)' }}>{nSel}</span> selected</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0,1fr))', gap: 6 }}>
                {FORMATS.map(([k, ext, purpose]) => {
                  const on = !!sel[k];
                  return (
                    <button key={k} type="button" aria-pressed={on} onClick={() => setSel((s) => ({ ...s, [k]: !s[k] }))}
                      style={{ position: 'relative', height: 52, padding: '7px 9px', boxSizing: 'border-box', borderRadius: 10, border: `1.5px solid ${on ? 'var(--accent)' : 'var(--line)'}`, background: on ? 'color-mix(in oklch, var(--accent) 12%, transparent)' : 'transparent', color: 'var(--ink)', font: 'inherit', textAlign: 'left', cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                      <span style={{ fontFamily: MONO, fontSize: 13, fontWeight: 500 }}>{ext}</span>
                      <span style={{ fontSize: 11, color: 'var(--muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{purpose}</span>
                      <span style={{ position: 'absolute', right: 6, top: 6, width: 14, height: 14, borderRadius: 4, border: `1.5px solid ${on ? 'var(--accent)' : 'var(--line)'}`, background: on ? 'var(--accent)' : 'transparent', color: 'var(--on-accent)', fontSize: 10, fontWeight: 700, display: 'grid', placeItems: 'center', boxSizing: 'border-box' }}>{on ? '✓' : ''}</span>
                    </button>
                  );
                })}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '84px minmax(0,1fr)', gap: 12, alignItems: 'center' }}>
                <span style={label}>Mesh</span>
                <div style={segWrap()}>
                  {(['draft', 'normal', 'fine'] as const).map((v) => (
                    <button key={v} type="button" aria-pressed={mesh === v} onClick={() => setMesh(v)} style={segBtn(mesh === v, { height: 32 })}>{v}</button>
                  ))}
                </div>
              </div>
              <button type="button" onClick={openDrawer} style={{ display: 'flex', alignItems: 'center', gap: 12, height: 50, flex: 'none', padding: '0 14px', borderRadius: 11, border: '1px solid var(--line)', background: 'var(--inset)', color: 'inherit', font: 'inherit', cursor: 'pointer', textAlign: 'left' }}>
                <span style={{ fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap' }}>3D print</span>
                <span style={{ fontFamily: MONO, fontSize: 12, color: 'var(--muted)', flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{`${print.material} · ${print.nozzle} mm · ${print.layer.toFixed(2)} · ${print.infill}% ${print.pattern}`}</span>
                <span style={{ color: 'var(--accent-ink)' }}>→</span>
              </button>
              {result?.abstain && <StopCard title="Export stopped" remedy={result.abstain.remedy} />}
              {exportErr && <div role="alert" style={{ fontSize: 13, color: 'var(--stop)' }}>{exportErr}</div>}
              <button type="button" onClick={doExport} disabled={!nSel || exporting || !!model?.abstain || expired}
                style={{ marginTop: 'auto', height: 52, flex: 'none', borderRadius: 12, border: 'none', background: 'var(--accent)', color: 'var(--on-accent)', boxShadow: 'var(--shadow)', font: 'inherit', fontSize: 16, fontWeight: 600, cursor: !nSel || exporting ? 'default' : 'pointer', opacity: !nSel || model?.abstain || expired ? 0.5 : 1 }}>
                {exporting ? 'Exporting…' : `Export ${nSel} selected →`}
              </button>
            </div>
          )}

          {showResult && result && (
            <div style={{ flex: 1, minHeight: 0, borderRadius: 14, border: '1.5px solid var(--trusted)', background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: 18, display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ width: 34, height: 34, flex: 'none', borderRadius: '50%', background: 'var(--trusted)', color: 'var(--raised)', display: 'grid', placeItems: 'center', fontWeight: 700 }}>✓</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 17, fontWeight: 600 }}>Your files are ready</div>
                  <div style={{ fontFamily: MONO, fontSize: 13, color: 'var(--muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{result.zip_url ? baseName(result.zip_url) : `${files.length} files`}</div>
                </div>
                {result.zip_url && (
                  <a href={result.zip_url} download style={{ height: 44, padding: '0 16px', display: 'flex', alignItems: 'center', borderRadius: 11, background: 'var(--accent)', color: 'var(--on-accent)', fontSize: 15, fontWeight: 600, textDecoration: 'none', whiteSpace: 'nowrap' }}>Download all (.zip)</a>
                )}
              </div>
              {result.abstain && <StopCard title="Export stopped" remedy={result.abstain.remedy} />}
              {result.warnings.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {result.warnings.map((w, i) => <span key={i} style={chip}><span aria-hidden="true">!</span>{w}</span>)}
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', borderTop: '1px solid var(--line)' }}>
                {files.map(([k, f]) => (
                  <div key={k} style={{ display: 'grid', gridTemplateColumns: '60px minmax(0,1fr) auto 40px', alignItems: 'center', gap: 10, height: 54, borderBottom: '1px solid var(--line)' }}>
                    <span style={{ fontFamily: MONO, fontSize: 12, fontWeight: 500, padding: '4px 0', textAlign: 'center', borderRadius: 6, background: 'var(--inset)' }}>{k.toUpperCase()}</span>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontFamily: MONO, fontSize: 13, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{f.name}</div>
                      <div style={{ fontSize: 12, color: 'var(--muted)' }}>{k === 'gcode' ? gcodeNote : NOTES[k] ?? FORMATS.find(([x]) => x === k)?.[2] ?? ''}</div>
                    </div>
                    <span style={{ fontFamily: MONO, fontSize: 13, color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>{fmtSize(f.size_bytes)}</span>
                    <a href={f.url} download={f.name} aria-label={`Download ${f.name}`} title="Download" style={{ width: 40, height: 40, boxSizing: 'border-box', borderRadius: 10, border: '1px solid var(--line)', background: 'transparent', color: 'var(--ink)', display: 'grid', placeItems: 'center', textDecoration: 'none' }}>↓</a>
                  </div>
                ))}
              </div>
              {(ptime != null || fil != null) && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px', borderRadius: 11, background: 'var(--inset)' }}>
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>G-code</span>
                  {ptime != null && <span style={{ fontFamily: MONO, fontSize: 16, fontWeight: 500 }}>{Math.round(ptime / 60)} min</span>}
                  {ptime != null && fil != null && <span style={{ color: 'var(--muted)' }}>·</span>}
                  {fil != null && <span style={{ fontFamily: MONO, fontSize: 16, fontWeight: 500 }}>{fil.toFixed(1)}<span style={{ fontWeight: 300, fontSize: 12, color: 'var(--muted)' }}> g</span></span>}
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>{print.material}</span>
                </div>
              )}
              <div style={{ marginTop: 'auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 13, color: 'var(--muted)' }}>
                <span>Links expire with your files</span>
                <button type="button" onClick={() => setShowResult(false)} style={{ height: 36, border: 'none', background: 'none', color: 'var(--accent-ink)', font: 'inherit', fontSize: 13, fontWeight: 500, cursor: 'pointer', padding: 0 }}>Change selection</button>
              </div>
            </div>
          )}

          {drawer && (
            <div role="dialog" aria-label="3D print settings" style={{ position: 'absolute', inset: 0, zIndex: 2, borderRadius: 14, background: 'var(--raised)', boxShadow: 'var(--shadow), -24px 0 48px oklch(0 0 0 / 0.25)', padding: 18, display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto', transform: `translateX(${dp ? 0 : 40}px)`, opacity: dp ? 1 : 0, transition: 'transform 280ms cubic-bezier(.2,.8,.2,1), opacity 200ms' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 17, fontWeight: 600 }}>3D print settings</span>
                <button type="button" onClick={() => setDrawer(false)} aria-label="Close" style={{ width: 40, height: 40, borderRadius: 10, border: '1px solid var(--line)', background: 'transparent', color: 'var(--ink)', font: 'inherit', cursor: 'pointer' }}>✕</button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '96px minmax(0,1fr)', gap: '14px 12px', alignItems: 'center' }}>
                <span style={label}>Material</span>
                <div style={segWrap(3)}>
                  {(['PLA', 'PETG', 'ABS', 'ASA', 'TPU'] as const).map((v) => (
                    <button key={v} type="button" aria-pressed={print.material === v} onClick={() => setP({ material: v })} style={segBtn(print.material === v, { height: 32, fontFamily: MONO, fontSize: 12 })}>{v}</button>
                  ))}
                </div>
                <span style={label}>Nozzle</span>
                <div style={segWrap(3)}>
                  {['0.2', '0.4', '0.6', '0.8'].map((v) => (
                    <button key={v} type="button" aria-pressed={print.nozzle === v} onClick={() => setPrint((p) => withNozzle(p, v))} style={segBtn(print.nozzle === v, { height: 32, fontFamily: MONO, fontSize: 13 })}>{v}</button>
                  ))}
                </div>
                {sliders.map((s) => [
                  <span key={s.label + 'l'} style={label}>{s.label}</span>,
                  <div key={s.label + 'c'} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <input type="range" min={s.min} max={s.max} step={s.step} value={s.value} aria-label={s.label} onChange={(e) => s.set(+e.target.value)} style={{ flex: 1, accentColor: 'var(--accent)' }} />
                    <span style={{ width: 66, textAlign: 'right', fontFamily: MONO, fontSize: 15, fontVariantNumeric: 'tabular-nums' }}>{s.text}<span style={{ fontWeight: 300, fontSize: 12, color: 'var(--muted)' }}>{s.unit}</span></span>
                  </div>,
                ])}
                <span style={label}>Pattern</span>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {(['grid', 'gyroid', 'rectilinear', 'honeycomb', 'cubic', 'lightning'] as const).map((v) => (
                    <button key={v} type="button" aria-pressed={print.pattern === v} onClick={() => setP({ pattern: v })} style={pillBtn(print.pattern === v)}>{v}</button>
                  ))}
                </div>
                <span style={label}>Supports</span>
                <div style={segWrap(3)}>
                  {SUPPORTS.map(([l, v]) => (
                    <button key={v} type="button" aria-pressed={print.supports === v} onClick={() => setP({ supports: v })} style={segBtn(print.supports === v, { height: 32, fontSize: 13, whiteSpace: 'nowrap' })}>{l}</button>
                  ))}
                </div>
              </div>
              <div style={{ marginTop: 'auto', fontSize: 13, lineHeight: 1.4, color: 'var(--muted)' }}>If the part is too big for the bed at this scale, G-code is skipped with a note — your other files still download.</div>
              <button type="button" onClick={() => setDrawer(false)} style={{ height: 50, flex: 'none', borderRadius: 12, border: 'none', background: 'var(--accent)', color: 'var(--on-accent)', font: 'inherit', fontSize: 15, fontWeight: 600, cursor: 'pointer' }}>Done</button>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
