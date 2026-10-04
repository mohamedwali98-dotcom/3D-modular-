import { useEffect, useRef, useState, type RefObject } from 'react';
import * as T from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { BadgeKey } from '../lib/provenance';
import { BADGE } from '../lib/provenance';
import { zoomed } from '../lib/orbit';

export interface DimLabel { value: string; prov: BadgeKey }

export interface ViewerProps {
  /** GLB in metres, Y-up. Shown in millimetres. */
  glbUrl: string | null;
  /** Width (X), height (Y), depth (Z) labels. */
  labels: [DimLabel, DimLabel, DimLabel] | null;
  showDims: boolean;
  section: boolean;
  /** Increment to reset the camera (the design's "Auto-fit"). */
  fitSignal: number;
  /** Shown when there is no GLB yet. */
  placeholder?: string;
}

// From Model v2.dc.html
const SCENE = { dark: { grid1: 0x3a3733, grid2: 0x2b2926, dim: 0xe9e5de }, light: { grid1: 0xa9a59f, grid2: 0xc6c3be, dim: 0x2a2825 } };

interface Orbit { th: number; ph: number; rad: number; drag: [number, number] | null; idle: number }
interface Fit { size: number; h: number; d: number }

interface Ctx {
  renderer: T.WebGLRenderer; scene: T.Scene; cam: T.PerspectiveCamera;
  grid: T.GridHelper; gridSize: number; gridTheme: string;
  plane: T.Plane; mat: T.MeshStandardMaterial; edgeMat: T.LineBasicMaterial; dimMat: T.LineBasicMaterial;
  part: T.Group; dimGroup: T.LineSegments | null;
  anchors: T.Vector3[]; orbit: Orbit; fit: Fit | null; grow: number | null;
}

const fitOrbit = (o: Orbit, size: number) => Object.assign(o, { th: 0.7, ph: 1.05, rad: size * 3, idle: 0 });

function makeGrid(size: number, theme: string): T.GridHelper {
  const c = SCENE[theme === 'light' ? 'light' : 'dark'];
  const divisions = Math.min(120, Math.max(40, Math.round((size / 200) * 40)));
  const g = new T.GridHelper(size, divisions, c.grid1, c.grid2);
  const m = g.material as T.Material;
  m.transparent = true; m.opacity = 0.8;
  return g;
}

function disposeGrid(g: T.GridHelper) {
  g.geometry.dispose();
  (Array.isArray(g.material) ? g.material : [g.material]).forEach((m) => m.dispose());
}

function clearPart(part: T.Group) {
  part.children.slice().forEach((c) => {
    part.remove(c);
    (c as T.Mesh).geometry?.dispose();
  });
}

const themeNow = () => (document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark');

/** Live 3D preview of the built part: the design's scene, lights, grid, orbit, section cut and dimension anchors. */
export function Viewer({ glbUrl, labels, showDims, section, fitSignal, placeholder = '3D preview unavailable' }: ViewerProps) {
  const viewRef = useRef<HTMLDivElement>(null);
  const dimW = useRef<HTMLDivElement>(null), dimH = useRef<HTMLDivElement>(null), dimD = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Ctx | null>(null);
  const flags = useRef({ section, showDims });
  flags.current = { section, showDims };
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Scene, renderer, orbit and render loop (mount only).
  useEffect(() => {
    const el = viewRef.current;
    if (!el) return;
    let dead = false, raf = 0;
    let r: T.WebGLRenderer;
    try {
      r = new T.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    } catch {
      setFailed(true);
      return;
    }
    r.setPixelRatio(Math.min(2, devicePixelRatio)); r.localClippingEnabled = true;
    el.appendChild(r.domElement);
    const scene = new T.Scene();
    const cam = new T.PerspectiveCamera(30, 1, 1, 2000);
    scene.add(new T.HemisphereLight(0xfff4ea, 0x2a2622, 1.2));
    const key = new T.DirectionalLight(0xffffff, 1.9); key.position.set(50, 90, 70); scene.add(key);
    const fill = new T.DirectionalLight(0xffd9b8, 0.6); fill.position.set(-60, 30, -60); scene.add(fill);
    const gridTheme = themeNow();
    const grid = makeGrid(200, gridTheme); scene.add(grid);
    const plane = new T.Plane(new T.Vector3(0, 0, -1), 1e6);
    const mat = new T.MeshStandardMaterial({ color: 0xc98a66, metalness: 0.0, roughness: 0.38, clippingPlanes: [plane], side: T.DoubleSide });
    const edgeMat = new T.LineBasicMaterial({ color: 0x5a3624, transparent: true, opacity: 0.55, clippingPlanes: [plane] });
    const dimMat = new T.LineBasicMaterial({ color: SCENE[gridTheme === 'light' ? 'light' : 'dark'].dim });
    const part = new T.Group(); scene.add(part);
    const ctx: Ctx = {
      renderer: r, scene, cam, grid, gridSize: 200, gridTheme, plane, mat, edgeMat, dimMat, part, dimGroup: null,
      anchors: [], orbit: { th: 0.7, ph: 1.05, rad: 150, drag: null, idle: 0 }, fit: null, grow: null,
    };
    ctxRef.current = ctx;

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      r.setSize(w, h); cam.aspect = w / h; cam.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize); ro.observe(el); resize();

    // one finger (or the mouse) orbits; two fingers pinch to zoom, within the wheel's limits
    const touches = new Map<number, [number, number]>();
    const spread = () => { const [a, b] = [...touches.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]); };
    let pinch = 0;
    const onDown = (e: PointerEvent) => {
      touches.set(e.pointerId, [e.clientX, e.clientY]);
      el.setPointerCapture(e.pointerId);
      if (touches.size === 2) { pinch = spread(); ctx.orbit.drag = null; return; }
      ctx.orbit.drag = [e.clientX, e.clientY]; el.style.cursor = 'grabbing';
    };
    const onMove = (e: PointerEvent) => {
      if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.clientX, e.clientY]);
      if (touches.size === 2) {
        const d = spread();
        if (pinch > 0 && d > 0) ctx.orbit.rad = zoomed(ctx.orbit.rad, ctx.fit?.size ?? 50, pinch / d);
        pinch = d; ctx.orbit.idle = 0;
        return;
      }
      const o = ctx.orbit; if (!o.drag) return;
      o.th -= (e.clientX - o.drag[0]) * 0.008;
      o.ph = Math.max(0.2, Math.min(1.5, o.ph - (e.clientY - o.drag[1]) * 0.006));
      o.drag = [e.clientX, e.clientY]; o.idle = 0;
    };
    const onUp = (e: PointerEvent) => { touches.delete(e.pointerId); pinch = 0; ctx.orbit.drag = null; el.style.cursor = 'grab'; };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const s = ctx.fit?.size ?? 50;
      ctx.orbit.rad = zoomed(ctx.orbit.rad, s, 1 + e.deltaY * 0.001);
    };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointercancel', onUp);
    el.addEventListener('wheel', onWheel, { passive: false });

    const rm = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const refs: RefObject<HTMLDivElement>[] = [dimW, dimH, dimD];
    const tgt = new T.Vector3();
    let last = performance.now();
    const loop = (ts: number) => {
      if (dead) return;
      const dt = Math.min(0.1, (ts - last) / 1000); last = ts;
      const o = ctx.orbit, f = flags.current, fit = ctx.fit;
      const th = themeNow();
      if (th !== ctx.gridTheme) {
        ctx.gridTheme = th;
        scene.remove(ctx.grid); disposeGrid(ctx.grid);
        ctx.grid = makeGrid(ctx.gridSize, th); scene.add(ctx.grid);
        dimMat.color.setHex(SCENE[th === 'light' ? 'light' : 'dark'].dim);
      }
      plane.constant = f.section ? 0 : 1e6;
      if (ctx.dimGroup) ctx.dimGroup.visible = f.showDims;
      o.idle += dt; if (!o.drag && o.idle > 1.5 && !rm) o.th += dt * 0.18;
      if (ctx.grow != null && fit) {
        ctx.grow = Math.min(1, ctx.grow + dt / 0.4);
        const e = 1 - Math.pow(1 - ctx.grow, 3);
        part.scale.set(1, 1, rm ? 1 : Math.max(0.02, e));
        part.position.z = rm ? 0 : fit.d * 0.5 * (1 - e);
        if (ctx.grow >= 1) ctx.grow = null;
      }
      tgt.set(0, fit ? fit.h / 3 : 10, 0);
      cam.position.set(o.rad * Math.sin(o.ph) * Math.sin(o.th), tgt.y + o.rad * Math.cos(o.ph), o.rad * Math.sin(o.ph) * Math.cos(o.th));
      cam.lookAt(tgt); r.render(scene, cam);
      const w = el.clientWidth, h = el.clientHeight;
      ctx.anchors.forEach((v, i) => {
        const node = refs[i].current; if (!node) return;
        const p = v.clone().project(cam);
        node.style.transform = `translate(${(p.x + 1) / 2 * w}px, ${(1 - p.y) / 2 * h}px) translate(-50%, -50%)`;
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      dead = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointercancel', onUp);
      el.removeEventListener('wheel', onWheel);
      clearPart(part);
      if (ctx.dimGroup) { scene.remove(ctx.dimGroup); ctx.dimGroup.geometry.dispose(); }
      disposeGrid(ctx.grid);
      mat.dispose(); edgeMat.dispose(); dimMat.dispose();
      r.dispose();
      r.forceContextLoss();  // phones allow few WebGL contexts: Review <-> Model trips must not use them up
      r.domElement.remove();
      ctxRef.current = null;
    };
  }, []);

  // Load (and reload) the GLB. Stale loads are ignored; the old part is disposed when the new one lands.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!glbUrl) {
      setLoaded(false); setFailed(false);
      if (ctx) {
        clearPart(ctx.part);
        if (ctx.dimGroup) { ctx.scene.remove(ctx.dimGroup); ctx.dimGroup.geometry.dispose(); ctx.dimGroup = null; }
        ctx.anchors = []; ctx.fit = null;
      }
      return;
    }
    if (!ctx) { setFailed(true); return; }
    let stale = false;
    new GLTFLoader().load(glbUrl, (gltf) => {
      const geos: T.BufferGeometry[] = [];
      gltf.scene.updateMatrixWorld(true);
      gltf.scene.traverse((o) => {
        const m = o as T.Mesh;
        if (m.isMesh && m.geometry) {
          if (!stale) {
            const g = m.geometry.clone();
            g.applyMatrix4(m.matrixWorld);
            g.scale(1000, 1000, 1000); // metres -> millimetres
            if (!g.getAttribute('normal')) g.computeVertexNormals();
            geos.push(g);
          }
          m.geometry.dispose();
          (Array.isArray(m.material) ? m.material : [m.material]).forEach((mt) => {
            Object.values(mt).forEach((v) => { if (v instanceof T.Texture) v.dispose(); });
            mt.dispose();
          });
        }
      });
      if (stale || ctxRef.current !== ctx) { geos.forEach((g) => g.dispose()); return; }
      if (!geos.length) { setFailed(true); return; }

      // Centre on the grid: X and Z centred, sitting on Y = 0.
      const box = new T.Box3();
      geos.forEach((g) => { g.computeBoundingBox(); box.union(g.boundingBox!); });
      const c = box.getCenter(new T.Vector3());
      geos.forEach((g) => { g.translate(-c.x, -box.min.y, -c.z); g.computeBoundingBox(); });
      const w = box.max.x - box.min.x, h = box.max.y - box.min.y, d = box.max.z - box.min.z;
      const size = Math.max(w, h, d, 1);

      const first = !ctx.fit;
      clearPart(ctx.part);
      geos.forEach((g) => {
        ctx.part.add(new T.Mesh(g, ctx.mat));
        ctx.part.add(new T.LineSegments(new T.EdgesGeometry(g, 25), ctx.edgeMat));
      });
      ctx.fit = { size, h, d };

      // Grid and camera range follow the part size.
      const gridSize = Math.max(200, Math.ceil((size * 4) / 50) * 50);
      if (gridSize !== ctx.gridSize) {
        ctx.scene.remove(ctx.grid); disposeGrid(ctx.grid);
        ctx.gridSize = gridSize; ctx.grid = makeGrid(gridSize, ctx.gridTheme); ctx.scene.add(ctx.grid);
      }
      ctx.cam.near = Math.max(0.1, size / 50); ctx.cam.far = size * 60; ctx.cam.updateProjectionMatrix();
      if (first) { fitOrbit(ctx.orbit, size); ctx.grow = 0; }
      else ctx.orbit.rad = Math.max(size * 1.4, Math.min(size * 6.4, ctx.orbit.rad));

      // Dimension lines and label anchors at the bounding-box edges (layout from the design's L-bracket).
      const x0 = -w / 2, x1 = w / 2, z0 = -d / 2, z1 = d / 2;
      const o1 = size * 0.12, o2 = size * 0.08, t = size * 0.04;
      const pts = [
        x0, 0, z1 + o1, x1, 0, z1 + o1, x0, 0, z1, x0, 0, z1 + o1 + t, x1, 0, z1, x1, 0, z1 + o1 + t,
        x1 + o2, 0, z1, x1 + o2, h, z1, x1, 0, z1, x1 + o2 + t, 0, z1, x1, h, z1, x1 + o2 + t, h, z1,
        x0 - o2, 0, z1, x0 - o2, 0, z0, x0, 0, z1, x0 - o2 - t, 0, z1, x0, 0, z0, x0 - o2 - t, 0, z0,
      ];
      if (ctx.dimGroup) { ctx.scene.remove(ctx.dimGroup); ctx.dimGroup.geometry.dispose(); }
      const dg = new T.BufferGeometry(); dg.setAttribute('position', new T.Float32BufferAttribute(pts, 3));
      ctx.dimGroup = new T.LineSegments(dg, ctx.dimMat); ctx.scene.add(ctx.dimGroup);
      ctx.anchors = [new T.Vector3(0, 0, z1 + o1), new T.Vector3(x1 + o2, h / 2, z1), new T.Vector3(x0 - o2, 0, 0)];
      setFailed(false); setLoaded(true);
    }, undefined, () => { if (!stale) setFailed(true); });
    return () => { stale = true; };
  }, [glbUrl]);

  // The design's "Auto-fit".
  useEffect(() => {
    const ctx = ctxRef.current;
    if (fitSignal && ctx) fitOrbit(ctx.orbit, ctx.fit?.size ?? 50);
  }, [fitSignal]);

  const dimsO = showDims && loaded && !failed && labels ? 1 : 0;
  const message = failed ? '3D preview unavailable' : !glbUrl ? placeholder : null;
  const label = (ref: RefObject<HTMLDivElement>, l: DimLabel | undefined) => {
    const b = l ? BADGE[l.prov] ?? BADGE.default : BADGE.default;
    return (
      <div ref={ref} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', opacity: l ? dimsO : 0, display: 'flex', alignItems: 'center', gap: 6, height: 30, padding: '0 10px', borderRadius: 8, border: `1.5px ${b.line} ${b.fg}`, background: 'var(--raised)', boxShadow: 'var(--shadow)', whiteSpace: 'nowrap' }}>
        <span style={{ fontFamily: "'Geist Mono', monospace", fontSize: 15, fontWeight: 500 }}>{l?.value}<span style={{ fontWeight: 300, fontSize: 12, color: 'var(--muted)' }}> mm</span></span>
        <span style={{ fontSize: 12, fontWeight: 500, color: b.fg }}>{b.icon} {b.label}</span>
      </div>
    );
  };

  return (
    <>
      <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px)', backgroundSize: '32px 32px' }} />
      <div ref={viewRef} aria-label="3D view of your part. Drag to orbit, scroll to zoom." role="img"
        style={{ position: 'absolute', inset: 0, cursor: 'grab', touchAction: 'none', opacity: failed ? 0 : 1 }} />
      {label(dimW, labels?.[0])}
      {label(dimH, labels?.[1])}
      {label(dimD, labels?.[2])}
      {message && (
        <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', pointerEvents: 'none' }}>
          <span style={{ fontFamily: "'Geist Mono', monospace", fontSize: 13, letterSpacing: '0.06em', color: 'var(--muted)', padding: '10px 14px', borderRadius: 10, background: 'var(--raised)', boxShadow: 'var(--shadow)' }}>{message}</span>
        </div>
      )}
    </>
  );
}
