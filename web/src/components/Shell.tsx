import { useCallback, useState, type ReactNode } from 'react';
import type { Screen } from '../state/store';
import { useDeadline, resetDeadline } from '../lib/deadline';

export { useDeadline, resetDeadline };

export type NavState = 'current' | 'done' | 'open' | 'locked';
export interface NavItem {
  num: string;
  label: string;
  st: NavState;
  sub: string;
  subC?: string | null;
  spinning?: boolean;
  screen?: Screen;
}
export type Theme = 'dark' | 'light';

export interface ShellProps {
  nav: NavItem[];
  onNav: (screen: Screen) => void;
  onAbout?: () => void;
  children: ReactNode;
}

const MONO = "'Geist Mono', monospace";

function readLS(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeLS(key: string, v: string) {
  try { localStorage.setItem(key, v); } catch { /* storage blocked */ }
}

/** Theme stored in `s2c_theme`, applied as `data-theme` on <html>, switched with a view transition. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(() => {
    const t: Theme = readLS('s2c_theme') === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', t);
    return t;
  });
  const setTheme = useCallback((th: Theme) => {
    writeLS('s2c_theme', th);
    const go = () => { document.documentElement.setAttribute('data-theme', th); setThemeState(th); };
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    if (doc.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) doc.startViewTransition(go);
    else go();
  }, []);
  return [theme, setTheme];
}


/** App frame: the sidebar from Analyzing v2.dc.html (<aside>) plus a scrolling <main>. */
export function Shell({ nav, onNav, onAbout, children }: ShellProps) {
  const [sb, setSb] = useState(() => readLS('s2c_sb') !== '0');
  const [theme, setTheme] = useTheme();
  const ex = sb;
  const dark = theme !== 'light';
  const toggleSb = () => { writeLS('s2c_sb', ex ? '0' : '1'); setSb(!ex); };
  const jc = ex ? 'flex-start' : 'center';

  return (
    <div className="s2c-app" style={{ width: '100%', height: '100vh', display: 'flex', overflow: 'hidden', background: 'var(--bg)', backgroundImage: 'radial-gradient(ellipse 70% 60% at 60% 45%, var(--surface), transparent 70%)', fontFamily: "'Geist', system-ui, sans-serif", color: 'var(--ink)' }}>
      <aside style={{ width: ex ? '248px' : '76px', flex: 'none', boxSizing: 'border-box', padding: '20px 14px', display: 'flex', flexDirection: 'column', gap: 20, background: 'var(--surface)', borderRight: '1px solid var(--line)', transition: 'width 260ms cubic-bezier(.2,.8,.2,1)', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, height: 40, padding: '0 6px' }}>
          <div style={{ width: 34, height: 34, flex: 'none', borderRadius: 9, background: 'var(--accent)', position: 'relative' }}>
            <div style={{ position: 'absolute', left: 10, top: 8, width: 10, height: 13, borderLeft: '3px solid var(--on-accent)', borderBottom: '3px solid var(--on-accent)' }} />
          </div>
          {ex && (
            <>
              <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: '-0.01em', whiteSpace: 'nowrap' }}>Sketch-to-CAD</span>
              <button onClick={toggleSb} aria-label="Collapse menu" style={{ marginLeft: 'auto', width: 36, height: 36, flex: 'none', borderRadius: 9, border: '1px solid var(--line)', background: 'transparent', color: 'var(--muted)', fontFamily: MONO, fontSize: 14, cursor: 'pointer' }}>«</button>
            </>
          )}
        </div>
        {!ex && (
          <button onClick={toggleSb} aria-label="Expand menu" style={{ width: 48, height: 36, borderRadius: 9, border: '1px solid var(--line)', background: 'transparent', color: 'var(--muted)', fontFamily: MONO, fontSize: 14, cursor: 'pointer' }}>»</button>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {ex && <span style={{ padding: '0 8px', fontFamily: MONO, fontSize: 11, letterSpacing: '0.12em', color: 'var(--muted)' }}>[ STEPS ]</span>}
          <nav style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: 5, borderRadius: 14, background: 'var(--inset)' }}>
            {nav.map((n) => {
              const cur = n.st === 'current', locked = n.st === 'locked';
              const bd = cur ? 'none' : locked ? '1.5px dashed var(--line)' : n.st === 'done' ? '1.5px solid var(--trusted)' : '1.5px solid var(--ink)';
              const badgeFg = cur ? 'var(--on-accent)' : n.st === 'done' ? 'var(--trusted)' : locked ? 'var(--muted)' : 'var(--ink)';
              return (
                <a
                  key={n.num}
                  href={n.screen ? `#${n.screen}` : '#'}
                  aria-disabled={locked}
                  aria-current={cur ? 'step' : undefined}
                  title={ex ? undefined : n.label}
                  onClick={(e) => { e.preventDefault(); if (!locked && n.screen) onNav(n.screen); }}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: jc, gap: 12, height: 56, padding: '0 9px', borderRadius: 10, background: cur ? 'var(--raised)' : 'transparent', boxShadow: cur ? 'var(--shadow)' : 'none', color: locked ? 'var(--muted)' : 'var(--ink)', textDecoration: 'none', pointerEvents: locked ? 'none' : 'auto' }}
                >
                  <span style={{ position: 'relative', width: 32, height: 32, flex: 'none', borderRadius: 9, boxSizing: 'border-box', border: bd, background: cur ? 'var(--accent)' : 'transparent', color: badgeFg, display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 12, fontWeight: 500 }}>
                    {n.st === 'done' ? '✓' : n.num}
                    <span className={n.spinning ? 's2c-spin' : undefined} style={{ position: 'absolute', inset: -4, borderRadius: 12, border: '2px solid transparent', borderTopColor: 'var(--accent)', borderRightColor: 'var(--accent)', opacity: n.spinning ? 1 : 0 }} />
                  </span>
                  {ex && (
                    <span style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0, whiteSpace: 'nowrap' }}>
                      <span style={{ fontSize: 15, fontWeight: 500 }}>{n.label}</span>
                      <span style={{ fontSize: 12, color: n.subC || 'var(--muted)' }}>{n.sub}</span>
                    </span>
                  )}
                </a>
              );
            })}
          </nav>
        </div>
        <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {onAbout && (
            <button onClick={onAbout} className="s2c-hover" style={{ display: 'flex', alignItems: 'center', justifyContent: jc, gap: 12, height: 44, padding: '0 12px', borderRadius: 10, border: '1px solid var(--line)', background: 'transparent', color: 'var(--ink)', font: 'inherit', fontSize: 14, cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <span style={{ width: 18, height: 18, flex: 'none', border: '1.5px solid var(--ink)', borderRadius: '50%', boxSizing: 'border-box', display: 'grid', placeItems: 'center', fontFamily: MONO, fontSize: 10 }}>i</span>
              {ex && <span>How this was made</span>}
            </button>
          )}
          {ex ? (
            <div role="radiogroup" aria-label="Theme" style={{ display: 'flex', gap: 4, padding: 4, borderRadius: 11, background: 'var(--inset)' }}>
              <button onClick={() => setTheme('dark')} aria-checked={dark} role="radio" style={{ flex: 1, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, borderRadius: 8, border: 'none', background: dark ? 'var(--raised)' : 'transparent', boxShadow: dark ? 'var(--shadow)' : 'none', color: 'var(--ink)', font: 'inherit', fontSize: 13, cursor: 'pointer' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--ink)' }} />Dark
              </button>
              <button onClick={() => setTheme('light')} aria-checked={!dark} role="radio" style={{ flex: 1, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, borderRadius: 8, border: 'none', background: !dark ? 'var(--raised)' : 'transparent', boxShadow: !dark ? 'var(--shadow)' : 'none', color: 'var(--ink)', font: 'inherit', fontSize: 13, cursor: 'pointer' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', border: '1.5px solid var(--ink)', boxSizing: 'border-box' }} />Light
              </button>
            </div>
          ) : (
            <button onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label="Switch theme" style={{ width: 48, height: 44, borderRadius: 10, border: '1px solid var(--line)', background: 'transparent', cursor: 'pointer', display: 'grid', placeItems: 'center' }}>
              <span style={{ width: 16, height: 16, borderRadius: '50%', border: '1.5px solid var(--ink)', boxSizing: 'border-box', background: 'linear-gradient(90deg, var(--ink) 50%, transparent 50%)' }} />
            </button>
          )}
        </div>
      </aside>
      <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>{children}</main>
    </div>
  );
}
