import type { FilledBy, Outline } from '../api/types';
import { canReject, faceInfo } from '../lib/faces';

const MONO = "'Geist Mono', monospace";

export interface FaceCardProps {
  label: string; // 'FRONT' | 'TOP' | 'RIGHT'
  outline: Outline | null;
  width: number;  // mm, this projection's horizontal extent
  height: number; // mm, this projection's vertical extent
  filledBy: FilledBy;
  rejected: boolean;
  onToggleReject?: () => void;
}

/** Flip y (data is y-up, SVG is y-down) and format an mm point list as an SVG points string. */
function toPoints(pts: [number, number][], h: number): string {
  return pts.map(([x, y]) => `${x},${h - y}`).join(' ');
}

/** One face card in the Review screen: an SVG silhouette from the fused views, with a provenance badge. */
export function FaceCard({ label, outline, width, height, filledBy, rejected, onToggleReject }: FaceCardProps) {
  const isAi = filledBy !== 'observed';
  const info = faceInfo(filledBy);
  const rejectable = canReject(filledBy);
  rejected = rejected && rejectable;  // a face rebuilt the same way whatever the user says is never shown as "not used"
  const w = width > 0 ? width : 1;
  const h = height > 0 ? height : 1;
  const pad = Math.max(w, h) * 0.12 || 1;
  const viewBox = `${-pad} ${-pad} ${w + 2 * pad} ${h + 2 * pad}`;
  const strokeColor = isAi ? 'var(--ai)' : 'oklch(0.58 0.10 45)';
  const fillColor = isAi ? 'color-mix(in oklch, var(--ai) 16%, transparent)' : 'oklch(0.66 0.10 48 / 0.12)';

  return (
    <div style={{
      position: 'relative', borderRadius: 14, padding: 14, display: 'flex', flexDirection: 'column', gap: 10,
      background: isAi ? 'color-mix(in oklch, var(--ai) 7%, var(--surface))' : 'var(--surface)',
      boxShadow: isAi ? 'none' : 'var(--shadow)',
      border: isAi ? '1.5px dashed var(--ai)' : 'none',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
        <span style={{ fontFamily: MONO, fontSize: 12, letterSpacing: '0.08em', color: isAi ? 'var(--ai)' : 'inherit' }}>{label}</span>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 5, height: 24, padding: '0 9px 0 7px', borderRadius: 12,
          border: `1.5px ${isAi ? 'dashed' : 'solid'} ${isAi ? 'var(--ai)' : 'var(--trusted)'}`,
          color: isAi ? 'var(--ai)' : 'var(--trusted)', fontSize: 12, fontWeight: 500, whiteSpace: 'nowrap',
        }}>
          {isAi ? (rejected ? '✕ Rejected' : `◇ ${info.badge}`) : `✓ ${info.badge}`}
        </span>
      </div>

      <div style={{
        position: 'relative', height: 130, borderRadius: 8, overflow: 'hidden', display: 'grid', placeItems: 'center',
        background: isAi
          ? 'var(--inset)'
          : '#f4f3f0',
        backgroundImage: isAi
          ? 'repeating-linear-gradient(135deg, color-mix(in oklch, var(--ai) 10%, transparent) 0 6px, transparent 6px 12px)'
          : undefined,
      }}>
        {outline && outline.outer.length > 0 ? (
          <svg viewBox={viewBox} style={{ width: '100%', height: '100%', opacity: rejected ? 0.25 : 1 }}>
            {outline.inner.map((hole, i) => (
              <polygon key={`hole-${i}`} points={toPoints(hole, h)} vectorEffect="non-scaling-stroke" style={{ fill: isAi ? 'var(--inset)' : '#f4f3f0', stroke: strokeColor, strokeWidth: 1.4 }} />
            ))}
            <polygon points={toPoints(outline.outer, h)} vectorEffect="non-scaling-stroke" style={{ fill: fillColor, stroke: strokeColor, strokeWidth: 1.8, strokeLinejoin: 'round' }} />
            {outline.inner.map((hole, i) => (
              <polygon key={`hole-outline-${i}`} points={toPoints(hole, h)} vectorEffect="non-scaling-stroke" style={{ fill: 'none', stroke: strokeColor, strokeWidth: 1.4 }} />
            ))}
          </svg>
        ) : (
          <span style={{ fontSize: 13, color: 'var(--muted)' }}>No outline yet</span>
        )}
        {rejected && (
          <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', fontSize: 14, fontWeight: 500, color: 'var(--ai)' }}>
            Rejected — not used
          </div>
        )}
      </div>

      <div style={{ fontSize: 13, color: isAi ? 'var(--ai)' : 'var(--muted)' }}>
        {isAi
          ? <>{info.verb}{outline ? <> · confidence <span style={{ fontFamily: MONO }}>{outline.confidence.toFixed(2)}</span></> : null}</>
          : 'From your sketch'}
      </div>

      {rejectable && (
        <div style={{ display: 'flex', gap: 8 }}>
          <button type="button" onClick={onToggleReject} aria-pressed={rejected} style={{
            flex: 1, height: 40, borderRadius: 10, border: '1.5px solid var(--ai)',
            background: rejected ? 'var(--ai)' : 'transparent', color: rejected ? 'var(--raised)' : 'var(--ai)',
            font: 'inherit', fontSize: 13, fontWeight: 500, cursor: 'pointer',
          }}>{rejected ? 'Undo reject' : 'Reject'}</button>
        </div>
      )}
    </div>
  );
}
