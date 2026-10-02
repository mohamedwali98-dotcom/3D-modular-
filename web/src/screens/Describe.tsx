import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react';
import { ApiError, chat } from '../api/client';
import type { ChatMessage } from '../api/types';
import { Badge } from '../components/Badge';
import { StopCard } from '../components/StopCard';
import { PART_LABELS, describedAnalysis, formatValue, frontPreview, partRows } from '../lib/describe';
import { useStore } from '../state/store';

const MONO = "'Geist Mono', monospace";
const SILK = "'Silkscreen', monospace";

const OPENING = 'Describe the part you need — for example: a 60 × 40 mm plate, 5 mm thick, with two 6 mm holes.';
const OPENING_OPTIONS = ['A flat plate', 'An L-bracket', 'A spacer', 'A flange'];

const panel: CSSProperties = {
  borderRadius: 14, background: 'var(--surface)', boxShadow: 'var(--shadow)', padding: '16px 18px',
  display: 'flex', flexDirection: 'column', gap: 12, boxSizing: 'border-box',
};
const tag = (color: string): CSSProperties => ({
  alignSelf: 'flex-start', fontFamily: SILK, fontSize: 10, letterSpacing: '0.04em', textTransform: 'uppercase',
  color, padding: '2px 6px', borderRadius: 4, border: `1px solid ${color}`, lineHeight: 1.3,
});

const CSS = `
@keyframes s2c-dots{0%,80%,100%{opacity:.25}40%{opacity:1}}
.s2c-dot{animation:s2c-dots 1.2s infinite}
.s2c-dot:nth-child(2){animation-delay:.15s}.s2c-dot:nth-child(3){animation-delay:.3s}
@media (max-width:1023px){
  .s2c-describe-grid{grid-template-columns:minmax(0,1fr)!important}
  .s2c-describe-header{height:auto!important;flex-wrap:wrap}
}`;

function Bubble({ msg, model }: { msg: ChatMessage; model?: string }) {
  const user = msg.role === 'user';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: user ? 'flex-end' : 'flex-start', gap: 6 }}>
      {!user && <span style={tag('var(--ai)')}>{model || 'Assistant'}</span>}
      <div
        style={{
          maxWidth: '82%', padding: '10px 14px', borderRadius: 14, fontSize: 15, lineHeight: 1.45, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere',
          ...(user
            ? { background: 'color-mix(in oklch, var(--accent) 16%, var(--surface))', border: '1.5px solid color-mix(in oklch, var(--accent) 45%, transparent)', borderBottomRightRadius: 4 }
            : { background: 'var(--raised)', border: '1.5px dashed var(--ai)', borderBottomLeftRadius: 4 }),
        }}
      >
        {msg.content}
      </div>
    </div>
  );
}

/** Describe a part in words: the chat model asks for what is missing, our own code builds it. */
export function Describe() {
  const { state, dispatch } = useStore();
  const { messages, last } = state.chat;
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, busy, error]);

  const send = async (text: string, history: ChatMessage[] = messages) => {
    const content = text.trim();
    if (!content || busy) return;
    const next: ChatMessage[] = [...history, { role: 'user', content }];
    dispatch({ type: 'CHAT', messages: next });
    setDraft('');
    setBusy(true);
    setError(null);
    try {
      const r = await chat(next.slice(-20));
      dispatch({ type: 'CHAT', messages: [...next, { role: 'assistant', content: r.reply, sig: r.sig }], last: r });
    } catch (e) {
      if (alive.current) setError(e instanceof ApiError ? e.message : 'Something went wrong. Try again.');
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  // Retry the last user message without adding it twice.
  const retry = () => {
    const i = messages.length - 1;
    if (i >= 0 && messages[i].role === 'user') void send(messages[i].content, messages.slice(0, i));
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send(draft);
    }
  };

  const startOver = () => {
    dispatch({ type: 'CHAT', messages: [], last: null });
    setError(null);
    setDraft('');
  };

  const spec = last?.spec ?? null;
  const part = last?.part ?? null;
  const rows = partRows(part, last?.missing ?? []);
  const preview = spec ? frontPreview(spec) : null;
  const lastIsAssistant = messages.length === 0 || messages[messages.length - 1].role === 'assistant';
  const options = busy || !lastIsAssistant ? [] : messages.length === 0 ? OPENING_OPTIONS : last?.options ?? [];

  const build = () => {
    if (!spec) return;
    dispatch({ type: 'ANALYSIS', analysis: describedAnalysis(spec) });
    dispatch({ type: 'MODEL', model: null });
    dispatch({ type: 'GOTO', screen: 'model' });
  };

  const canSend = !!draft.trim() && !busy;

  return (
    <section data-screen="Describe" style={{ padding: 28, display: 'flex', flexDirection: 'column', gap: 20, flex: 1, minHeight: 0, boxSizing: 'border-box' }}>
      <style>{CSS}</style>
      <header className="s2c-describe-header" style={{ height: 64, flex: 'none', display: 'flex', alignItems: 'flex-end', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <span style={{ fontFamily: MONO, fontSize: 13, letterSpacing: '0.08em', color: 'var(--muted)' }}>[1/3] DESCRIBE</span>
          <h1 style={{ margin: 0, fontFamily: SILK, fontWeight: 400, fontSize: 40, lineHeight: 1, letterSpacing: '0.01em' }}>Describe</h1>
        </div>
        <p style={{ margin: '0 0 4px', maxWidth: 440, fontSize: 15, lineHeight: 1.4, color: 'var(--muted)' }}>
          Tell the AI what you need. It asks for every size it is missing — it never guesses one. Our own code builds the part.
        </p>
        <button
          type="button" onClick={() => dispatch({ type: 'GOTO', screen: 'capture' })}
          style={{ marginLeft: 'auto', marginBottom: 4, height: 40, padding: '0 16px', borderRadius: 10, border: '1px solid var(--line)', background: 'var(--raised)', color: 'var(--ink)', font: 'inherit', fontSize: 14, fontWeight: 500, whiteSpace: 'nowrap', cursor: 'pointer', boxShadow: 'var(--shadow)' }}
        >
          ← Use a sketch or photo
        </button>
      </header>

      <div className="s2c-describe-grid" style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: 'minmax(0,3fr) minmax(0,2fr)', gap: 24 }}>
        <section style={{ ...panel, padding: 0, gap: 0, minHeight: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '14px 18px', borderBottom: '1px solid var(--line)' }}>
            <span style={{ fontSize: 15, fontWeight: 600 }}>Design chat</span>
            <span style={tag('var(--ai)')}>AI asks, you answer</span>
            {messages.length > 0 && (
              <button type="button" onClick={startOver} disabled={busy}
                style={{ marginLeft: 'auto', height: 30, padding: '0 12px', borderRadius: 8, border: '1px solid var(--line)', background: 'transparent', color: 'var(--muted)', font: 'inherit', fontSize: 13, cursor: busy ? 'not-allowed' : 'pointer' }}>
                Start over
              </button>
            )}
          </div>

          <div ref={scroller} style={{ flex: 1, minHeight: 200, overflow: 'auto', padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Bubble msg={{ role: 'assistant', content: OPENING }} model="Sketch-to-CAD" />
            {messages.map((m, i) => <Bubble key={i} msg={m} model={last?.model} />)}
            {busy && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
                <span style={tag('var(--ai)')}>{last?.model || 'Assistant'}</span>
                <div style={{ padding: '10px 14px', borderRadius: 14, borderBottomLeftRadius: 4, border: '1.5px dashed var(--ai)', color: 'var(--muted)', fontFamily: MONO, fontSize: 13 }}>
                  thinking<span className="s2c-dot">.</span><span className="s2c-dot">.</span><span className="s2c-dot">.</span>
                </div>
              </div>
            )}
            {error && <StopCard title="The chat did not answer" remedy={error} actionLabel="Try again" onAction={retry} />}
            {options.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, paddingLeft: 2 }}>
                {options.map((o) => (
                  <button key={o} type="button" onClick={() => void send(o)}
                    style={{ height: 34, padding: '0 14px', borderRadius: 17, border: '1.5px solid var(--ai)', background: 'color-mix(in oklch, var(--ai) 10%, transparent)', color: 'var(--ink)', font: 'inherit', fontSize: 14, cursor: 'pointer', whiteSpace: 'nowrap' }}>
                    {o}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', padding: 14, borderTop: '1px solid var(--line)', background: 'var(--surface)' }}>
            <textarea
              value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={onKey} rows={2} maxLength={2000}
              placeholder="A 60 × 40 mm plate, 5 mm thick…" aria-label="Describe your part"
              style={{ flex: 1, resize: 'none', minHeight: 48, maxHeight: 140, borderRadius: 10, border: '1px solid var(--line)', background: 'var(--inset)', color: 'var(--ink)', font: 'inherit', fontSize: 15, lineHeight: 1.4, padding: '12px 12px', boxSizing: 'border-box' }}
            />
            <button type="button" onClick={() => void send(draft)} disabled={!canSend}
              style={{ height: 48, padding: '0 20px', borderRadius: 10, border: 'none', background: canSend ? 'var(--accent)' : 'var(--inset)', color: canSend ? 'var(--on-accent)' : 'var(--muted)', font: 'inherit', fontSize: 15, fontWeight: 600, cursor: canSend ? 'pointer' : 'not-allowed', boxShadow: canSend ? 'var(--shadow)' : 'none' }}>
              Send
            </button>
          </div>
          <div style={{ padding: '0 14px 10px', fontFamily: MONO, fontSize: 11, color: 'var(--muted)' }}>Enter to send · Shift+Enter for a new line</div>
        </section>

        <section style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0, overflow: 'auto' }}>
          <div style={panel}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>Your part</span>
              {part && <span style={tag('var(--accent-ink)')}>{PART_LABELS[part.type] ?? part.type}</span>}
            </div>
            {rows.length === 0 ? (
              <span style={{ fontSize: 14, color: 'var(--muted)' }}>Nothing yet. The sizes you write appear here, each marked as yours.</span>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {rows.map((r) => (
                  <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', borderTop: '1px solid var(--line)' }}>
                    <span style={{ fontSize: 14, flex: 1, minWidth: 0 }}>{r.label}</span>
                    <span style={{ fontFamily: MONO, fontSize: 15, color: r.value == null ? 'var(--muted)' : 'var(--ink)' }}>{r.value == null ? '—' : formatValue(r.key, r.value)}</span>
                    <Badge prov={r.value == null ? 'required' : 'user_written'} />
                  </div>
                ))}
                {part && part.holes.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', borderTop: '1px solid var(--line)' }}>
                    <span style={{ fontSize: 14, flex: 1 }}>Holes</span>
                    <span style={{ fontFamily: MONO, fontSize: 15 }}>{part.holes.length} × ⌀{formatValue('d_mm', part.holes[0].diameter_mm)}</span>
                    <Badge prov="user_written" />
                  </div>
                )}
              </div>
            )}
          </div>

          <div style={{ ...panel, alignItems: 'stretch' }}>
            <span style={{ fontFamily: MONO, fontSize: 11, letterSpacing: '0.1em', color: 'var(--muted)' }}>FRONT VIEW</span>
            <div style={{ aspectRatio: '4 / 3', borderRadius: 10, background: 'var(--inset)', display: 'grid', placeItems: 'center', padding: 14, boxSizing: 'border-box' }}>
              {preview ? (
                <svg viewBox={preview.viewBox} style={{ width: '100%', height: '100%' }} role="img" aria-label="Front outline of the part">
                  <path d={preview.path} fillRule="evenodd" fill="color-mix(in oklch, var(--accent) 22%, transparent)" stroke="var(--accent-ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
                  {preview.holes.map((h, i) => (
                    <circle key={i} cx={h.cx} cy={h.cy} r={h.r} fill="var(--inset)" stroke="var(--accent-ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                  ))}
                </svg>
              ) : (
                <span style={{ fontSize: 13, color: 'var(--muted)', textAlign: 'center' }}>The outline appears once every size is known.</span>
              )}
            </div>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <button
              type="button" onClick={build} disabled={!spec}
              style={{
                width: '100%', height: 52, borderRadius: 12, border: 'none',
                background: spec ? 'var(--accent)' : 'var(--inset)', color: spec ? 'var(--on-accent)' : 'var(--muted)',
                boxShadow: spec ? 'var(--shadow)' : 'none', fontSize: 16, fontWeight: 600, cursor: spec ? 'pointer' : 'not-allowed',
              }}
            >
              Build this part →
            </button>
          </div>
        </section>
      </div>
    </section>
  );
}
