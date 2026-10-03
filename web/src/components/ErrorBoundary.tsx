import { Component, type ReactNode } from 'react';

interface Props { children: ReactNode }
interface State { error: Error | null }

/** A screen that throws shows a card with Try again instead of a blank page; the analysis in the store is kept. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" style={{ flex: 1, display: 'grid', placeItems: 'center', padding: 24 }}>
        <div style={{ maxWidth: 420, display: 'grid', gap: 12, textAlign: 'center' }}>
          <strong style={{ fontSize: 17 }}>This screen hit a problem.</strong>
          <span style={{ color: 'var(--muted)', fontSize: 14 }}>Your analysis is kept. Try again, or go back with the menu.</span>
          <button type="button" onClick={() => this.setState({ error: null })} style={{
            justifySelf: 'center', height: 40, padding: '0 18px', borderRadius: 10, border: '1.5px solid var(--ink, currentColor)',
            background: 'transparent', font: 'inherit', fontSize: 14, cursor: 'pointer',
          }}>Try again</button>
        </div>
      </div>
    );
  }
}
