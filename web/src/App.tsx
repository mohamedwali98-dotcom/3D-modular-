import { lazy, Suspense } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Shell } from './components/Shell';
import { buildNav } from './lib/nav';
import { Analyzing } from './screens/Analyzing';
import { Capture } from './screens/Capture';
import { Describe } from './screens/Describe';
import { Review } from './screens/Review';
import { useStore, type Screen } from './state/store';

export { buildNav };

// three.js only loads with the Model screen, in its own chunk.
const Model = lazy(() => import('./screens/Model').then((m) => ({ default: m.Model })));

function Loading() {
  return (
    <div style={{ flex: 1, display: 'grid', placeItems: 'center', color: 'var(--muted)', fontFamily: "'Geist Mono', monospace", fontSize: 12 }}>
      Loading the 3D viewer…
    </div>
  );
}

export function App() {
  const { state, dispatch } = useStore();
  const onNav = (screen: Screen) => {
    // Going back to Capture while a finished analysis exists keeps it; Analyzing is reached through Capture.
    if (screen === 'capture' && state.screen === 'analyzing') return;
    dispatch({ type: 'GOTO', screen });
  };
  return (
    <Shell nav={buildNav(state)} onNav={onNav}>
      <ErrorBoundary key={state.screen}>
        {state.screen === 'capture' && <Capture />}
        {state.screen === 'describe' && <Describe />}
        {state.screen === 'analyzing' && <Analyzing />}
        {state.screen === 'review' && <Review />}
        {state.screen === 'model' && <Suspense fallback={<Loading />}><Model /></Suspense>}
      </ErrorBoundary>
    </Shell>
  );
}
