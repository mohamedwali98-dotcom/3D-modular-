import type { NavItem } from '../components/Shell';
import type { State } from '../state/store';
import { countChecks } from './provenance';

export function buildNav(state: State): NavItem[] {
  const { screen, items, jobId, jobError, job, analysis, model, typed } = state;
  // Spin only while a started job is really running: not before one exists, and not once it was lost.
  const running = screen === 'analyzing' && !!jobId && !jobError && (!job || job.status === 'running');
  const inCapture = screen === 'capture' || screen === 'analyzing';
  const spec = analysis?.spec ?? null;
  const stop = !!analysis?.abstain;
  const nCheck = spec ? countChecks(spec, typed) : 0;
  const n = items.length;
  const failed = job?.status === 'failed' || (!!jobError && job?.status !== 'done');

  const capture: NavItem = {
    num: '01', label: 'Capture', screen: 'capture',
    st: inCapture ? 'current' : 'done',
    sub: screen === 'analyzing' && jobId
      ? (running ? 'Analyzing…' : job?.status === 'done' ? 'Analysis done' : job?.status === 'failed' ? 'Analysis failed'
        : jobError ? 'Analysis lost' : 'Stopped')
      : n ? `${n} ${n === 1 ? 'image' : 'images'}` : 'Add your sketches',
    subC: screen === 'analyzing' && jobId ? (failed ? 'var(--stop)' : 'var(--accent-ink)') : null,
    spinning: running,
  };
  // The Describe chat takes Capture's slot: it is the other way to start a part.
  const described = analysis?.request_id === '' && !inCapture;
  const first: NavItem = screen === 'describe' || described
    ? { num: '01', label: 'Describe', screen: 'describe', st: screen === 'describe' ? 'current' : 'done', sub: 'Chat with the AI', subC: null, spinning: false }
    : capture;
  const review: NavItem = {
    num: '02', label: 'Review', screen: 'review',
    st: !analysis ? 'locked' : screen === 'review' ? 'current' : screen === 'model' ? 'done' : 'open',
    sub: !analysis ? 'After analysis' : stop ? 'Needs your input' : nCheck ? `${nCheck} to check` : 'All checked',
    subC: !analysis ? null : stop ? 'var(--stop)' : nCheck ? 'var(--check)' : 'var(--trusted)',
  };
  const modelOpen = !!spec && !stop && (!!model || screen === 'model');
  const modelItem: NavItem = {
    num: '03', label: 'Model', screen: 'model',
    st: !modelOpen ? 'locked' : screen === 'model' ? 'current' : 'open',
    sub: !modelOpen ? (spec && !stop ? 'Build to open' : 'After review') : model?.abstain ? 'Could not build' : model ? 'Built' : 'Building…',
  };
  return [first, review, modelItem];
}
