import { createContext, useCallback, useContext, useMemo, useReducer, useRef, type Dispatch, type ReactNode } from 'react';
import type { AiSettings, Analysis, ChatMessage, ChatResponse, Face, GeometrySettings, Job, ModelResult, Spec } from '../api/types';
import { resetDeadline } from '../lib/deadline';

export type Screen = 'capture' | 'describe' | 'analyzing' | 'review' | 'model';
export type CaptureKind = 'auto' | 'sketch' | 'photo' | 'drawing';
export type CaptureMode = 'photos' | 'sheet';
/** How a drawing sheet lays out its views: read from the drawing (auto), ISO first-angle or US third-angle.
 * A projection symbol or the view labels on the sheet win. */
export type Projection = 'auto' | 'first' | 'third';
export interface CaptureItem { id: string; file: File; url: string; face: Face | 'auto'; kind: CaptureKind }
/** The images a job was started with, in upload order: job.images[i] is jobItems[i] even after Capture changes. */
export interface JobItem { url: string; face: Face | 'auto'; kind: CaptureKind }

export interface State {
  screen: Screen;
  mode: CaptureMode;
  projection: Projection;
  items: CaptureItem[];
  reference: string;
  ai: AiSettings;
  jobId: string | null;
  jobItems: JobItem[];
  /** Set when polling gave up on the job (lost connection or unknown job). */
  jobError: string | null;
  job: Job | null;
  analysis: Analysis | null;
  typed: Record<string, number>;
  rejected: Face[];
  geometry: GeometrySettings;
  model: ModelResult | null;
  /** The spec `model` was built from, so the Model screen can rebuild when the analysis changes. */
  modelSpec: Spec | null;
  /** The Describe chat, oldest first. */
  chat: { messages: ChatMessage[]; last: ChatResponse | null };
}

export type Action =
  | { type: 'SET_MODE'; mode: CaptureMode }
  | { type: 'SET_PROJECTION'; projection: Projection }
  | { type: 'ADD_FILES'; items: CaptureItem[] }
  | { type: 'SET_ITEM'; id: string; patch: Partial<Omit<CaptureItem, 'id'>> }
  | { type: 'REMOVE_ITEM'; id: string }
  | { type: 'SET_REFERENCE'; reference: string }
  | { type: 'SET_AI'; patch: Partial<AiSettings> }
  | { type: 'START_JOB'; jobId: string }
  | { type: 'JOB_UPDATE'; job: Job }
  | { type: 'JOB_LOST'; error: string }
  | { type: 'ANALYSIS'; analysis: Analysis }
  | { type: 'TYPE_VALUE'; path: string; value: number }
  | { type: 'TOGGLE_REJECT'; face: Face }
  | { type: 'SET_GEOMETRY'; patch: Partial<GeometrySettings> }
  | { type: 'MODEL'; model: ModelResult | null; spec?: Spec | null; requestId?: string }
  | { type: 'CHAT'; messages: ChatMessage[]; last?: ChatResponse | null }
  | { type: 'GOTO'; screen: Screen }
  | { type: 'RESET' };

export const MAX_ITEMS = 6;

export const initialAi: AiSettings = {
  // Qwen-Image, its rescue, TripoSR and Solaria send the images to hosted services: off until the user opts in.
  use_reader: true, use_qwen_image: false, use_rescue: false, use_triposr: false, use_solaria: false,
  seed: 7, randomize_seed: false, attempts: 2,
};

export const initialGeometry: GeometrySettings = {
  snap: true, clearance: 'medium', finish: 'none', finish_mm: 1.0, finish_edges: 'all_vertical',
};

export const initialState: State = {
  screen: 'capture', mode: 'photos', projection: 'auto', items: [], reference: '', ai: initialAi, jobId: null, jobItems: [], jobError: null,
  job: null, analysis: null, typed: {}, rejected: [], geometry: initialGeometry, model: null, modelSpec: null,
  chat: { messages: [], last: null },
};

let seq = 0;
/** Build a CaptureItem (with an object URL for previews) to pass to ADD_FILES. */
export function toCaptureItem(file: File, face: Face | 'auto' = 'auto', kind: CaptureKind = 'auto'): CaptureItem {
  seq += 1;
  return { id: `img-${Date.now().toString(36)}-${seq}`, file, url: URL.createObjectURL(file), face, kind };
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'SET_MODE':
      // Switching mode changes what one upload means (a face vs. the whole sheet); start the tray over.
      return { ...state, mode: action.mode, items: [] };
    case 'SET_PROJECTION':
      return { ...state, projection: action.projection };
    case 'ADD_FILES':
      return { ...state, items: [...state.items, ...action.items].slice(0, MAX_ITEMS) };
    case 'SET_ITEM':
      return { ...state, items: state.items.map((it) => (it.id === action.id ? { ...it, ...action.patch } : it)) };
    case 'REMOVE_ITEM':
      return { ...state, items: state.items.filter((it) => it.id !== action.id) };
    case 'SET_REFERENCE':
      return { ...state, reference: action.reference };
    case 'SET_AI':
      return { ...state, ai: { ...state.ai, ...action.patch } };
    case 'START_JOB':
      return {
        ...state, screen: 'analyzing', jobId: action.jobId, jobItems: state.items.map(({ url, face, kind }) => ({ url, face, kind })),
        jobError: null, job: null, analysis: null, typed: {}, rejected: [], model: null, modelSpec: null,
      };
    case 'JOB_UPDATE':
      return action.job.job_id === state.jobId ? { ...state, job: action.job } : state;
    case 'JOB_LOST':
      return { ...state, jobError: action.error };
    case 'ANALYSIS':
      // A part described in the chat has no job: it replaces whatever the photos produced.
      if (action.analysis.request_id === '') {
        return { ...state, analysis: action.analysis, jobId: null, job: null, jobItems: [], jobError: null, typed: {}, rejected: [] };
      }
      // A late response for an older job (or a merge that outlived its job) must not replace the current one.
      return action.analysis.request_id === state.jobId ? { ...state, analysis: action.analysis } : state;
    case 'TYPE_VALUE':
      return { ...state, typed: { ...state.typed, [action.path]: action.value } };
    case 'TOGGLE_REJECT':
      return {
        ...state,
        rejected: state.rejected.includes(action.face) ? state.rejected.filter((f) => f !== action.face) : [...state.rejected, action.face],
        // features are numbered per fuse and a reject can renumber them: a typed features[k] value could land on
        // another feature, so only the typed sizes survive the change
        typed: Object.fromEntries(Object.entries(state.typed).filter(([path]) => !path.startsWith('features['))),
      };
    case 'SET_GEOMETRY':
      return { ...state, geometry: { ...state.geometry, ...action.patch } };
    case 'MODEL':
      // a build that finishes after the user moved to another analysis must not take over
      if (action.requestId !== undefined && action.requestId !== (state.analysis?.request_id ?? '')) return state;
      return { ...state, model: action.model, modelSpec: action.model ? action.spec ?? null : null };
    case 'CHAT':
      return { ...state, chat: { messages: action.messages, last: action.last === undefined ? state.chat.last : action.last } };
    case 'GOTO':
      return { ...state, screen: action.screen };
    case 'RESET':
      return { ...initialState, ai: state.ai, geometry: state.geometry };
    default:
      return state;
  }
}

interface StoreValue { state: State; dispatch: Dispatch<Action> }
const StoreContext = createContext<StoreValue | null>(null);

export function StoreProvider({ children, initial }: { children: ReactNode; initial?: Partial<State> }) {
  const [state, rawDispatch] = useReducer(reducer, { ...initialState, ...initial });
  const stateRef = useRef(state);
  stateRef.current = state;
  const dispatch = useCallback<Dispatch<Action>>((action) => {
    // Side effects live here, not in the reducer (StrictMode runs reducers twice).
    if (action.type === 'START_JOB') resetDeadline();
    if (action.type === 'REMOVE_ITEM') {
      const it = stateRef.current.items.find((i) => i.id === action.id);
      // Keep the URL alive while the current job still shows it (crops, Analyzing panes).
      if (it && !stateRef.current.jobItems.some((j) => j.url === it.url)) URL.revokeObjectURL(it.url);
    }
    rawDispatch(action);
  }, []);
  const value = useMemo(() => ({ state, dispatch }), [state, dispatch]);
  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore(): StoreValue {
  const v = useContext(StoreContext);
  if (!v) throw new Error('useStore must be used inside StoreProvider');
  return v;
}
