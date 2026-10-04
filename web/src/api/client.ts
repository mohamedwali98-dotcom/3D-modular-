import type {
  AiSettings, Analysis, ChatMessage, ChatResponse, Example, ExportResult, Face, GeometrySettings, Job, ModelResult, PrintSettings, Spec, Status,
} from './types';
import { askToken, storedToken, withToken } from '../lib/auth';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export const API_BASE = '/api';

const FALLBACK: Record<number, string> = {
  0: 'Could not reach the server. Check that it is running.',
  400: 'The request was not accepted.',
  404: 'Unknown or expired analysis. Analyze again.',
  413: 'That file is too large (10 MB max).',
  415: 'That file is not an image.',
};

async function request<T>(path: string, init?: RequestInit, asked = false): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, withToken(init, storedToken()));
  } catch {
    throw new ApiError(0, FALLBACK[0]);
  }
  let body: unknown = null;
  const text = await res.text();
  if (text) {
    try { body = JSON.parse(text); } catch { body = null; }
  }
  if (!res.ok) {
    const err = body && typeof body === 'object' && typeof (body as { error?: unknown }).error === 'string'
      ? (body as { error: string }).error
      : FALLBACK[res.status] ?? `Something went wrong (${res.status}). Try again.`;
    if (res.status === 401 && !asked) {  // the server wants its access token: ask (once for all), then retry
      const reason = body && typeof body === 'object' ? (body as { reason?: string }).reason : undefined;
      if (await askToken(err, undefined, reason)) return request<T>(path, init, true);
    }
    throw new ApiError(res.status, err);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function getStatus(): Promise<Status> {
  return request<Status>('/status');
}

export function getExamples(): Promise<Example[]> {
  return request<Example[]>('/examples');
}

/** Uploads the images and returns the new job id. `mode: 'sheet'` sends one sheet image in place
 * of one photo per face; `faces`/`kinds`/`reference` are ignored server-side in that mode. */
export async function startAnalysis(
  files: File[], faces: string[], kinds: string[], reference: string, ai?: Partial<AiSettings>,
  mode: 'photos' | 'sheet' = 'photos', projection: 'auto' | 'first' | 'third' = 'auto',
): Promise<string> {
  const fd = new FormData();
  for (const f of files) fd.append('files', f, f.name);
  fd.append('faces', JSON.stringify(faces));
  fd.append('kinds', JSON.stringify(kinds));
  fd.append('reference', reference);
  fd.append('mode', mode);
  fd.append('projection', projection);
  if (ai) fd.append('ai', JSON.stringify(ai));
  const r = await request<{ job_id: string }>('/analyze', { method: 'POST', body: fd });
  return r.job_id;
}

/** One poll; a request that hangs past `timeoutMs` fails like a dropped connection instead of freezing the screen. */
export function getJob(id: string, timeoutMs = 10_000): Promise<Job> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  return request<Job>(`/jobs/${encodeURIComponent(id)}`, { signal: ctl.signal }).finally(() => clearTimeout(timer));
}

export async function cancelJob(id: string): Promise<void> {
  await request<{ ok: boolean }>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export function merge(body: { request_id: string; user_values: Record<string, number>; accepted: Face[]; rejected: Face[] }): Promise<Analysis> {
  return request<Analysis>('/merge', json(body));
}

export function buildModel(body: { request_id: string | null; spec: Spec; geometry: GeometrySettings }): Promise<ModelResult> {
  return request<ModelResult>('/model', json(body));
}

export interface ExportSettings {
  geometry: GeometrySettings;
  mesh: { quality: 'draft' | 'normal' | 'fine' };
  printing: PrintSettings;
  export: { formats: string[] };
}

export function exportFiles(body: { spec: Spec; settings: ExportSettings }): Promise<ExportResult> {
  return request<ExportResult>('/export', json(body));
}

export function chat(messages: ChatMessage[]): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', json({ messages }));
}
