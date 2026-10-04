// web/src/api/types.ts — the frontend's API types. The spec and the settings come from models.gen.ts,
// generated from the Pydantic models (scripts/gen_ts_models.py); the rest are the routes' own answers.
import type { AiSettings, FaceBoss, FaceHole, FacePocket, FaceSlot, GeometrySettings, MultiViewSpec, Outline, PrintSettings } from './models.gen';

export type { AiSettings, GeometrySettings, Outline, PrintSettings };
export type Spec = MultiViewSpec;
export type Hole = FaceHole;
export type Slot = FaceSlot;
export type Pocket = FacePocket;
export type Boss = FaceBoss;
export type Face = FaceHole['face'];
export type Provenance = Spec['provenance'][string];
export type StageKey = 'label' | 'outline' | 'read' | 'draw' | 'fuse' | 'views' | 'lines' | 'values';
export type StageState = 'pending' | 'running' | 'done' | 'skipped' | 'failed';
export type FilledBy = 'observed' | 'qwen-image' | 'triposr' | 'mirrored' | 'assumed' | 'inferred';

export interface Status { providers: { vision: boolean; reader: boolean; qwen_image: boolean; triposr: boolean; solaria: boolean; slicer: boolean; blender: boolean }; ttl_s: number }
export interface Example { name: string; url: string; face: Face; kind: 'sketch' | 'photo' | 'drawing' }

export interface Stage { key: StageKey; state: StageState; tool: string; ai: boolean; detail: string; started: number | null; ended: number | null }
export interface ReadValue { text: string; value_mm: number; kind: 'linear' | 'diameter' | 'radius'; bbox: [number, number, number, number]; confidence: number }
export interface JobImage { index: number; width: number; height: number; face: Face | 'unknown' | null; kind: 'sketch' | 'photo' | 'drawing' | null;
  outline: [number, number][] | null; circles: { cx: number; cy: number; d: number }[]; reads: ReadValue[] }
export interface Job { job_id: string; /** "sheet" also when one untagged photo turned out to hold every view. */ mode?: 'photos' | 'sheet'; status: 'running' | 'done' | 'failed' | 'cancelled'; stages: Stage[]; images: JobImage[];
  coverage: Record<Face, FilledBy | 'empty'>; result: Analysis | null; error: string | null }

export interface Abstain { stage: string; reason: string; remedy: string; partial: Record<string, number> | null;
  missing?: string[]; suggested?: Record<string, number>; partial_provenance?: Record<string, Provenance> }
export interface Analysis { request_id: string; spec: Spec | null; abstain: Abstain | null; filled_by: Partial<Record<Face, FilledBy>> }

export interface ModelResult { key: string | null; glb_url: string | null; volume_cm3: number | null; bbox_mm: [number, number, number] | null;
  iou: Partial<Record<Face, number>>; iou_mean: number | null; views: Partial<Record<Face, string>>; warnings: string[]; abstain: Abstain | null }
export interface ExportFile { url: string; name: string; size_bytes: number }
export interface ExportResult { files: Record<string, ExportFile>; zip_url: string | null; print_time_s: number | null; filament_g: number | null; warnings: string[]; abstain: Abstain | null }

export type ChatRole = 'user' | 'assistant';
/** `sig`: the server's signature on its own reply, sent back with the conversation. */
export interface ChatMessage { role: ChatRole; content: string; sig?: string }
export type PartType = 'plate' | 'l_bracket' | 'spacer' | 'flange';
export interface PartRequest { type: PartType; values: Record<string, number>; holes: { a_mm: number; b_mm: number; diameter_mm: number }[] }
export interface ChatResponse { reply: string; sig: string; options: string[]; part: PartRequest | null; missing: string[]; spec: Spec | null; model: string }
