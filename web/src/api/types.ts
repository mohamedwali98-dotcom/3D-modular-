// web/src/api/types.ts — the single source of truth for the frontend
export type Face = 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
export type Provenance = 'user_written' | 'measured' | 'user_edited' | 'scaled' | 'inferred' | 'estimated' | 'default';
export type StageKey = 'label' | 'outline' | 'read' | 'draw' | 'fuse' | 'views' | 'lines' | 'values';
export type StageState = 'pending' | 'running' | 'done' | 'skipped' | 'failed';
export type FilledBy = 'observed' | 'qwen-image' | 'triposr' | 'mirrored' | 'assumed';

export interface Status { providers: { vision: boolean; reader: boolean; qwen_image: boolean; triposr: boolean; solaria: boolean; slicer: boolean; blender: boolean }; ttl_s: number }
export interface Example { name: string; url: string; face: Face; kind: 'sketch' | 'photo' | 'drawing' }

export interface Stage { key: StageKey; state: StageState; tool: string; ai: boolean; detail: string; started: number | null; ended: number | null }
export interface ReadValue { text: string; value_mm: number; kind: 'linear' | 'diameter' | 'radius'; bbox: [number, number, number, number]; confidence: number }
export interface JobImage { index: number; width: number; height: number; face: Face | 'unknown' | null; kind: 'sketch' | 'photo' | 'drawing' | null;
  outline: [number, number][] | null; circles: { cx: number; cy: number; d: number }[]; reads: ReadValue[] }
export interface Job { job_id: string; status: 'running' | 'done' | 'failed' | 'cancelled'; stages: Stage[]; images: JobImage[];
  coverage: Record<Face, FilledBy | 'empty'>; result: Analysis | null; error: string | null }

export interface Outline { outer: [number, number][]; inner: [number, number][][]; source: 'observed' | 'mirrored' | 'inferred' | 'assumed'; confidence: number }
export interface Hole { type: 'hole'; face: Face; a_mm: number; b_mm: number; diameter_mm: number; depth_mm?: number | null }
export interface Slot { type: 'slot'; face: Face; a_mm: number; b_mm: number; width_mm: number; length_mm: number; angle_deg: number; depth_mm?: number | null }
/** A rectangular cut from a face: a pocket, a step or a corner notch (null depth: through). */
export interface Pocket { type: 'pocket'; face: Face; a_mm: number; b_mm: number; width_mm: number; height_mm: number; depth_mm?: number | null }
/** A round pin standing on a face. */
export interface Boss { type: 'boss'; face: Face; a_mm: number; b_mm: number; diameter_mm: number; height_mm: number }
export interface Spec { version: 'mv1'; envelope: { x_mm: number; y_mm: number; z_mm: number }; views: { front: Outline; top: Outline; right: Outline };
  features: (Hole | Slot | Pocket | Boss)[]; finishes: { type: 'fillet' | 'chamfer'; edges: string; radius_mm: number }[];
  provenance: Record<string, Provenance>; snapped: string[]; warnings: string[]; confidence: number }
export interface Abstain { stage: string; reason: string; remedy: string; partial: Record<string, number> | null;
  missing?: string[]; suggested?: Record<string, number>; partial_provenance?: Record<string, Provenance> }
export interface Analysis { request_id: string; spec: Spec | null; abstain: Abstain | null; filled_by: Partial<Record<Face, FilledBy>> }

export interface AiSettings { use_reader: boolean; use_qwen_image: boolean; use_rescue: boolean; use_triposr: boolean; use_solaria: boolean; seed: number; randomize_seed: boolean; attempts: number }
export interface GeometrySettings { snap: boolean; clearance: 'fine' | 'medium' | 'coarse'; finish: 'none' | 'fillet' | 'chamfer'; finish_mm: number; finish_edges: 'all_vertical' | 'top' | 'bottom' | 'all' }
export interface PrintSettings { material: 'PLA' | 'PETG' | 'ABS' | 'ASA' | 'TPU'; nozzle_mm: number; layer_mm: number; infill_pct: number;
  infill_pattern: 'grid' | 'gyroid' | 'rectilinear' | 'honeycomb' | 'cubic' | 'lightning'; perimeters: number; supports: 'off' | 'buildplate' | 'everywhere'; brim_mm: number; scale_pct: number }
export interface ModelResult { key: string | null; glb_url: string | null; volume_cm3: number | null; bbox_mm: [number, number, number] | null;
  iou: Partial<Record<Face, number>>; iou_mean: number | null; views: Partial<Record<Face, string>>; warnings: string[]; abstain: Abstain | null }
export interface ExportFile { url: string; name: string; size_bytes: number }
export interface ExportResult { files: Record<string, ExportFile>; zip_url: string | null; print_time_s: number | null; filament_g: number | null; warnings: string[]; abstain: Abstain | null }

export type ChatRole = 'user' | 'assistant';
export interface ChatMessage { role: ChatRole; content: string }
export type PartType = 'plate' | 'l_bracket' | 'spacer' | 'flange';
export interface PartRequest { type: PartType; values: Record<string, number>; holes: { a_mm: number; b_mm: number; diameter_mm: number }[] }
export interface ChatResponse { reply: string; options: string[]; part: PartRequest | null; missing: string[]; spec: Spec | null; model: string }
