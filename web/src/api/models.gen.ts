// Generated from the Pydantic models by scripts/gen_ts_models.py. Do not edit; after changing
// s2c/multiview/spec.py or settings.py run: uv run python scripts/gen_ts_models.py

/** The AI helpers. */
export interface AiSettings {
  use_reader: boolean;
  use_qwen_image: boolean;
  use_rescue: boolean;
  use_triposr: boolean;
  use_solaria: boolean;
  seed: number;
  randomize_seed: boolean;
  attempts: number;
}

export interface Chamfer {
  type: 'chamfer';
  edges: 'all' | 'all_vertical' | 'top' | 'bottom';
  radius_mm: number;
}

export interface Envelope {
  x_mm: number;
  y_mm: number;
  z_mm: number;
}

export interface ExportSettings {
  formats: string[];
}

/** A round pin standing on the face: within height_mm of the envelope face only its cylinder is kept. */
export interface FaceBoss {
  type: 'boss';
  face: 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
  a_mm: number;
  b_mm: number;
  diameter_mm: number;
  height_mm: number;
}

export interface FaceHole {
  type: 'hole';
  face: 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
  a_mm: number;
  b_mm: number;
  diameter_mm: number;
  depth_mm: number | null;
}

/** An axis-aligned rectangular cut from the envelope face inward, centred at (a, b): a pocket, a step or, running off the outline, a corner notch. */
export interface FacePocket {
  type: 'pocket';
  face: 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
  a_mm: number;
  b_mm: number;
  width_mm: number;
  height_mm: number;
  depth_mm: number | null;
}

export interface FaceSlot {
  type: 'slot';
  face: 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom';
  a_mm: number;
  b_mm: number;
  width_mm: number;
  length_mm: number;
  angle_deg: number;
  depth_mm: number | null;
}

export interface Fillet {
  type: 'fillet';
  edges: 'all' | 'all_vertical' | 'top' | 'bottom';
  radius_mm: number;
}

export interface GeometrySettings {
  snap: boolean;
  clearance: 'fine' | 'medium' | 'coarse';
  finish: 'none' | 'fillet' | 'chamfer';
  finish_mm: number;
  finish_edges: 'all_vertical' | 'top' | 'bottom' | 'all';
}

export interface MeshSettings {
  quality: 'draft' | 'normal' | 'fine';
}

export interface MultiViewSpec {
  version: 'mv1';
  envelope: Envelope;
  views: Views;
  features: (FaceHole | FaceSlot | FacePocket | FaceBoss)[];
  finishes: (Fillet | Chamfer)[];
  provenance: Record<string, 'user_written' | 'measured' | 'user_edited' | 'scaled' | 'inferred' | 'estimated' | 'default'>;
  snapped: string[];
  warnings: string[];
  confidence: number;
}

export interface MvAbstain {
  stage: 'label' | 'outline' | 'dimensions' | 'complete' | 'build' | 'slice' | 'verify';
  reason: string;
  remedy: string;
  partial: Record<string, unknown> | null;
}

export interface Outline {
  outer: [number, number][];
  inner: [number, number][][];
  source: 'observed' | 'mirrored' | 'inferred' | 'assumed';
  confidence: number;
}

export interface PrintSettings {
  material: 'PLA' | 'PETG' | 'ABS' | 'ASA' | 'TPU';
  nozzle_mm: number;
  layer_mm: number;
  infill_pct: number;
  infill_pattern: 'grid' | 'gyroid' | 'rectilinear' | 'honeycomb' | 'cubic' | 'lightning';
  perimeters: number;
  supports: 'off' | 'buildplate' | 'everywhere';
  brim_mm: number;
  scale_pct: number;
}

export interface Views {
  front: Outline;
  top: Outline;
  right: Outline;
}
