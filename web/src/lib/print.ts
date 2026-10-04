// Print settings the slicer accepts: a layer is at most three quarters of the nozzle (s2c/multiview/settings.py).

const SLIDER_TOP = 0.32; // mm, the Layer slider's own top

/** The tallest layer for a nozzle, to 0.01 mm. */
export function layerMax(nozzle: string): number {
  return Math.min(SLIDER_TOP, Math.floor(0.75 * Number(nozzle) * 100 + 1e-9) / 100);
}

/** The settings with a new nozzle, the layer lowered to fit it. */
export function withNozzle<T extends { nozzle: string; layer: number }>(print: T, nozzle: string): T {
  return { ...print, nozzle, layer: Math.min(print.layer, layerMax(nozzle)) };
}
