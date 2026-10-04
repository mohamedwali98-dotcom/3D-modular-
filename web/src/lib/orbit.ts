// The 3D viewer's camera distance: the wheel and a two-finger pinch zoom within the same limits.

/** The camera distance after scaling it by `factor` (< 1 closer), kept between 1.4 and 6.4 part sizes. */
export function zoomed(radius: number, partSize: number, factor: number): number {
  return Math.max(partSize * 1.4, Math.min(partSize * 6.4, radius * factor));
}
