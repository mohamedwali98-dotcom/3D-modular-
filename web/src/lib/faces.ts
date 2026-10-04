// What the Review screen says about where each face's outline came from (the server's `filled_by`).

export interface FaceInfo { badge: string; verb: string }

const INFO: Record<string, FaceInfo> = {
  observed: { badge: 'Observed', verb: 'Observed in your photo' },
  'qwen-image': { badge: 'AI-drawn (Qwen-Image)', verb: 'Drawn by Qwen-Image' },
  triposr: { badge: '3D fallback (TripoSR)', verb: 'Estimated by TripoSR' },
  mirrored: { badge: 'Mirrored', verb: 'Mirrored from the opposite face' },
  assumed: { badge: 'Assumed', verb: 'Assumed, no view of this face' },
  inferred: { badge: 'Turned', verb: 'Follows from your other views: the part is round' },
};

const UNKNOWN: FaceInfo = { badge: 'Not observed', verb: 'Not seen in your images' };

/** Never undefined: a source the server adds later still gets a badge instead of a blank page. */
export function faceInfo(by: string): FaceInfo {
  return INFO[by] ?? UNKNOWN;
}

/** Rejecting changes the part only for a face an AI helper drew; any other face would be rebuilt the same way. */
export function canReject(by: string): boolean {
  return by === 'qwen-image' || by === 'triposr';
}

/** The face card's button: Undo on a rejected face (the server refills it, as an assumed rectangle for one), Reject on
 * a face an AI drew, none otherwise. */
export function rejectAction(by: string, rejected: boolean): 'reject' | 'undo' | null {
  if (rejected) return 'undo';
  return canReject(by) ? 'reject' : null;
}
