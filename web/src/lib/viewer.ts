// The part Model shows: never one built from another spec or other shape settings once its rebuild has stopped
// (failed or busy server), since its sizes, volume and preview would contradict what Export builds.

function sameSettings(a: object | undefined, b: object | undefined): boolean {
  if (a === undefined || b === undefined) return true; // not known (a fresh screen): no reason to doubt the part
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => (a as Record<string, unknown>)[k] === (b as Record<string, unknown>)[k]);
}

export function viewerModel<M>(model: M | null, modelSpec: unknown, spec: unknown, building: boolean,
  builtGeometry?: object, geometry?: object): M | null {
  const fresh = modelSpec === spec && sameSettings(builtGeometry, geometry);
  return model && (fresh || building) ? model : null;
}
