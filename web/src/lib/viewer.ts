// The part Model shows: never one built from another spec once its rebuild has stopped (failed or busy server),
// since its sizes, volume and preview would contradict what Export builds.

export function viewerModel<M>(model: M | null, modelSpec: unknown, spec: unknown, building: boolean): M | null {
  return model && (modelSpec === spec || building) ? model : null;
}
