// When Review may build: a merge due or in flight holds Build, a failed one offers Retry, an invalid value holds it.

export interface GateInput {
  requestId: string | null; // null: a part described in the chat, with no analysis on the server to merge with
  pending: boolean;         // typed values or rejects newer than what the analysis reflects
  merging: boolean;
  mergeErr: string | null;
  building: boolean;
  invalid: boolean;         // a typed value that cannot be sent
}

export function reviewGate(g: GateInput): { updating: boolean; retry: boolean; blocked: boolean } {
  const updating = !!g.requestId && (g.merging || g.pending);
  const retry = updating && !g.merging && !!g.mergeErr;
  return { updating, retry, blocked: g.building || (updating && !retry) || g.invalid };
}
