// How often Analyzing asks for its job: quickly while the server answers, backing off while the network is down,
// and giving up only after two minutes without an answer (a phone moving from Wi-Fi to 4G, a lift ride).

export const POLL_MS = 400;
export const MAX_DELAY_MS = 5000;
export const GIVE_UP_MS = 120_000;

/** The wait before the next poll after `fails` failures in a row, failing for `failingForMs`; null: stop. */
export function nextPoll(fails: number, failingForMs: number): number | null {
  if (fails === 0) return POLL_MS;
  if (failingForMs >= GIVE_UP_MS) return null;
  return Math.min(MAX_DELAY_MS, POLL_MS * 2 ** fails);
}
