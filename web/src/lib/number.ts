// The numbers typed on Review: one parser for every keyboard, and the rule each field's value must meet.

const MAX_MM = 10000; // the server's limit for any size (s2c/multiview/spec.py MAX_MM)
const NUMBER = /^-?(\d+\.?\d*|\.\d+)$/;
const SIZE = /(^envelope\.[xyz]_mm|\.(diameter|width|height|length|depth)_mm)$/;

/** A typed number in millimetres, or null. A decimal comma (a French keypad) and Arabic-Indic or Persian digits
 * are read; hex, exponents and trailing letters are refused instead of half-read. */
export function parseMm(raw: string): number | null {
  const s = raw.trim()
    .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x0660))
    .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x06f0))
    .replace(/[,٫]/g, '.');
  return NUMBER.test(s) ? Number(s) : null;
}

/** Why a typed value cannot be sent, or null. A size is above 0 and at most 10 000 mm; a position is any number. */
export function fieldProblem(path: string, raw: string): string | null {
  const value = parseMm(raw);
  if (value === null) return 'Use a number like 12.5';
  if (SIZE.test(path) && !(value > 0 && value <= MAX_MM)) return 'Use a size above 0 and up to 10 000 mm';
  return null;
}

/** The drafts left when the rejected faces change: features may renumber, so their drafts go; sizes stay. */
export function dropFeatureDrafts(drafts: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(drafts).filter(([path]) => !path.startsWith('features[')));
}

/** The typed values that cannot be sent, by path, among those still on screen: a draft for a feature the user
 * removed (or that a merge no longer lists) never holds Build with nothing left to fix. */
export function shownProblems(drafts: Record<string, string>, shown: Set<string>): Record<string, string> {
  return Object.fromEntries(Object.entries(drafts).flatMap(([path, raw]) => {
    const why = shown.has(path) && raw.trim() !== '' ? fieldProblem(path, raw) : null;
    return why ? [[path, why]] : [];
  }));
}
