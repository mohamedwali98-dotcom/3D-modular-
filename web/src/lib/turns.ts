import type { ChatMessage } from '../api/types';

/** The last `n` turns of a conversation, starting at a user turn: the server checks each reply's signature
 * against the user turn just before it, so a window must not open on a reply. */
export function lastTurns(messages: ChatMessage[], n: number): ChatMessage[] {
  const window = messages.slice(-n);
  const first = window.findIndex((m) => m.role === 'user');
  return first < 0 ? [] : window.slice(first);
}

/** What the chat's error card offers: a 400 is the server refusing the conversation itself (it no longer checks
 * after a restart, or a message is too long), which sending it again cannot fix. */
export function chatErrorAction(e: unknown): 'retry' | 'start_over' {
  return typeof e === 'object' && e !== null && (e as { status?: number }).status === 400 ? 'start_over' : 'retry';
}
