import type { ChatMessage } from '../api/types';

/** The last `n` turns of a conversation, starting at a user turn: the server checks each reply's signature
 * against the user turn just before it, so a window must not open on a reply. */
export function lastTurns(messages: ChatMessage[], n: number): ChatMessage[] {
  const window = messages.slice(-n);
  const first = window.findIndex((m) => m.role === 'user');
  return first < 0 ? [] : window.slice(first);
}
