import { describe, expect, it } from 'vitest';
import type { ChatMessage } from '../api/types';
import { lastTurns } from './turns';

const u = (content: string): ChatMessage => ({ role: 'user', content });
const a = (content: string): ChatMessage => ({ role: 'assistant', content, sig: 'f'.repeat(64) });

describe('chat window', () => {
  it('keeps the last turns, starting at a user turn so every reply keeps the turn it answered', () => {
    const convo = [u('1'), a('r1'), u('2'), a('r2'), u('3')];
    expect(lastTurns(convo, 4)).toEqual([u('2'), a('r2'), u('3')]);
    expect(lastTurns(convo, 5)).toEqual(convo);
  });
});
