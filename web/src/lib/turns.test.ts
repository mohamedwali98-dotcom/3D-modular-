import { describe, expect, it } from 'vitest';
import type { ChatMessage } from '../api/types';
import { ApiError } from '../api/client';
import { chatErrorAction, lastTurns } from './turns';

const u = (content: string): ChatMessage => ({ role: 'user', content });
const a = (content: string): ChatMessage => ({ role: 'assistant', content, sig: 'f'.repeat(64) });

describe('chat window', () => {
  it('keeps the last turns, starting at a user turn so every reply keeps the turn it answered', () => {
    const convo = [u('1'), a('r1'), u('2'), a('r2'), u('3')];
    expect(lastTurns(convo, 4)).toEqual([u('2'), a('r2'), u('3')]);
    expect(lastTurns(convo, 5)).toEqual(convo);
  });
});

describe('chatErrorAction', () => {
  it('offers Start over when the server refused the conversation itself, Try again otherwise', () => {
    expect(chatErrorAction(new ApiError(400, 'The conversation could not be checked. Start a new one.'))).toBe('start_over');
    expect(chatErrorAction(new ApiError(0, 'Could not reach the server.'))).toBe('retry');
    expect(chatErrorAction(new ApiError(429, 'Too many requests.'))).toBe('retry');
    expect(chatErrorAction(new Error('boom'))).toBe('retry');
  });
});
