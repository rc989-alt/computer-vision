// src/toolRunner.ts
import { Mutex } from 'async-mutex';

const convoLocks = new Map<string, Mutex>();
function getLock(conversationId: string) {
  if (!convoLocks.has(conversationId)) convoLocks.set(conversationId, new Mutex());
  return convoLocks.get(conversationId)!;
}

// Simple in-memory idempotency cache for this process
const resultCache = new Map<string, unknown>();

export async function runToolSerialized<T>(
  conversationId: string,
  idempotencyKey: string,
  callTool: () => Promise<T>,
  isTransientError: (e: any) => boolean = (e) =>
    e?.status === 400 || e?.status === 409 || e?.status === 429
): Promise<T> {
  const lock = getLock(conversationId);
  return lock.runExclusive(async () => {
    const cacheKey = `${conversationId}:${idempotencyKey}`;
    if (resultCache.has(cacheKey)) return resultCache.get(cacheKey) as T;

    let lastErr: any;
    for (let i = 0; i < 3; i++) {
      try {
        const res = await callTool();
        resultCache.set(cacheKey, res);
        return res;
      } catch (e: any) {
        lastErr = e;
        if (!isTransientError(e)) throw e;
        await new Promise(r => setTimeout(r, 200 * 2 ** i)); // 200ms, 400ms, 800ms
      }
    }
    throw lastErr;
  });
}
