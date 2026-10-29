import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoundedQueue } from '../src/queue.js';

test('bounded queue drops the oldest and counts it', async () => {
  const q = new BoundedQueue<number>(4);
  for (let i = 0; i < 10; i++) q.push(i);
  assert.equal(q.dropped, 6);
  assert.equal(q.queued, 4);
  const got: number[] = [];
  for (let i = 0; i < 4; i++) got.push((await q.next()).value as number);
  assert.deepEqual(got, [6, 7, 8, 9]);
});

test('a waiting consumer gets the next item directly; close ends iteration', async () => {
  let closed = 0;
  const q = new BoundedQueue<string>(2, () => closed++);
  const p = q.next();
  q.push('a');
  assert.deepEqual(await p, { value: 'a', done: false });
  const it = (async () => {
    const seen: string[] = [];
    for await (const v of q) seen.push(v);
    return seen;
  })();
  q.push('b');
  await new Promise((r) => setTimeout(r, 1));
  q.close();
  assert.deepEqual(await it, ['b']);
  assert.equal(closed, 1);
  q.push('ignored');
  assert.equal(q.queued, 0);
});

test('maxQueue must be positive', () => {
  assert.throws(() => new BoundedQueue(0));
});
