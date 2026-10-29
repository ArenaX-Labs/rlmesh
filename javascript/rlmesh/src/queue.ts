// A bounded async queue: the producer never blocks; when full, the oldest item is dropped and counted.

export interface Subscription<T> extends AsyncIterableIterator<T>, AsyncDisposable {
  /** Items dropped because the consumer fell more than `maxQueue` behind. */
  readonly dropped: number;
  /** Items waiting. */
  readonly queued: number;
  readonly closed: boolean;
  close(): void;
}

export class BoundedQueue<T> implements Subscription<T> {
  private items: T[] = [];
  private head = 0;
  private waiter: ((r: IteratorResult<T>) => void) | null = null;
  private _dropped = 0;
  private _closed = false;

  constructor(
    readonly maxQueue: number,
    private onClose: () => void = () => {},
  ) {
    if (!(maxQueue >= 1)) throw new RangeError('maxQueue must be >= 1');
  }

  get dropped() {
    return this._dropped;
  }
  get queued() {
    return this.items.length - this.head;
  }
  get closed() {
    return this._closed;
  }

  push(v: T) {
    if (this._closed) return;
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      w({ value: v, done: false });
      return;
    }
    this.items.push(v);
    if (this.queued > this.maxQueue) {
      this.head++;
      this._dropped++;
    }
    // Compact occasionally so `head` does not grow without bound.
    if (this.head > 1024 && this.head * 2 > this.items.length) {
      this.items = this.items.slice(this.head);
      this.head = 0;
    }
  }

  next(): Promise<IteratorResult<T>> {
    if (this.queued > 0) {
      const v = this.items[this.head++] as T;
      return Promise.resolve({ value: v, done: false });
    }
    if (this._closed) return Promise.resolve({ value: undefined, done: true });
    return new Promise((r) => (this.waiter = r));
  }

  return(): Promise<IteratorResult<T>> {
    this.close();
    return Promise.resolve({ value: undefined, done: true });
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    this.items = [];
    this.head = 0;
    this.onClose();
    const w = this.waiter;
    this.waiter = null;
    w?.({ value: undefined, done: true });
  }

  [Symbol.asyncIterator]() {
    return this;
  }

  async [Symbol.asyncDispose]() {
    this.close();
  }
}
