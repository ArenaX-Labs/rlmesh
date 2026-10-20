using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace RLMesh.Internal;

// The low bit is admission; the remaining bits count operations. Closing and
// admission contend on the same CAS, including constructors not yet published.
internal static class Admission
{
    private static long word = 1;
    private static long sequence;
    [ThreadStatic] private static int depth;
    [ThreadStatic] private static Operation? outermost;
    internal static readonly ConcurrentDictionary<long, string> Operations = new();
    internal static long Count => (long)((ulong)Volatile.Read(ref word) >> 1);
    internal static bool Open => (Volatile.Read(ref word) & 1) != 0;
    internal static void Close()
    {
        long before;
        do { before = Volatile.Read(ref word); }
        while (Interlocked.CompareExchange(ref word, before & ~1L, before) != before);
    }
    internal static Operation Enter(string name, bool asynchronous = false)
    {
        LifetimeEngine.EnsureInitialized();
        while (true)
        {
            long before = Volatile.Read(ref word);
            if ((before & 1) == 0 && depth == 0 && CallbackContext.Depth == 0)
                throw new ObjectDisposedException("RLMesh", "the lifetime is shutting down");
            if (Interlocked.CompareExchange(ref word, before + 2, before) != before) continue;
            long id = Interlocked.Increment(ref sequence);
            try
            {
                var operation = new Operation(id, !asynchronous);
                Operations.TryAdd(id, name);
                if (!asynchronous) { if (depth == 0) outermost = operation; depth++; }
                return operation;
            }
            catch { Operations.TryRemove(id, out _); Interlocked.Add(ref word, -2); throw; }
        }
    }
    internal static void Publish(NativeHandle record)
    {
        // Delay publication until the enclosing constructor has adopted all
        // outputs and dependency references. The counted operation covers this
        // interval, so teardown cannot miss an unpublished native root.
        if (outermost != null) outermost.AddPublication(record);
        else record.Publish();
    }
    internal sealed class Operation : IDisposable
    {
        private readonly long id;
        private readonly bool synchronous;
        private int disposed;
        private List<NativeHandle>? publications;
        internal Operation(long id, bool synchronous) { this.id = id; this.synchronous = synchronous; }
        internal void AddPublication(NativeHandle record) => (publications ??= new()).Add(record);
        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0) return;
            try
            {
                if (publications != null) foreach (var record in publications) record.Publish();
            }
            finally
            {
                if (synchronous) { depth--; if (depth == 0) outermost = null; }
                Operations.TryRemove(id, out _);
                Interlocked.Add(ref word, -2);
            }
        }
    }
}
