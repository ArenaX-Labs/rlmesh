using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using RLMesh;
using RLMesh.Internal;
using RLMesh.Interop;
using RLMesh.Spaces;

sealed class Counter : IEnv
{
    internal readonly Tensor tensor = new(new byte[16], new long[] { 4 }, DType.Float32);
    internal readonly Value observation;
    internal readonly Metadata info = new();
    private long steps;
    internal Counter() { observation = tensor; }
    public Space ObservationSpace { get; } = new Box(-1f, 1f, new long[] { 4 });
    public Space ActionSpace { get; } = new Discrete(2);
    public ResetResult Reset(long? seed, Metadata? options) { steps = 0; return new(observation, info); }
    public StepResult Step(Value? action) { steps++; return new(observation, 0, false, false, info); }
}

static unsafe class Program
{
    static void Checkpoint() { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); GC.WaitForPendingFinalizers(); }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void Steps(RLMeshBinding* binding, RLMeshValue* action, int count)
    {
        for (int i = 0; i < count; i++)
        {
            RLMeshValue* observation = null; RLMeshValue* info = null;
            double reward = 0; bool terminated = false, truncated = false;
            try
            {
                Status.Check(Native.rlmesh_binding_step(binding, action, &observation, &reward, &terminated, &truncated, &info));
                // Same ownership adoption as a public Value return. Keep it live
                // through the iteration, then let its SafeHandle finalize.
                Value result = new(observation); observation = null;
                GC.KeepAlive(result);
            }
            finally { Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info); }
        }
    }
    static void Main(string[] args)
    {
        var authored = new Counter();
        using var env = Env.For(authored);
        using var call = env.Record.Acquire(env);
        Value action = 0;
        using var input = action.Handle.Acquire(action);
        var options = new RLMeshBindOptions { struct_size = (UIntPtr)sizeof(RLMeshBindOptions), execution_horizon = 1 };
        RLMeshBinding* binding = null;
        Status.Check(Native.rlmesh_bind(new RLMeshModelTarget { kind = RLMeshModelTargetKind.RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE }, Native.rlmesh_env_target(call.Env), &options, &binding));
        try
        {
            RLMeshValue* observation = null; RLMeshValue* info = null;
            Status.Check(Native.rlmesh_binding_reset(binding, null, null, &observation, &info));
            Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info);
            Steps(binding, input.Value, 1000);
            Checkpoint();
            if (args.Length != 0 && args[0] == "--timing")
            {
                // One long-lived JIT process. The parent alternates this batch
                // with the C++ process and sends the next request on stdin.
                Console.WriteLine("ready");
                while (Console.ReadLine() != null)
                {
                    var timer = Stopwatch.StartNew(); Steps(binding, input.Value, 10000); timer.Stop();
                    Console.WriteLine(timer.Elapsed.TotalNanoseconds / 10000);
                }
                return;
            }
            long handles = Interlocked.Read(ref HandleMetrics.FinalizableConstructions);
            long bytes = GC.GetTotalAllocatedBytes(precise: true);
            Steps(binding, input.Value, 10000);
            Checkpoint();
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - bytes;
            long constructed = Interlocked.Read(ref HandleMetrics.FinalizableConstructions) - handles;
            long live = Interlocked.Read(ref HandleMetrics.LiveHandles);
            Steps(binding, input.Value, 100000);
            Checkpoint();
            long after = Interlocked.Read(ref HandleMetrics.LiveHandles);
            Console.WriteLine($"C# gate: {constructed / 10000.0:F3} finalizable SafeHandles/step; {allocated / 10000.0:F3} B/step; live handles {live} -> {after} after 100000 steps");
            if (constructed > 20000 || allocated > 5120000 || after != live)
                throw new Exception("E-3 allocation or no-leak budget failed");
        }
        finally { Native.rlmesh_binding_free(binding); authored.tensor.Dispose(); }
        GC.KeepAlive(authored); GC.KeepAlive(action);
    }
}
