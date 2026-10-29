using System;
using System.Threading;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh.Tests;

// Test-only raw C client. The public foundation intentionally has no run or
// RemoteEnv API; this exercises the C ABI's cancellation inputs directly. A run
// dials the address; the remote env connects on first use, holding the env's
// lease from then on.
internal sealed unsafe class NativeClient : IDisposable
{
    private readonly byte[] address;
    private RLMeshRemoteEnv* remote;
    internal NativeClient(string address) { this.address = Utf8.Encode(address, true); }
    private RLMeshRemoteEnv* Remote()
    {
        if (remote != null) return remote;
        fixed (byte* text = address)
        {
            var options = new RLMeshRemoteEnvOptions { struct_size = (UIntPtr)sizeof(RLMeshRemoteEnvOptions), address = (sbyte*)text };
            RLMeshRemoteEnv* output = null;
            Status.Check(Native.rlmesh_remote_env_new(&options, &output));
            remote = output;
        }
        return remote;
    }
    internal void Run(CancellationToken cancellationToken = default, ulong episodes = 1)
    {
        CancellationBridge.Run(cancellationToken, token =>
        {
            var options = new RLMeshRunOptions
            {
                struct_size = (UIntPtr)sizeof(RLMeshRunOptions), cancel = token,
                has_episodes = 1, episodes = episodes, execution_horizon = 1,
            };
            RLMeshRunResult* result = null;
            try
            {
                fixed (byte* text = address)
                    Status.Check(Native.rlmesh_run(new RLMeshModelTarget { kind = RLMeshModelTargetKind.RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE },
                        Native.rlmesh_address_target((sbyte*)text), &options, &result), cancellationToken);
                return 0;
            }
            finally { Native.rlmesh_run_result_free(result); }
        });
    }
    public void Dispose() { if (remote != null) Native.rlmesh_remote_env_free(remote); }
    internal void Step(CancellationToken cancellationToken)
    {
        CancellationBridge.Run(cancellationToken, token =>
        {
            RLMeshValue* action = null; RLMeshValue* observation = null; RLMeshValue* info = null;
            double reward = 0; bool terminated = false, truncated = false;
            try
            {
                Status.Check(Native.rlmesh_value_int(0, &action));
                Status.Check(Native.rlmesh_ext_remote_env_step(Remote(), action, token,
                    &observation, &reward, &terminated, &truncated, &info), cancellationToken);
                return 0;
            }
            finally { Native.rlmesh_value_free(action); Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info); }
        });
    }
    internal bool RawResetFails()
    {
        RLMeshValue* observation = null; RLMeshValue* info = null;
        try
        {
            RLMeshRemoteEnv* client;
            try { client = Remote(); }
            catch (Exception) { return true; }
            return Native.rlmesh_remote_env_reset(client, null, null, &observation, &info) != RLMeshStatus.RLMESH_OK;
        }
        finally { Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info); }
    }
}
