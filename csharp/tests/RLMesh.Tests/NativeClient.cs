using System;
using System.Threading;
using RLMesh.Internal;
using RLMesh.Interop;

namespace RLMesh.Tests;

// Test-only raw C client. The public foundation intentionally has no run or
// RemoteEnv API; this exercises the ABI 5 cancellation inputs directly.
internal sealed unsafe class NativeClient : IDisposable
{
    private readonly RLMeshRemoteEnv* remote;
    internal NativeClient(string address)
    {
        byte[] bytes = Utf8.Encode(address, true);
        fixed (byte* text = bytes)
        {
            var options = new RLMeshRemoteEnvOptions { struct_size = (UIntPtr)sizeof(RLMeshRemoteEnvOptions), address = (sbyte*)text };
            RLMeshRemoteEnv* output = null;
            Status.Check(Native.rlmesh_remote_env_new(&options, &output));
            remote = output;
        }
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
                Status.Check(Native.rlmesh_run(new RLMeshModelTarget { kind = RLMeshModelTargetKind.RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE },
                    Native.rlmesh_remote_env_target(remote), &options, &result), cancellationToken);
                return 0;
            }
            finally { Native.rlmesh_run_result_free(result); }
        });
    }
    public void Dispose() => Native.rlmesh_remote_env_free(remote);
    internal void Step(CancellationToken cancellationToken)
    {
        CancellationBridge.Run(cancellationToken, token =>
        {
            RLMeshValue* action = null; RLMeshValue* observation = null; RLMeshValue* info = null;
            double reward = 0; bool terminated = false, truncated = false;
            try
            {
                Status.Check(Native.rlmesh_value_int(0, &action));
                Status.Check(Native.rlmesh_ext_remote_env_step(remote, action, token,
                    &observation, &reward, &terminated, &truncated, &info), cancellationToken);
                return 0;
            }
            finally { Native.rlmesh_value_free(action); Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info); }
        });
    }
    internal bool RawResetFails()
    {
        RLMeshValue* observation = null; RLMeshValue* info = null;
        try { return Native.rlmesh_remote_env_reset(remote, null, null, &observation, &info) != RLMeshStatus.RLMESH_OK; }
        finally { Native.rlmesh_value_free(observation); Native.rlmesh_value_free(info); }
    }
}
