using System;

namespace RLMesh.Interop;

// Object-like macros and inline helpers are deliberately outside generated code.
internal static unsafe partial class Initializers
{
    internal const uint AbiVersion = 6;
    internal static RLMeshBuildInfo BuildInfo() => new() { struct_size = (UIntPtr)sizeof(RLMeshBuildInfo) };
    internal static RLMeshBoxOptions BoxOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshBoxOptions), dtype = RLMeshDType.RLMESH_DTYPE_FLOAT32 };
    internal static RLMeshDiscreteOptions DiscreteOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshDiscreteOptions), dtype = RLMeshDType.RLMESH_DTYPE_INT64 };
    internal static RLMeshServeOptions ServeOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshServeOptions) };
    // The C macro's host-specific string is replaced by ENV-38's managed host.
    // The caller pins its UTF-8 bytes for the duration of server construction.
    internal static RLMeshExtEnvServerOptions ExtEnvServerOptions(sbyte* hostRuntimeJson) => new() { struct_size = (UIntPtr)sizeof(RLMeshExtEnvServerOptions), host_runtime_json = hostRuntimeJson };
    internal static RLMeshEnvServerOptions EnvServerOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshEnvServerOptions), close_env_on_shutdown = 1 };
    internal static RLMeshEnvFactoryServeOptions EnvFactoryServeOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshEnvFactoryServeOptions), num_envs = 1 };
    internal static RLMeshExtEnvFactoryServeOptions ExtEnvFactoryServeOptions(sbyte* hostRuntimeJson) => new() { struct_size = (UIntPtr)sizeof(RLMeshExtEnvFactoryServeOptions), host_runtime_json = hostRuntimeJson };
    internal static RLMeshExtModelServeOptions ExtModelServeOptions() => new() { struct_size = (UIntPtr)sizeof(RLMeshExtModelServeOptions) };
    internal static RLMeshEnvAttributes EnvAttributes() => new() { struct_size = (UIntPtr)sizeof(RLMeshEnvAttributes) };
    internal static RLMeshEnvVtable EnvVtable() => new() { struct_size = (UIntPtr)sizeof(RLMeshEnvVtable) };
}

internal static partial class Native
{
    internal const string LibraryName = "rlmesh_capi";
}
