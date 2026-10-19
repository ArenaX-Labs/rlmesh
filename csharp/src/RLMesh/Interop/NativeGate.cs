using System;
using System.Runtime.CompilerServices;

namespace RLMesh.Interop;

internal static unsafe class NativeGate
{
    internal const string Library = Native.LibraryName;
    private static readonly RuntimeError? Failure;

    static NativeGate()
    {
        try
        {
            if (IntPtr.Size != 8) throw new RuntimeError("rlmesh_capi requires a 64-bit process");
            uint actual = Native.rlmesh_abi_version();
            if (actual != Initializers.AbiVersion)
                throw new RuntimeError($"rlmesh_capi ABI generation {actual}; expected {Initializers.AbiVersion}");
            var info = Initializers.BuildInfo();
            var status = Native.rlmesh_build_info(&info);
            if (status != RLMeshStatus.RLMESH_OK)
                throw new RuntimeError("rlmesh_capi build info: " + Utf8.Borrowed(Native.rlmesh_last_error_message()));
            string version = Utf8.Borrowed(info.version);
            if (version != Initializers.PackageVersion)
                throw new RuntimeError($"rlmesh_capi build version {version}; expected {Initializers.PackageVersion}");
        }
        catch (RuntimeError e) { Failure = e; }
        catch (Exception e) when (IsLoadFailure(e)) { Failure = LoadError(e, "ABI gate"); }
    }

    internal static void Ensure() { if (Failure != null) throw Failure; }
    internal static bool IsLoadFailure(Exception e) => e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException;
    internal static RuntimeError LoadError(Exception e, string symbol) => new($"{Library} ({symbol}): {e.Message}", e);
    internal static T Call<T>(Func<T> call, [CallerMemberName] string symbol = "")
    {
        Ensure();
        try { return call(); }
        catch (Exception e) when (IsLoadFailure(e)) { throw LoadError(e, symbol); }
    }
    internal static void Call(Action call, [CallerMemberName] string symbol = "") => Call(() => { call(); return 0; }, symbol);
}
