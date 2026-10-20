using System;
using System.Runtime.CompilerServices;
using RLMesh.Internal;

namespace RLMesh.Interop;

internal static unsafe class NativeGate
{
    internal const string Library = Native.LibraryName;
    private static readonly Lazy<RuntimeError?> Failure = new(Validate);

    private static RuntimeError? Validate()
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
        catch (RuntimeError e) { return e; }
        catch (Exception e) when (IsLoadFailure(e)) { return LoadError(e, "ABI gate"); }
        return null;
    }

    internal static void Ensure() { var failure = Failure.Value; if (failure != null) throw failure; }
    internal static bool IsLoadFailure(Exception e) => e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException;
    internal static RuntimeError LoadError(Exception e, string symbol) => new($"{Library} ({symbol}): {e.Message}", e);
    internal static T Call<T>(Func<T> call, [CallerMemberName] string symbol = "")
    {
        using var operation = Admission.Enter(symbol);
        return Continue(call, symbol);
    }
    // Already-admitted workers and teardown must preserve load diagnostics
    // without asking the terminal gate to admit the operation a second time.
    internal static T Continue<T>(Func<T> call, [CallerMemberName] string symbol = "")
    {
        Ensure();
        using var nativeReturn = new NativeReturnScope();
        try { return call(); }
        catch (Exception e) when (IsLoadFailure(e)) { throw LoadError(e, symbol); }
    }
    internal static void Continue(Action call, [CallerMemberName] string symbol = "") => Continue(() => { call(); return 0; }, symbol);
    internal static void Call(Action call, [CallerMemberName] string symbol = "") => Call(() => { call(); return 0; }, symbol);
}
