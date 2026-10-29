// The C# gate program of python/rlmesh/tests/conformance/native_gate.py: the
// Gymnasium gate's echo env on a v2 session. C#'s v0.2 surface serves envs
// only (D-45), so every other verb is unsupported.
using System;
using RLMesh;

static class Program
{
    static int Main(string[] args) => Gate.Run(args, (env, address) =>
    {
        using (var server = new EnvServer(env, new() { Address = address }))
        {
            server.Start();
            Console.WriteLine($"listening on {server.Address}");
            Console.Out.Flush();
            Console.In.ReadToEnd();
            server.Shutdown();
        }
        if (!RLMesh.Ext.Lifetime.Shutdown(TimeSpan.FromSeconds(10)))
            throw new Exception(string.Join("; ", RLMesh.Ext.Lifetime.Pending()));
    });
}
