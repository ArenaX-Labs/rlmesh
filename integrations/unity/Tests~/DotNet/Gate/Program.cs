using System;
using System.Threading;
using System.Threading.Tasks;
using RLMesh;
using RLMesh.Unity;

static class Program
{
    static int Main(string[] args) => Gate.Run(args, (env, address) =>
    {
        UnityEnvHost.InitializeMainThread();
        var dispatcher = UnityEnvHost.BeginServe();
        var server = new EnvServer(new MainThreadEnv(dispatcher, env), new()
        {
            Address = address,
            Ext = new() { HostRuntimeJson = HostRuntime.Json("gate", false, System.Environment.Version.ToString()) },
        });
        server.Start();
        Console.WriteLine($"listening on {server.Address}");
        Console.Out.Flush();
        var eof = Task.Run(() => Console.In.ReadToEnd());
        while (!eof.IsCompleted)
        {
            UnityEnvHost.Pump();
            Thread.Sleep(1);
        }
        var stop = Task.Run(server.Dispose);
        while (!stop.IsCompleted) UnityEnvHost.Pump();
        stop.Wait();
        dispatcher.Dispose();
        if (!UnityEnvHost.Shutdown(TimeSpan.FromSeconds(10)))
            throw new Exception(string.Join("; ", UnityEnvHost.Pending()));
    });
}
