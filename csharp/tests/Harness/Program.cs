using System;
using RLMesh;
using RLMesh.Spaces;

sealed class Counter : IEnv, IClose
{
    private int steps;
    private readonly Value observation = 0;
    private readonly Metadata info = new();
    public Space ObservationSpace { get; } = new Discrete(100);
    public Space ActionSpace { get; } = new Discrete(2);
    public ResetResult Reset(long? seed, Metadata? options) { steps = 0; return new(observation, info); }
    public StepResult Step(Value? action) => new(observation, 1, ++steps == 2, false, info);
    public void Close() { }
}

static class Program
{
    static void Main()
    {
        using (var server = new EnvServer(new Counter()))
        {
            server.Start();
            Console.WriteLine(server.Address);
            Console.Out.Flush();
            Console.ReadLine();
            server.Shutdown();
        }
        if (!RLMesh.Ext.Lifetime.Shutdown(TimeSpan.FromSeconds(10)))
            throw new Exception(string.Join("; ", RLMesh.Ext.Lifetime.Pending()));
    }
}
