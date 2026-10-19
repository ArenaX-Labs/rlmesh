using System;
using RLMesh.Spaces;

namespace RLMesh;

/// <summary>Experimental. An authored environment. Served calls run on native library threads.</summary>
public interface IEnv
{
    Space ObservationSpace { get; }
    Space ActionSpace { get; }
    ResetResult Reset(long? seed, Metadata? options);
    StepResult Step(Value? action);
}
/// <summary>Experimental. Optional environment close, invoked at most once by the server.</summary>
public interface IClose { void Close(); }

/// <summary>Experimental. Observation and information returned by reset.</summary>
public readonly struct ResetResult
{
    public ResetResult(Value observation, Metadata info)
    {
        Observation = observation ?? throw new ArgumentNullException(nameof(observation));
        Info = info ?? throw new ArgumentNullException(nameof(info));
    }
    public Value Observation { get; }
    public Metadata Info { get; }
    public void Deconstruct(out Value observation, out Metadata info) { observation = Observation; info = Info; }
}
/// <summary>Experimental. Observation, reward, termination flags and information returned by step.</summary>
public readonly struct StepResult
{
    public StepResult(Value observation, double reward, bool terminated, bool truncated, Metadata info)
    {
        Observation = observation ?? throw new ArgumentNullException(nameof(observation));
        Reward = reward; Terminated = terminated; Truncated = truncated;
        Info = info ?? throw new ArgumentNullException(nameof(info));
    }
    public Value Observation { get; }
    public double Reward { get; }
    public bool Terminated { get; }
    public bool Truncated { get; }
    public Metadata Info { get; }
    public void Deconstruct(out Value observation, out double reward, out bool terminated, out bool truncated, out Metadata info)
    { observation = Observation; reward = Reward; terminated = Terminated; truncated = Truncated; info = Info; }
}
