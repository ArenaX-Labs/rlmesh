namespace RLMesh.Ext;

/// <summary>Experimental. Binding-specific server options.</summary>
public sealed class EnvServerOptions
{
    /// <summary>Replaces detected C# host runtime JSON. Null selects the detected runtime.</summary>
    public string? HostRuntimeJson { get; set; }
}
