using System;
using System.IO;

namespace RLMesh;

/// <summary>Experimental. An invalid argument.</summary>
public class ValueError : ArgumentException
{
    public ValueError(string message) : base(message) { }
    public ValueError(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. A value of the wrong type.</summary>
public class TypeError : InvalidCastException
{
    public TypeError(string message) : base(message) { }
    public TypeError(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. An operation refused by the runtime.</summary>
public class RuntimeError : InvalidOperationException
{
    public RuntimeError(string message) : base(message) { }
    public RuntimeError(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. A failure reported by RLMesh.</summary>
public class RLMeshException : RuntimeError
{
    public RLMeshException(string message) : base(message) { }
    public RLMeshException(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. An invalid protocol exchange.</summary>
public class ProtocolException : RLMeshException
{
    public ProtocolException(string message) : base(message) { }
    public ProtocolException(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. An authored environment failure.</summary>
public class EnvironmentException : RLMeshException
{
    public bool IsRecoverable => this is RecoverableEnvironmentException;
    public EnvironmentException(string message) : base("[INTERNAL] " + message) { RawMessage = message; }
    public EnvironmentException(string message, Exception? innerException) : base("[INTERNAL] " + message, innerException) { RawMessage = message; }
    internal string RawMessage { get; }
    internal EnvironmentException(string formatted, string raw) : base(formatted) { RawMessage = raw; }
}
/// <summary>Experimental. An environment failure permitting another reset.</summary>
public class RecoverableEnvironmentException : EnvironmentException
{
    internal RecoverableEnvironmentException(string formatted, string raw) : base(formatted, raw) { }
    public RecoverableEnvironmentException(string message) : base(message) { }
    public RecoverableEnvironmentException(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. An operating system failure.</summary>
public class OSError : IOException
{
    public OSError(string message) : base(message) { }
    public OSError(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. A transport connection failure.</summary>
public class ConnectionError : OSError
{
    public ConnectionError(string message) : base(message) { }
    public ConnectionError(string message, Exception? innerException) : base(message, innerException) { }
}
/// <summary>Experimental. An operation exceeded its deadline.</summary>
public class TimeoutError : OSError
{
    public TimeoutError(string message) : base(message) { }
    public TimeoutError(string message, Exception? innerException) : base(message, innerException) { }
}
