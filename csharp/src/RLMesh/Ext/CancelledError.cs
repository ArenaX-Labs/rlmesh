using System;
using System.Threading;

namespace RLMesh.Ext;

/// <summary>Experimental. An asynchronous operation's wait was cancelled.</summary>
public class CancelledError : OperationCanceledException
{
    public CancelledError(string message) : base(message) { }
    public CancelledError(string message, Exception? innerException) : base(message, innerException) { }
    internal CancelledError(CancellationToken token) : base("RLMesh operation cancelled", null, token) { }
}
