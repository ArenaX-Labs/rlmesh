using System;
using System.Collections.Generic;
using System.Threading;
using RLMesh.Internal;

namespace RLMesh.Ext;

/// <summary>Experimental. The terminal shutdown barrier's state.</summary>
public enum LifetimeState { Open, ShuttingDown, Completed, Incomplete }

/// <summary>Experimental. Drain native callbacks before unloading the managed runtime.</summary>
public static class Lifetime
{
    public static bool Shutdown(TimeSpan timeout) => LifetimeEngine.Shutdown(timeout);
    public static IReadOnlyList<string> Pending() => LifetimeEngine.Pending();
    public static CancellationToken ShuttingDown => LifetimeEngine.Notification;
    public static LifetimeState State => LifetimeEngine.State;
    public static bool WaitFreed(Env env, TimeSpan timeout)
    {
        if (env == null) throw new ArgumentNullException(nameof(env));
        return env.WaitFreed(timeout);
    }
    public static bool WaitFreed(EnvServer server, TimeSpan timeout)
    {
        if (server == null) throw new ArgumentNullException(nameof(server));
        return server.WaitFreed(timeout);
    }
}
