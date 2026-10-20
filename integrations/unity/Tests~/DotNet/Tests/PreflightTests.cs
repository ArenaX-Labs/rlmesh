using System;
using System.Collections.Generic;
using System.Threading;
using System.Diagnostics;
using Xunit;

namespace RLMesh.Unity.Tests;

public sealed class PreflightTests
{
    [Fact]
    public void QuitIsBoundedEvenWhenShutdownCannotFinish()
    {
        using var release = new ManualResetEventSlim();
        using var finished = new ManualResetEventSlim();
        int main = Thread.CurrentThread.ManagedThreadId;
        var messages = new List<string>();
        var clock = Stopwatch.StartNew();
        try
        {
            Assert.False(QuitPreflight.TryShutdown(TimeSpan.FromMilliseconds(50), _ =>
            {
                Assert.NotEqual(main, Thread.CurrentThread.ManagedThreadId);
                try { release.Wait(); return false; }
                finally { finished.Set(); }
            }, () => new[] { "stuck callback" }, messages.Add));
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1));
            Assert.Contains("stuck callback", Assert.Single(messages));
        }
        finally { release.Set(); Assert.True(finished.Wait(TimeSpan.FromSeconds(5))); }
    }

    [Fact]
    public void ForceUnlockIsExplicitAndDoesNotPretendTheBarrierCompleted()
    {
        int unlocks = 0;
        var messages = new List<string>();
        var preflight = new ShutdownPreflight(_ => false, () => new[] { "still unsafe" }, () => { }, () => unlocks++, messages.Add);
        preflight.Hold(); preflight.ForceUnlock(); preflight.ForceUnlock();
        Assert.False(preflight.Completed); Assert.False(preflight.IsLocked); Assert.Equal(1, unlocks);
        preflight.DiagnoseUnload();
        Assert.Contains("unsafe unload", Assert.Single(messages));
    }

    [Fact]
    public void FailedBarrierRetainsLockReportsPendingAndCanRetry()
    {
        int main = Thread.CurrentThread.ManagedThreadId;
        int locks = 0, unlocks = 0;
        bool succeeds = false;
        var messages = new List<string>();
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var preflight = new ShutdownPreflight(_ =>
        {
            Assert.NotEqual(main, Thread.CurrentThread.ManagedThreadId);
            entered.Set(); Assert.True(release.Wait(TimeSpan.FromSeconds(5)));
            return succeeds;
        }, () => new[] { "wedged env step" }, () => locks++, () => { Assert.Equal(main, Thread.CurrentThread.ManagedThreadId); unlocks++; }, messages.Add);
        preflight.Hold(); preflight.Hold();
        preflight.Request(TimeSpan.FromSeconds(1));
        Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
        preflight.Poll(); Assert.True(preflight.IsRunning); Assert.True(preflight.IsLocked);
        preflight.Request(TimeSpan.FromSeconds(1)); // coalesces, never stacks attempts
        release.Set();
        PollUntilDone(preflight);
        Assert.False(preflight.Completed); Assert.True(preflight.IsLocked);
        Assert.Equal(1, locks); Assert.Equal(0, unlocks);
        Assert.Contains("wedged env step", Assert.Single(messages));
        preflight.DiagnoseUnload();
        Assert.Equal(0, unlocks);
        succeeds = true;
        preflight.Request(TimeSpan.FromSeconds(1)); PollUntilDone(preflight);
        Assert.True(preflight.Completed); Assert.False(preflight.IsLocked); Assert.Equal(1, unlocks);
        Assert.Throws<InvalidOperationException>(preflight.Hold);
    }

    [Fact]
    public void ThrowingBarrierKeepsLock()
    {
        string message = null;
        var preflight = new ShutdownPreflight(_ => throw new Exception("failed"), () => new[] { "still live" }, () => { }, () => throw new Exception("must not unlock"), text => message = text);
        preflight.Request(TimeSpan.Zero); PollUntilDone(preflight);
        Assert.True(preflight.IsLocked); Assert.Contains("Shutdown threw", message); Assert.Contains("still live", message);
    }

    private static void PollUntilDone(ShutdownPreflight preflight)
    {
        Assert.True(SpinWait.SpinUntil(() => { preflight.Poll(); return !preflight.IsRunning; }, TimeSpan.FromSeconds(5)));
    }
}
