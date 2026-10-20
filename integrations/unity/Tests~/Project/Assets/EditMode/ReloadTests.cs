using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using NUnit.Framework;
using RLMesh.Ext;
using RLMesh.Spaces;
using RLMesh.Unity.Editor;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace RLMesh.Unity.Verification
{
    public sealed class ReloadTests
    {
        // SessionState survives a domain reload; this value deliberately does not.
        private static readonly string Domain = Guid.NewGuid().ToString();

        [UnityTest]
        public IEnumerator ServeReloadServeAgain()
        {
            SessionState.SetString("RLMesh.safe.oldDomain", Domain);
            var dispatcher = UnityEnvHost.BeginServe();
            var env = new TestEnv();
            var server = new EnvServer(new MainThreadEnv(dispatcher, env));
            server.Start();
            Assert.IsTrue(ReloadPreflight.State.IsLocked);
            using (var client = StartClient(server.Address))
            {
                yield return WaitFor(() => client.HasExited);
                Assert.AreEqual(0, client.ExitCode);
                Assert.IsTrue(env.Entered.IsSet);
                Assert.AreEqual(Thread.CurrentThread.ManagedThreadId, env.StepThread);
            }
            ReloadPreflight.RequestShutdown(TimeSpan.FromSeconds(5));
            yield return WaitFor(() => !ReloadPreflight.State.IsRunning);
            Assert.IsTrue(ReloadPreflight.State.Completed);
            Assert.IsFalse(ReloadPreflight.State.IsLocked);
            dispatcher.Dispose(); server.Dispose();
            EditorUtility.RequestScriptReload();
            yield return new WaitForDomainReload();
            Assert.AreNotEqual(SessionState.GetString("RLMesh.safe.oldDomain", ""), Domain);
            Assert.AreEqual(LifetimeState.Open, Lifetime.State);
            // A fresh domain can bind and serve again, without a stale delegate.
            dispatcher = UnityEnvHost.BeginServe();
            env = new TestEnv();
            server = new EnvServer(new MainThreadEnv(dispatcher, env));
            server.Start();
            Assert.IsNotEmpty(server.Address);
            using (var client = StartClient(server.Address))
            {
                yield return WaitFor(() => client.HasExited);
                Assert.AreEqual(0, client.ExitCode);
                Assert.IsTrue(env.Entered.IsSet);
                Assert.AreEqual(Thread.CurrentThread.ManagedThreadId, env.StepThread);
            }
            ReloadPreflight.RequestShutdown(TimeSpan.FromSeconds(5));
            yield return WaitFor(() => !ReloadPreflight.State.IsRunning);
            Assert.IsTrue(ReloadPreflight.State.Completed);
            dispatcher.Dispose(); server.Dispose();
            EditorUtility.RequestScriptReload();
            yield return new WaitForDomainReload();
            UnityEngine.Debug.Log("Unity Editor reload OK: served before and after domain reload");
        }

        [UnityTest]
        public IEnumerator FailedBarrierDefersReloadAndRetryReleasesIt()
        {
            // An artificial native-thread wedge keeps the Editor responsive.
            // Wedging authored code on the MAIN thread itself freezes any Editor.
            var env = new TestEnv { Wedge = true };
            ReloadPreflight.Hold();
            var server = new EnvServer(env); server.Start();
            using (var client = StartClient(server.Address))
            {
                try
                {
                    yield return WaitFor(() => env.Entered.IsSet);
                    SessionState.SetString("RLMesh.failed.oldDomain", Domain);
                    LogAssert.Expect(LogType.Error, new Regex("RLMesh: Shutdown returned false.*Pending:"));
                    ReloadPreflight.RequestShutdown(TimeSpan.FromMilliseconds(100));
                    yield return WaitFor(() => !ReloadPreflight.State.IsRunning);
                    Assert.AreEqual(LifetimeState.Incomplete, Lifetime.State);
                    Assert.IsTrue(ReloadPreflight.State.IsLocked);
                    Assert.IsNotEmpty(Lifetime.Pending());
                    EditorUtility.RequestScriptReload();
                    // Exercise the REAL Unity lock, not only the coordinator flag.
                    for (int tick = 0; tick < 30; tick++)
                    {
                        Assert.AreEqual(SessionState.GetString("RLMesh.failed.oldDomain", ""), Domain);
                        Assert.IsTrue(ReloadPreflight.State.IsLocked);
                        yield return null;
                    }
                    Assert.IsTrue(env.Entered.IsSet); // Editor kept ticking, no abort/crash.
                }
                finally { env.Release.Set(); }
                yield return WaitFor(() => client.HasExited);
            }
            ReloadPreflight.RequestShutdown(TimeSpan.FromSeconds(5));
            // The pending reload proceeds ONLY when the true result unlocks.
            yield return new WaitForDomainReload();
            Assert.AreNotEqual(SessionState.GetString("RLMesh.failed.oldDomain", ""), Domain);
            Assert.AreEqual(LifetimeState.Open, Lifetime.State);
            UnityEngine.Debug.Log("Unity failed-barrier reload OK: deferred, responsive, released, reloaded");
        }

        private static IEnumerator WaitFor(Func<bool> predicate)
        {
            double deadline = EditorApplication.timeSinceStartup + 15;
            while (!predicate())
            {
                Assert.Less(EditorApplication.timeSinceStartup, deadline, "Unity verification timed out");
                yield return null;
            }
        }

        private static Process StartClient(string address)
        {
            string project = Directory.GetParent(Application.dataPath).FullName;
            var start = new ProcessStartInfo(Path.Combine(project, "Assets/Tools/step-client"), address)
            { UseShellExecute = false, WorkingDirectory = project };
            start.EnvironmentVariables["LD_LIBRARY_PATH"] = Path.Combine(project, "Packages/com.rlmesh.rlmesh/Runtime/Plugins/x86_64/linux");
            return Process.Start(start);
        }

        private sealed class TestEnv : IEnv
        {
            public bool Wedge;
            public int StepThread;
            public readonly ManualResetEventSlim Entered = new ManualResetEventSlim();
            public readonly ManualResetEventSlim Release = new ManualResetEventSlim();
            public Space ObservationSpace => new Discrete(2);
            public Space ActionSpace => new Discrete(3);
            public ResetResult Reset(long? seed, Metadata options) => new ResetResult(0, new Metadata());
            public StepResult Step(Value action)
            {
                Entered.Set();
                StepThread = Thread.CurrentThread.ManagedThreadId;
                if (Wedge) Release.Wait();
                return new StepResult(1, -1, true, false, new Metadata());
            }
        }
    }
}
