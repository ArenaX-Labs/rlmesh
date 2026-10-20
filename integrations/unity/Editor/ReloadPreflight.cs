using System;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEngine;

namespace RLMesh.Unity.Editor
{
    [InitializeOnLoad]
    public static class ReloadPreflight
    {
        private const string ResumePlayModeKey = "RLMesh.resumePlayMode";
        private const string LockHeldKey = "RLMesh.reloadLockHeld";
        public static ShutdownPreflight State { get; private set; }
        private static bool requestReload;

        static ReloadPreflight()
        {
            UnityEnvHost.InitializeMainThread();
            if (SessionState.GetBool(LockHeldKey, false))
            {
                Debug.LogWarning("RLMesh recovered a reload lock surviving an unsafe domain unload; releasing its orphaned lock count.");
                ReleaseLock();
            }
            State = new ShutdownPreflight(UnityEnvHost.Shutdown, UnityEnvHost.Pending,
                AcquireLock, ReleaseLock, Debug.LogError);
            UnityEnvHost.Serving += Hold;
            EditorApplication.update += Update;
            CompilationPipeline.compilationStarted += CompilationStarted;
            EditorApplication.playModeStateChanged += PlayModeChanged;
            AssemblyReloadEvents.beforeAssemblyReload += BeforeReload;
            EditorApplication.wantsToQuit += WantsToQuit;
            Application.quitting += OnQuitting;
            int resume = SessionState.GetInt(ResumePlayModeKey, -1);
            if (resume >= 0)
            {
                SessionState.EraseInt(ResumePlayModeKey);
                EditorApplication.delayCall += () => EditorApplication.isPlaying = resume == 1;
            }
        }

        public static void Hold() { State.Hold(); }
        public static void RequestShutdown(TimeSpan timeout) { State.Request(timeout); }

        private static void AcquireLock()
        {
            EditorApplication.LockReloadAssemblies();
            SessionState.SetBool(LockHeldKey, true);
        }
        private static void ReleaseLock()
        {
            EditorApplication.UnlockReloadAssemblies();
            SessionState.EraseBool(LockHeldKey);
        }

        [MenuItem("RLMesh/Force unlock reload")]
        public static void ForceUnlockReload()
        {
            if (!State.IsLocked) return;
            if (EditorUtility.DisplayDialog("Unsafe RLMesh reload unlock",
                "Native callbacks may still be running. Reloading now can crash the Editor. Release blocked code and retry normal shutdown instead whenever possible.",
                "Force unlock anyway", "Cancel"))
            {
                Debug.LogWarning("RLMesh reload lock forcibly released without a completed shutdown barrier. Pending: " + string.Join("; ", UnityEnvHost.Pending()));
                State.ForceUnlock();
            }
        }

        [MenuItem("RLMesh/Shutdown and Reload")]
        public static void ShutdownAndReload()
        {
            requestReload = true;
            RequestShutdown(TimeSpan.FromSeconds(5));
        }

        private static void CompilationStarted(object context)
        {
            if (State.IsLocked) RequestShutdown(TimeSpan.FromSeconds(5));
        }

        private static void PlayModeChanged(PlayModeStateChange change)
        {
            if (change != PlayModeStateChange.ExitingEditMode) return;
            if (EditorApplication.enterPlayModeOptionsEnabled &&
                (EditorApplication.enterPlayModeOptions & EnterPlayModeOptions.DisableDomainReload) != 0) return;
            if (!State.IsLocked) return;
            // Play Mode transitions can unload even while the assembly reload
            // lock is held on some Unity versions. Defer the transition itself.
            SessionState.SetInt(ResumePlayModeKey, 1);
            EditorApplication.isPlaying = false;
            ShutdownAndReload();
        }

        private static bool WantsToQuit()
        {
            if (!State.IsLocked || State.Completed) return true;
            bool safe = QuitPreflight.TryShutdown(TimeSpan.FromSeconds(5), UnityEnvHost.Shutdown, UnityEnvHost.Pending, Debug.LogWarning);
            return safe || Application.isBatchMode || EditorUtility.DisplayDialog("RLMesh shutdown incomplete",
                "Native work is still pending (see Console). Quit anyway, or cancel and release the blocked environment?",
                "Quit anyway", "Cancel");
        }

        private static void Update()
        {
            UnityEnvHost.Pump(); // Editor pumps during compilation and while Play Mode is stopped.
            State.Poll();
            if (!State.Completed) return;
            if (requestReload) { requestReload = false; EditorUtility.RequestScriptReload(); }
        }

        private static void BeforeReload()
        {
            // Notifications cannot veto an unload. They are diagnostics only;
            // all authorization to unlock comes from the earlier preflight.
            State.DiagnoseUnload();
            if (State.NeedsUnloadPreflight)
                QuitPreflight.TryShutdown(TimeSpan.FromSeconds(1), UnityEnvHost.Shutdown, UnityEnvHost.Pending, Debug.LogError);
        }

        private static void OnQuitting()
        {
            if (RLMesh.Ext.Lifetime.State != RLMesh.Ext.LifetimeState.Completed)
                Debug.LogWarning("RLMesh process exiting. Pending: " + string.Join("; ", UnityEnvHost.Pending()));
        }
    }
}
