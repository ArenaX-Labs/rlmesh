using System;
using UnityEngine;

namespace RLMesh.Unity
{
    // A persistent pump survives destruction/disable of the served component.
    public sealed class PlayerShutdown : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void InitializeMainThread() { UnityEnvHost.InitializeMainThread(); }
#if !UNITY_EDITOR
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Install()
        {
            var host = new GameObject("RLMesh shutdown pump");
            DontDestroyOnLoad(host);
            host.AddComponent<PlayerShutdown>();
        }
        private void Awake()
        {
            Application.wantsToQuit += WantsToQuit;
            Application.quitting += OnQuitting;
        }
        private bool WantsToQuit()
        {
            QuitPreflight.TryShutdown(TimeSpan.FromSeconds(5), UnityEnvHost.Shutdown, UnityEnvHost.Pending, Debug.LogWarning);
            return true; // Accept the original Quit(code), including failure codes.
        }
        private void Update() { UnityEnvHost.Pump(); }
        private void OnQuitting()
        {
            if (RLMesh.Ext.Lifetime.State != RLMesh.Ext.LifetimeState.Completed)
                Debug.LogWarning("RLMesh process exiting with pending work: " + string.Join("; ", UnityEnvHost.Pending()));
        }
        private void OnDestroy()
        {
            Application.wantsToQuit -= WantsToQuit;
            Application.quitting -= OnQuitting;
        }
#endif
    }
}
