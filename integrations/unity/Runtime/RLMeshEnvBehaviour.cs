using System;
using System.Threading.Tasks;
using RLMesh.Ext;
using RLMesh.Spaces;
using UnityEngine;

namespace RLMesh.Unity
{
    public abstract class RLMeshEnvBehaviour : MonoBehaviour, IEnv
    {
        public string Address = "tcp://127.0.0.1:5555";
        public string EnvId = "Unity-v0";
        private MainThreadDispatcher dispatcher;
        private EnvServer server;
        private string servedAddress;
        private bool started;
        private bool ownsBackground;
        private Task stopping = Task.CompletedTask;
        private bool restartRequested;
        private static int servingCount;
        private static bool previousRunInBackground;

        protected abstract Space ObservationSpace { get; }
        protected abstract Space ActionSpace { get; }
        protected abstract ResetResult ResetEnv(long? seed, Metadata options);
        protected abstract StepResult StepEnv(Value action);

        Space IEnv.ObservationSpace => ObservationSpace;
        Space IEnv.ActionSpace => ActionSpace;
        ResetResult IEnv.Reset(long? seed, Metadata options)
        {
            var result = ResetEnv(seed, options);
            result.Info["env_id"] = EnvId;
            return result;
        }
        StepResult IEnv.Step(Value action) => StepEnv(action);

        protected virtual void Start() { started = true; StartServing(); }
        protected virtual void OnEnable() { if (started) StartServing(); }

        public void StartServing()
        {
            UnityEnvHost.AssertMainThreadIfInitialized();
            if (server != null) return;
            if (!stopping.IsCompleted) { restartRequested = true; return; }
            restartRequested = false;
            if (Lifetime.State != LifetimeState.Open)
                throw new InvalidOperationException("RLMesh shutdown is terminal; reload the domain before serving again");
            dispatcher = UnityEnvHost.BeginServe();
            try
            {
#if ENABLE_IL2CPP && !UNITY_EDITOR
                const bool il2cpp = true;
                string backendVersion = Application.unityVersion; // IL2CPP ships with this Editor version.
#else
                const bool il2cpp = false;
                string backendVersion = System.Environment.Version.ToString();
#endif
                string hostRuntimeJson = HostRuntime.Json(Application.unityVersion, il2cpp, backendVersion);
                server = new EnvServer(new MainThreadEnv(dispatcher, this), new RLMesh.EnvServerOptions
                {
                    Address = Address,
                    Ext = new RLMesh.Ext.EnvServerOptions
                    {
                        HostRuntimeJson = hostRuntimeJson
                    }
                });
                servedAddress = server.Address;
                server.Start();
                if (servingCount++ == 0) previousRunInBackground = Application.runInBackground;
                ownsBackground = true;
                Application.runInBackground = true;
                Debug.Log("RLMesh host runtime: " + hostRuntimeJson, this);
                Debug.Log("Serving " + EnvId + " on " + servedAddress, this);
            }
            catch { StopServing(); throw; }
        }

        public string GetServedAddress() => servedAddress;
        protected virtual void Update()
        {
            if (restartRequested && stopping.IsCompleted) StartServing();
            dispatcher?.Pump();
            if (server == null || Lifetime.State != LifetimeState.Open) return;
            try
            {
                if (server.Wait(TimeSpan.Zero))
                {
                    Debug.LogWarning("RLMesh server stopped: " + EnvId, this);
                    StopServing();
                }
            }
            catch (Exception error)
            {
                if (Lifetime.State == LifetimeState.Open)
                    Debug.LogError("RLMesh serving failed for " + EnvId + ": " + error, this);
                StopServing();
            }
        }
        protected virtual void OnDisable() { StopServing(); }
        public void StopServing()
        {
            bool wasServing = ownsBackground;
            Task stop = UnityEnvHost.Stop(dispatcher, server);
            if (server != null) stopping = stop; // Repeated Stop must retain an earlier drain.
            restartRequested = false;
            dispatcher = null; server = null; servedAddress = null;
            ownsBackground = false;
            if (wasServing && --servingCount == 0) Application.runInBackground = previousRunInBackground;
        }
    }
}
