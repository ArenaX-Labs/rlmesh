using System;
using System.IO;
using UnityEngine;
using RLMesh.Unity.Samples;

namespace RLMesh.Unity.Verification
{
    public sealed class SmokeBootstrap : MonoBehaviour
    {
        private RLMeshReachTargetBehaviour env;
        private string completionFile;
        private int episodes;
        private float started;
        private bool finishRequested;
        private void Awake()
        {
            completionFile = Argument("--rlmesh-complete-file", "");
            episodes = int.Parse(Argument("--rlmesh-episodes", "3"));
            started = Time.realtimeSinceStartup;
            env = gameObject.AddComponent<RLMeshReachTargetBehaviour>();
            env.Address = "tcp://127.0.0.1:0";
            env.EnvId = "Unity-ReachTarget-v0";
        }
        private void Update()
        {
            if (finishRequested) return;
            if (Time.realtimeSinceStartup - started > 120)
            {
                Debug.LogError("RLMesh player timed out waiting for the Python client");
                finishRequested = true;
                Application.Quit(1);
                return;
            }
            if (completionFile.Length > 0 && File.Exists(completionFile))
            {
                if (File.ReadAllText(completionFile).Trim() != episodes.ToString()) throw new Exception("wrong episode count");
                Debug.Log("Unity IL2CPP smoke OK: " + episodes + " episodes");
                finishRequested = true;
                Application.Quit(0); // wantsToQuit runs the asynchronous terminal preflight.
            }
        }
        private static string Argument(string key, string fallback)
        {
            string[] args = System.Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, key);
            return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
        }
    }
}
