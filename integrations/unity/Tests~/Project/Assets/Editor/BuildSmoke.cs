using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RLMesh.Unity.Verification
{
    public static class BuildSmoke
    {
        public static void Build()
        {
            // A generated minimal scene avoids stale serialized component GUIDs.
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("RLMesh smoke").AddComponent<SmokeBootstrap>();
            const string scenePath = "Assets/Smoke.unity";
            EditorSceneManager.SaveScene(scene, scenePath);
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Standalone, ScriptingImplementation.IL2CPP);
            PlayerSettings.SetApiCompatibilityLevel(NamedBuildTarget.Standalone, ApiCompatibilityLevel.NET_Standard);
            PlayerSettings.SetManagedStrippingLevel(NamedBuildTarget.Standalone, ManagedStrippingLevel.High);
            string[] args = System.Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, "-rlmeshBuildPath");
            string output = index >= 0 ? args[index + 1] : "build/unity/Smoke.x86_64";
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("IL2CPP smoke build failed: " + report.summary.result);
        }
    }
}
