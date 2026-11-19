using System.IO;
using UnityEditor;
using UnityEngine;

namespace RLMesh.Unity.Editor
{
    /// <summary>
    /// Copies Samples~/ReachTarget into Assets/rlmesh/Demo/ReachTarget once when
    /// the package is present and that folder is missing.
    /// </summary>
    [InitializeOnLoad]
    static class ReachTargetSampleInstall
    {
        const string Dest = "Assets/rlmesh/Demo/ReachTarget";

        static ReachTargetSampleInstall()
        {
            EditorApplication.delayCall += EnsureInstalled;
        }

        static void EnsureInstalled()
        {
            if (File.Exists(Path.Combine(Dest, "Agent.cs")))
                return;

            string source = SampleSource();
            if (string.IsNullOrEmpty(source) || !Directory.Exists(source))
                return;

            Directory.CreateDirectory(Dest);
            foreach (string path in Directory.GetFiles(source))
                File.Copy(path, Path.Combine(Dest, Path.GetFileName(path)), overwrite: true);

            AssetDatabase.Refresh();
            Debug.Log("RLMesh: installed Reach Target demo at " + Dest);
        }

        static string SampleSource()
        {
            var info = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(ReachTargetSampleInstall).Assembly);
            if (info == null || string.IsNullOrEmpty(info.resolvedPath))
                return null;
            return Path.Combine(info.resolvedPath, "Samples~", "ReachTarget");
        }
    }
}
