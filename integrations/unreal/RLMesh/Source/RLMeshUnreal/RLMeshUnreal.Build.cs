using System.IO;
using UnrealBuildTool;

public class RLMeshUnreal : ModuleRules
{
	public RLMeshUnreal(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		// rlmesh.hpp reports every failure as a C++ exception.
		bEnableExceptions = true;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });

		// An unpacked RLMesh release package: include/, lib/ and, on Windows, bin/.
		string RLMeshDir = Path.Combine(PluginDirectory, "Source", "ThirdParty", "rlmesh");
		string IncludeDir = Path.Combine(RLMeshDir, "include");
		string LibDir = Path.Combine(RLMeshDir, "lib");
		if (!File.Exists(Path.Combine(IncludeDir, "rlmesh.hpp")))
		{
			throw new BuildException(
				"RLMesh: unpack an RLMesh release package into " + RLMeshDir + " (see the plugin README)");
		}
		PublicIncludePaths.Add(IncludeDir);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string ImportLib = Path.Combine(LibDir, "rlmesh_capi.lib");
			string Dll = Path.Combine(RLMeshDir, "bin", "rlmesh_capi.dll");
			if (!File.Exists(ImportLib) || !File.Exists(Dll))
			{
				throw new BuildException(
					"RLMesh: Win64 needs lib/rlmesh_capi.lib and bin/rlmesh_capi.dll under " + RLMeshDir);
			}
			// Link the import library; the DLL is staged beside the game binary.
			PublicAdditionalLibraries.Add(ImportLib);
			RuntimeDependencies.Add("$(BinaryOutputDir)/rlmesh_capi.dll", Dll);
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			string Shared = Path.Combine(LibDir, "librlmesh_capi.so");
			if (!File.Exists(Shared))
			{
				throw new BuildException(
					"RLMesh: Linux needs lib/librlmesh_capi.so under " + RLMeshDir);
			}
			// The loader looks for the versioned name in the library's SONAME
			// (librlmesh_capi.so.<ABI generation>), so stage that file.
			PublicAdditionalLibraries.Add(Shared);
			string[] Versioned = Directory.GetFiles(LibDir, "librlmesh_capi.so.*");
			if (Versioned.Length == 0)
			{
				throw new BuildException(
					"RLMesh: Linux needs a versioned lib/librlmesh_capi.so.* under " + RLMeshDir);
			}
			foreach (string PathToVersioned in Versioned)
			{
				RuntimeDependencies.Add("$(BinaryOutputDir)/" + Path.GetFileName(PathToVersioned), PathToVersioned);
			}
		}
	}
}
