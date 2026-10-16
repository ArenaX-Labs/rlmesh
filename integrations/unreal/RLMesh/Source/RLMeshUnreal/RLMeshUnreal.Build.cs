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
			// Link the import library; the DLL is staged beside the game binary.
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "rlmesh_capi.lib"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/rlmesh_capi.dll",
				Path.Combine(RLMeshDir, "bin", "rlmesh_capi.dll"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			// The loader looks for the versioned name in the library's SONAME
			// (librlmesh_capi.so.<ABI generation>), so stage that file.
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "librlmesh_capi.so"));
			foreach (string Versioned in Directory.GetFiles(LibDir, "librlmesh_capi.so.*"))
			{
				RuntimeDependencies.Add("$(BinaryOutputDir)/" + Path.GetFileName(Versioned), Versioned);
			}
		}
	}
}
