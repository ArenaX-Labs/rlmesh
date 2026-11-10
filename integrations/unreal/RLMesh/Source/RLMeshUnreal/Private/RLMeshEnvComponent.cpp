#include "RLMeshEnvComponent.h"

#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#include <exception>
#include <string>
#include <utility>

DEFINE_LOG_CATEGORY(LogRLMesh);

namespace
{
const TCHAR* HostIdentityEnvVar = TEXT("RLMESH_HOST_IDENTITY");
}

URLMeshEnvComponent::URLMeshEnvComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

FString URLMeshEnvComponent::DefaultHostIdentityPath()
{
#if PLATFORM_WINDOWS
	const FString Base = FPlatformMisc::GetEnvironmentVariable(TEXT("LOCALAPPDATA"));
	if (!Base.IsEmpty())
	{
		return FPaths::Combine(Base, TEXT("rlmesh"), TEXT("unreal_host_identity"));
	}
#endif
	return FPaths::Combine(FPlatformProcess::UserSettingsDir(), TEXT("rlmesh"),
						   TEXT("unreal_host_identity"));
}

bool URLMeshEnvComponent::ApplyHostIdentityFromPath(const FString& Path)
{
	FString Contents;
	if (!FFileHelper::LoadFileToString(Contents, *Path))
	{
		return false;
	}
	Contents.TrimStartAndEndInline();
	if (Contents.IsEmpty())
	{
		UE_LOG(LogRLMesh, Warning, TEXT("Host identity file %s is empty"), *Path);
		return false;
	}
	// Win32 process env — rlmesh_capi (Rust std::env) reads the same block.
	FPlatformMisc::SetEnvironmentVar(HostIdentityEnvVar, *Contents);
	UE_LOG(LogRLMesh, Log, TEXT("Loaded RLMESH_HOST_IDENTITY from %s"), *Path);
	return true;
}

void URLMeshEnvComponent::ApplyDefaultHostIdentity()
{
	const FString Path = DefaultHostIdentityPath();
	if (ApplyHostIdentityFromPath(Path))
	{
		return;
	}
	UE_LOG(LogRLMesh, Warning,
		   TEXT("No host identity at %s — Serving #keys will change each Play. "
				"Run: python integrations/unreal/reach_target_app.py --init-identity"),
		   *Path);
}

void URLMeshEnvComponent::ApplyHostIdentityFromFile()
{
	if (!HostIdentityFile.IsEmpty())
	{
		if (!ApplyHostIdentityFromPath(HostIdentityFile))
		{
			UE_LOG(LogRLMesh, Warning, TEXT("No host identity at %s"), *HostIdentityFile);
		}
		return;
	}
	// Default path already applied at module startup; refresh in case the file
	// was created after the editor launched.
	ApplyDefaultHostIdentity();
}

void URLMeshEnvComponent::BeginPlay()
{
	Super::BeginPlay();

	// Unreal code runs without exceptions, so none may leave this function.
	ApplyHostIdentityFromFile();
	GameThread = std::make_shared<FRLMeshGameThread>();
	try
	{
		FRLMeshGameThreadEnv Env(
			GameThread, ObservationSpace(), ActionSpace(), std::string(TCHAR_TO_UTF8(*EnvId)),
			[this](std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
			{
				return ResetEnv(Seed, Options);
			},
			[this](std::optional<rlmesh::Value> Action)
			{
				return StepEnv(Action);
			});
		auto Options = rlmesh::EnvServerOptions{}.with_address(std::string(TCHAR_TO_UTF8(*Address)));
		Server = std::make_unique<rlmesh::EnvServer>(std::move(Env), std::move(Options));
		Server->start();
		ServedAddress = UTF8_TO_TCHAR(Server->address().c_str());
		UE_LOG(LogRLMesh, Log, TEXT("Serving %s on %s"), *EnvId, *ServedAddress);
	}
	catch (const std::exception& Error)
	{
		Server.reset();
		GameThread.reset();
		UE_LOG(LogRLMesh, Error, TEXT("Could not serve %s on %s: %s"), *EnvId, *Address,
			   UTF8_TO_TCHAR(Error.what()));
	}
}

void URLMeshEnvComponent::TickComponent(float DeltaTime, ELevelTick TickType,
										FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (GameThread)
	{
		GameThread->Pump();
	}
}

void URLMeshEnvComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	// Fail any request waiting for a tick first, so shutdown does not wait on one.
	if (GameThread)
	{
		GameThread->Close();
	}
	if (Server)
	{
		try
		{
			Server->shutdown();
		}
		catch (const std::exception& Error)
		{
			UE_LOG(LogRLMesh, Warning, TEXT("Shutting down %s: %s"), *EnvId, UTF8_TO_TCHAR(Error.what()));
		}
		Server.reset();
	}
	GameThread.reset();
	ServedAddress.Reset();
	Super::EndPlay(EndPlayReason);
}
