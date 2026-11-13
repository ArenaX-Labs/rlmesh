#include "RLMeshEnvComponent.h"

#include "Engine/Engine.h"
#include "HAL/IConsoleManager.h"
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
				"Run: python integrations/unreal/demos/reach_target/app.py --init-identity"),
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

void URLMeshEnvComponent::OnResetOptions(const std::optional<rlmesh::InfoDict>& /*Options*/)
{
}

void URLMeshEnvComponent::ApplyServingMaxFPS(float NewMaxFPS)
{
	const float Target = NewMaxFPS > 0.f ? NewMaxFPS : ServingMaxFPS;
	if (Target <= 0.f)
	{
		return;
	}
	if (GEngine)
	{
		if (!bHaveSavedMaxFPS)
		{
			SavedMaxFPS = GEngine->GetMaxFPS();
			bHaveSavedMaxFPS = true;
		}
		GEngine->SetMaxFPS(Target);
	}
	auto SetCvar = [](const TCHAR* Name, float Value)
	{
		if (IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(Name))
		{
			CVar->Set(Value, ECVF_SetByCode);
		}
	};
	SetCvar(TEXT("t.MaxFPS"), Target);
	SetCvar(TEXT("r.VSync"), 0.f);
	// Keep ticking when the editor is unfocused so Python lockstep does not stall.
	SetCvar(TEXT("t.IdleWhenNotForeground"), 0.f);
	SetCvar(TEXT("EditorPerformance.ThrottleCPUWhenNotForeground"), 0.f);
	UE_LOG(LogRLMesh, Log, TEXT("%s ServingMaxFPS=%.0f (lockstep can step up to ~that Hz)"), *EnvId,
		   Target);
}

void URLMeshEnvComponent::ApplyServingMaxFPSFromOptions(const std::optional<rlmesh::InfoDict>& Options)
{
	float FromOptions = 0.f;
	if (Options)
	{
		const auto It = Options->find("serving_max_fps");
		if (It != Options->end())
		{
			try
			{
				FromOptions = static_cast<float>(It->second.as_float());
			}
			catch (...)
			{
				try
				{
					FromOptions = static_cast<float>(It->second.as_int());
				}
				catch (...)
				{
					UE_LOG(LogRLMesh, Warning,
						   TEXT("%s reset option serving_max_fps was not a number"), *EnvId);
				}
			}
		}
	}
	if (FromOptions > 0.f)
	{
		ServingMaxFPS = FromOptions;
		ApplyServingMaxFPS(FromOptions);
	}
}

void URLMeshEnvComponent::RestoreServingMaxFPS()
{
	if (!bHaveSavedMaxFPS)
	{
		return;
	}
	if (GEngine)
	{
		GEngine->SetMaxFPS(SavedMaxFPS);
	}
	if (IConsoleVariable* MaxFPSCvar = IConsoleManager::Get().FindConsoleVariable(TEXT("t.MaxFPS")))
	{
		MaxFPSCvar->Set(SavedMaxFPS, ECVF_SetByCode);
	}
	bHaveSavedMaxFPS = false;
	UE_LOG(LogRLMesh, Log, TEXT("%s restored MaxFPS=%.0f"), *EnvId, SavedMaxFPS);
}

void URLMeshEnvComponent::BeginPlay()
{
	Super::BeginPlay();

	// Unreal code runs without exceptions, so none may leave this function.
	ApplyHostIdentityFromFile();
	ApplyServingMaxFPS();
	GameThread = std::make_shared<FRLMeshGameThread>();
	try
	{
		FRLMeshGameThreadEnv Env(
			GameThread, ObservationSpace(), ActionSpace(), std::string(TCHAR_TO_UTF8(*EnvId)),
			[this](std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
			{
				ApplyServingMaxFPSFromOptions(Options);
				OnResetOptions(Options);
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
	if (bRestoreMaxFPSOnEndPlay)
	{
		RestoreServingMaxFPS();
	}
	Super::EndPlay(EndPlayReason);
}
