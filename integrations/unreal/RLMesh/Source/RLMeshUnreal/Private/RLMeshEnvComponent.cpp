#include "RLMeshEnvComponent.h"

#include <exception>
#include <string>
#include <utility>

DEFINE_LOG_CATEGORY_STATIC(LogRLMesh, Log, All);

URLMeshEnvComponent::URLMeshEnvComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void URLMeshEnvComponent::BeginPlay()
{
	Super::BeginPlay();

	// Unreal code runs without exceptions, so none may leave this function.
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
		UE_LOG(LogRLMesh, Error, TEXT("Could not serve %s on %s: %s"), *EnvId, *Address, UTF8_TO_TCHAR(Error.what()));
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
