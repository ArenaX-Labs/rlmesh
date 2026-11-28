#include "RLMeshEnvComponent.h"

#include "Components/SceneComponent.h"
#include "GameFramework/Actor.h"

#include <exception>
#include <string>
#include <utility>

DEFINE_LOG_CATEGORY(LogRLMesh);

URLMeshEnvComponent::URLMeshEnvComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void URLMeshEnvComponent::BeginPlay()
{
	Super::BeginPlay();

	USceneComponent* Root = GetOwner()->GetRootComponent();
	if (bMakeOwnerMovable && Root && Root->Mobility != EComponentMobility::Movable)
	{
		Root->SetMobility(EComponentMobility::Movable);
		UE_LOG(LogRLMesh, Log, TEXT("Set %s to Movable so %s can move it"), *GetOwner()->GetName(), *EnvId);
	}
	Random.GenerateNewSeed();

	GameThread = std::make_shared<FRLMeshGameThread>();
	try
	{
		FRLMeshGameThreadEnv Env(
			GameThread, ObservationSpace(), ActionSpace(), std::string(TCHAR_TO_UTF8(*EnvId)),
			[this](std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
			{
				if (Seed)
				{
					// FRandomStream takes an int32: fold in the high half so seeds differing only there differ.
					const uint64 Bits = static_cast<uint64>(*Seed);
					Random.Initialize(static_cast<int32>(Bits ^ (Bits >> 32)));
				}
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
		UE_LOG(LogRLMesh, Log, TEXT("Serving %s: %s"), *EnvId, UTF8_TO_TCHAR(Server->to_string().c_str()));
	}
	catch (const std::exception& Error)
	{
		Server.reset();
		GameThread.reset();
		UE_LOG(LogRLMesh, Error, TEXT("Could not serve %s: %s"), *EnvId, UTF8_TO_TCHAR(Error.what()));
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
