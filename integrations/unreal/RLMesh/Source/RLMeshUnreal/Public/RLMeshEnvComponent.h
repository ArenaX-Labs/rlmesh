#pragma once

#include <memory>

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "RLMeshGameThreadEnv.h"

#include "RLMeshEnvComponent.generated.h"

/**
 * Serves game state as an RLMesh env while the game plays: from BeginPlay until
 * EndPlay, any RLMesh model can connect to Address and call reset and step.
 *
 * Subclass it in C++ and override ObservationSpace, ActionSpace, ResetEnv and
 * StepEnv. ResetEnv and StepEnv run on the game thread, from this component's
 * tick, so they may read and change the world; a step therefore takes at least
 * one frame. Throw rlmesh::RecoverableEnvironmentException from them to fail one
 * request while the client keeps its session.
 */
UCLASS(Abstract)
class RLMESHUNREAL_API URLMeshEnvComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	URLMeshEnvComponent();

	/** Session participant address to serve at. Port 0 picks a free port; GetServedAddress returns it. */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	FString Address = TEXT("127.0.0.1:7447/env");

	/** Env id reported to clients. */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	FString EnvId = TEXT("Unreal-v0");

	/** The address the env is served on, or empty when it is not serving. */
	UFUNCTION(BlueprintCallable, Category = "RLMesh")
	FString GetServedAddress() const { return ServedAddress; }

	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
							   FActorComponentTickFunction* ThisTickFunction) override;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	virtual rlmesh::spaces::Space ObservationSpace() const
		PURE_VIRTUAL(URLMeshEnvComponent::ObservationSpace, return rlmesh::spaces::Discrete(1););
	virtual rlmesh::spaces::Space ActionSpace() const
		PURE_VIRTUAL(URLMeshEnvComponent::ActionSpace, return rlmesh::spaces::Discrete(1););
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed, const std::optional<rlmesh::InfoDict>& Options)
		PURE_VIRTUAL(URLMeshEnvComponent::ResetEnv, return rlmesh::ResetResult(););
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action)
		PURE_VIRTUAL(URLMeshEnvComponent::StepEnv, return rlmesh::StepResult(););

private:
	std::shared_ptr<FRLMeshGameThread> GameThread;
	std::unique_ptr<rlmesh::EnvServer> Server;
	FString ServedAddress;
};
