#pragma once

#include <memory>

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "RLMeshGameThreadEnv.h"

#include "RLMeshEnvComponent.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogRLMesh, Log, All);

/**
 * Serves game state as an RLMesh env while the game plays: from BeginPlay until
 * EndPlay, any RLMesh model can connect to Address and call reset and step.
 *
 * Subclass it in C++ and override ObservationSpace, ActionSpace, ResetEnv and
 * StepEnv. ResetEnv and StepEnv run on the game thread, from this component's
 * tick, so they may read and change the world; a step therefore takes at least
 * one frame. Throw rlmesh::RecoverableEnvironmentException from them to fail one
 * request while the client keeps its session.
 *
 * Loopback Serving invites include #keys derived from HostIdentity. The plugin
 * loads a static identity from HostIdentityFile (or
 * %LOCALAPPDATA%/rlmesh/unreal_host_identity) into RLMESH_HOST_IDENTITY before
 * EnvServer starts — generate it once with reach_target_app.py --init-identity.
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

	/**
	 * Optional path to a one-line RLMESH_HOST_IDENTITY secret file.
	 * Empty uses %LOCALAPPDATA%/rlmesh/unreal_host_identity (or the platform
	 * equivalent). Loaded into the process env before EnvServer starts.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	FString HostIdentityFile;

	/** The address the env is served on, or empty when it is not serving. */
	UFUNCTION(BlueprintCallable, Category = "RLMesh")
	FString GetServedAddress() const { return ServedAddress; }

	/** Load the default host-identity file into RLMESH_HOST_IDENTITY (module startup). */
	static void ApplyDefaultHostIdentity();

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
	static FString DefaultHostIdentityPath();
	static bool ApplyHostIdentityFromPath(const FString& Path);

	void ApplyHostIdentityFromFile();

	std::shared_ptr<FRLMeshGameThread> GameThread;
	std::unique_ptr<rlmesh::EnvServer> Server;
	FString ServedAddress;
};
