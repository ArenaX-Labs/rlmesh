#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshReachTargetComponent.generated.h"

/**
 * Example env: move the owning actor in world space toward a 3D target.
 *
 * Each reset puts the actor back where play began. If TargetActor is set, the
 * goal is that actor's world location; otherwise a target offset is picked
 * uniformly in [-TargetRange, TargetRange]^3 (from the reset seed, when one is
 * given). The observation is the remaining XYZ offset to the target (Box of
 * shape [3]). The action is Box([-1, 1]^3): a direction that is normalized and
 * scaled by StepSize (near-zero actions stay put). The reward is minus the
 * remaining Euclidean distance over TargetRange; an episode terminates within
 * half a step of the target and truncates after MaxSteps.
 */
UCLASS(ClassGroup = (RLMesh), meta = (BlueprintSpawnableComponent))
class RLMESHUNREAL_API URLMeshReachTargetComponent : public URLMeshEnvComponent
{
	GENERATED_BODY()

public:
	/** Distance one step moves the actor, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float StepSize = 50.f;

	/** Farthest a random target is placed from the start, in units. Also scales reward. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float TargetRange = 1000.f;

	/** Steps before an episode truncates. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 200;

	/**
	 * Optional level actor used as the 3D goal (e.g. a Target box).
	 * When set, reset uses this actor's world location instead of a random target.
	 */
	UPROPERTY(EditAnywhere, Category = "RLMesh")
	TObjectPtr<AActor> TargetActor;

protected:
	virtual void BeginPlay() override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	FVector Remaining() const;
	double Distance() const;
	rlmesh::Value Observe() const;
	/** agent_xyz, target_xyz, distance, steps — extension-path leaves beyond the Box[3] obs. */
	rlmesh::InfoDict MakeInfo() const;

	FVector Start = FVector::ZeroVector;
	/** Goal as world-space offset from Start (TargetWorld - Start). */
	FVector TargetOffset = FVector::ZeroVector;
	int32 Steps = 0;
	FRandomStream Random;
};
