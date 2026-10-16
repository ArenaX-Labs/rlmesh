#pragma once

#include "CoreMinimal.h"
#include "Math/RandomStream.h"
#include "RLMeshEnvComponent.h"

#include "RLMeshReachTargetComponent.generated.h"

/**
 * Example env: move the owning actor along its world X axis to a target.
 *
 * Each reset puts the actor back where play began and picks a target up to
 * TargetRange away (from the reset seed, when one is given). The observation is
 * the offset to the target in units (a Box of shape [1]); the action is
 * Discrete(3): 0 moves back by StepSize, 1 stays, 2 moves forward. The reward is
 * minus the remaining distance over TargetRange; an episode terminates within
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

	/** Farthest a target is placed from the start, in units. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	float TargetRange = 1000.f;

	/** Steps before an episode truncates. */
	UPROPERTY(EditAnywhere, Category = "RLMesh", meta = (ClampMin = "1"))
	int32 MaxSteps = 200;

protected:
	virtual void BeginPlay() override;

	virtual rlmesh::spaces::Space ObservationSpace() const override;
	virtual rlmesh::spaces::Space ActionSpace() const override;
	virtual rlmesh::ResetResult ResetEnv(std::optional<int64_t> Seed,
										 const std::optional<rlmesh::InfoDict>& Options) override;
	virtual rlmesh::StepResult StepEnv(const std::optional<rlmesh::Value>& Action) override;

private:
	double Offset() const;
	rlmesh::Value Observe() const;

	FVector Start = FVector::ZeroVector;
	double Target = 0.0;
	int32 Steps = 0;
	FRandomStream Random;
};
